import torch  # 导入PyTorch库
import csv  # 导入CSV处理库
import os  # 导入操作系统相关库
import sys  # 导入系统相关库
import pickle  # 导入序列化库
import random  # 导入随机数生成库
import copy  # 导入复制库
import processing  # 导入处理模块
import numpy as np  # 导入NumPy库
import scipy.sparse as sp  # 导入SciPy稀疏矩阵库
import torch_geometric.transforms as T  # 导入PyTorch Geometric转换库
from configloader import Config  # 导入配置加载器
from torch_geometric.data import HeteroData  # 导入异构图数据结构

from math import ceil

random.seed(15)  # 设置随机种子，保证结果可复现
os.chdir(os.path.dirname(__file__))  # 切换当前工作目录到文件所在目录
config = Config()  # 初始化配置


def read_csv(path):
    """读取CSV为FloatTensor"""
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.FloatTensor(md_data)


def read_txt(path):
    """读取txt为FloatTensor（以空格分割）"""
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return torch.FloatTensor(md_data)


def read_txt1(path):
    """读取txt为FloatTensor（稳健实现）"""
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split()
            row_vals = []
            for k in line:
                row_vals.append(float(k))
            md_data.append(row_vals)
        md_data = np.array(md_data)
        return torch.FloatTensor(md_data)


def get_edge_index(matrix):
    """从矩阵中获取非零元素位置作为边索引"""
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def prepare_data(mm_path, md_path, dd_path):
    """准备数据集字典"""
    dataset = dict()

    dataset['label'] = read_txt1(md_path)

    md_matrix = read_txt1(md_path)
    md_edge_index = get_edge_index(md_matrix)  # miRNA-疾病
    dataset['md'] = md_edge_index

    dd_matrix = read_txt1(dd_path)
    dd_edge_index = get_edge_index(dd_matrix)  # 疾病-疾病
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}

    mm_matrix = read_txt1(mm_path)
    mm_edge_index = get_edge_index(mm_matrix)  # miRNA-miRNA
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}
    return dataset


class Databuilder(object):
    """
    数据构建器：三划分（train/val/test），训练图仅含训练边，避免泄露
    """
    def __init__(self, mm_path, md_path, dd_path):
        super(Databuilder, self).__init__()
        self.dataset = prepare_data(mm_path, md_path, dd_path)
        return

    def load_adj_mats(self):
        self.adj_mats = {
            'gene': copy.deepcopy(self.dataset['mm']['edge_index']),
            'gene_disease': copy.deepcopy(self.dataset['md']),
            'disease': copy.deepcopy(self.dataset['dd']['edge_index']),
        }

    def load_features(self):
        self.feat = {
            'gene': copy.deepcopy(self.dataset['mm']['data']),
            'disease': copy.deepcopy(self.dataset['dd']['data']),
        }

    def load_label(self):
        self.label = self.dataset['label']  # [num_gene, num_disease]

    def build_data(self):
        self.load_adj_mats()
        self.load_features()
        self.load_label()

        self.data = HeteroData()
        self.data['gene'].x = self.feat['gene'].contiguous()
        self.data['disease'].x = self.feat['disease'].contiguous()

        # 原始异构边（先存，再做无向）
        self.data['gene','gene_to_disease_edge','disease'].edge_index = self.adj_mats['gene_disease'].contiguous()
        self.data['gene','gene_to_gene_edge','gene'].edge_index = self.adj_mats['gene'].contiguous()
        self.data['disease','disease_to_disease_edge','disease'].edge_index = self.adj_mats['disease'].contiguous()

        self.data = T.ToUndirected()(self.data)  # 自动生成反向关系
        # self.data = T.AddSelfLoops()(self.data)

        self.data.label = self.label  # [G, D]
        return

    def _ensure_perm(self, num_edges):
        """
        获取或生成边的随机排列；保证可复现。
        """
        perm_path = os.path.join(config.data_save_path, 'perm.pkl')
        if os.path.exists(perm_path):
            with open(perm_path, 'rb') as file:
                perm = pickle.loads(file.read())
            # 如果历史 perm 长度与当前不一致，重新生成（兼容不同相似度矩阵边数）
            if isinstance(perm, torch.Tensor) and perm.numel() == num_edges:
                return perm
        # 生成新的 perm
        perm = torch.randperm(num_edges)
        with open(perm_path, 'wb') as f:
            f.write(pickle.dumps(perm))
        return perm

    def _draw_negatives(self, target_count, forbidden_marks):
        """
        从 label 中采样负样本 (label==0)，数量 target_count。
        forbidden_marks: 集合，包含不能作为负样本的特殊标记值（如已被占用的 -1/-2/-3/-4）。
        返回：list[np.array([row, col])]
        """
        negatives = []
        used = set()  # 防止重复
        G, D = self.data.label.size(0), self.data.label.size(1)
        while len(negatives) < target_count:
            row = random.randint(0, G - 1)
            col = random.randint(0, D - 1)
            if (row, col) in used:
                continue
            val = self.data.label[row][col].item()
            if val == 0:  # 仅能采样真实为0的位置
                used.add((row, col))
                negatives.append(np.array([row, col]))
            else:
                # 若被标记为任何正/负或异常，跳过
                if val in forbidden_marks:
                    continue
                else:
                    continue
        return negatives

    def _split_sizes(self, num_pos_edges):
        """
        计算 test / valid / train 三者正样本数量。
        - 测试集：使用 config.test_set（与原逻辑一致，理解为“正样本对”数量）
        - 验证集：优先使用 config.valid_set；若不存在，取 ceil(0.1 * num_pos_edges)
        """
        num_test = int(getattr(config, 'test_set', 0))
        if num_test < 0 or num_test >= num_pos_edges:
            raise ValueError(f"config.test_set={num_test} 非法或过大，正样本数={num_pos_edges}")

        # valid_set 优先来自配置；否则按 10%（至少为 1）
        if hasattr(config, 'valid_set'):
            num_valid = int(getattr(config, 'valid_set'))
            if num_valid <= 0:
                raise ValueError("config.valid_set 必须为正整数")
        else:
            num_valid = max(1, ceil(0.1 * num_pos_edges))

        if num_test + num_valid >= num_pos_edges:
            # 留足训练样本
            num_valid = max(1, min(num_valid, num_pos_edges - num_test - 1))

        num_train = num_pos_edges - num_test - num_valid
        return num_train, num_valid, num_test

    def traindata(self):
        """
        构建训练图，并生成 train/val/test 三集合坐标（含正负）。
        训练图仅保留训练边，验证/测试边不出现在训练图中，避免泄露。
        """
        traindata = copy.deepcopy(self.data)

        # 拿到有向边（ToUndirected后包含正向与反向）
        row_g_d, col_g_d = traindata.edge_index_dict['gene', 'gene_to_disease_edge', 'disease']
        row_d_g, col_d_g = traindata.edge_index_dict['disease', 'rev_gene_to_disease_edge', 'gene']

        # 清空两类异构边，稍后仅放回训练边
        traindata['gene', 'gene_to_disease_edge', 'disease'].edge_index = None
        traindata['disease', 'rev_gene_to_disease_edge', 'gene'].edge_index = None

        # 仅以 gene->disease 的边数为正样本数基准
        num_pos_edges = row_g_d.size(0)
        if num_pos_edges != row_d_g.size(0):
            raise RuntimeError("正向与反向边数量不一致，数据异常。")

        # 复现性：统一打散
        perm = self._ensure_perm(num_pos_edges)
        row_g_d, col_g_d = row_g_d[perm], col_g_d[perm]
        row_d_g, col_d_g = row_d_g[perm], col_d_g[perm]

        # 三划分大小
        num_train, num_valid, num_test = self._split_sizes(num_pos_edges)

        # --- 训练边 ---
        train_r, train_c = row_g_d[num_valid + num_test:], col_g_d[num_valid + num_test:]
        traindata['gene', 'gene_to_disease_edge', 'disease'].edge_index = torch.stack([train_r, train_c], dim=0).contiguous()

        train_r_rev, train_c_rev = row_d_g[num_valid + num_test:], col_d_g[num_valid + num_test:]
        traindata['disease', 'rev_gene_to_disease_edge', 'gene'].edge_index = torch.stack([train_r_rev, train_c_rev], dim=0).contiguous()

        # --- 验证正样本坐标 ---
        val_r, val_c = row_g_d[num_test:num_test + num_valid], col_g_d[num_test:num_test + num_valid]
        val_coords = torch.stack([val_r, val_c], dim=0).t()

        # --- 测试正样本坐标 ---
        test_r, test_c = row_g_d[:num_test], col_g_d[:num_test]
        test_coords = torch.stack([test_r, test_c], dim=0).t()

        # 标注与保存坐标：True 部分
        traindata.val_set_coords_dict = dict()
        traindata.test_set_coords_dict = dict()
        traindata.val_set_coords_dict['val_true'] = []
        traindata.test_set_coords_dict['test_true'] = []

        for coord in val_coords:
            if traindata.label[coord[0]][coord[1]] == 1:
                traindata.label[coord[0]][coord[1]] = -3  # 验证集正样本
                traindata.val_set_coords_dict['val_true'].append(np.array(coord))
            else:
                raise Exception('Error in build traindata: val_true not 1.')
        for coord in test_coords:
            if traindata.label[coord[0]][coord[1]] == 1:
                traindata.label[coord[0]][coord[1]] = -1  # 测试集正样本
                traindata.test_set_coords_dict['test_true'].append(np.array(coord))
            else:
                raise Exception('Error in build traindata: test_true not 1.')

        # 负样本采样：确保不与任何已占用位置重叠
        # 已占用标记集合
        forbidden_marks = {-1, -2, -3, -4}

        # 先采样验证负样本
        traindata.val_set_coords_dict['val_false'] = self._draw_negatives(num_valid, forbidden_marks)
        for row, col in traindata.val_set_coords_dict['val_false']:
            # 双重保障：只能从 0 变为 -4
            if traindata.label[row][col] == 0:
                traindata.label[row][col] = -4
        # 再采样测试负样本
        traindata.test_set_coords_dict['test_false'] = self._draw_negatives(num_test, forbidden_marks)
        for row, col in traindata.test_set_coords_dict['test_false']:
            if traindata.label[row][col] == 0:
                traindata.label[row][col] = -2

        # 保存划分信息，便于审计复现
        split_info = {
            'num_pos_edges_total': int(num_pos_edges),
            'num_train_pos': int(num_train),
            'num_valid_pos': int(num_valid),
            'num_test_pos': int(num_test),
        }
        split_path = os.path.join(config.data_save_path, 'split_info.pkl')
        with open(split_path, 'wb') as f:
            f.write(pickle.dumps(split_info))

        return traindata

    def data_save(self, save_path):
        """构建并保存（包含训练图与 val/test 坐标与标记）"""
        self.build_data()
        with open(save_path, 'wb') as data_pkl:
            data_pkl.write(pickle.dumps(self.traindata()))


# ========= 可选：一个简单的验证/测试评估辅助函数 =========
def evaluate_on_coords(predict_fn, data: HeteroData, coords: list):
    """
    使用你模型的打分函数 predict_fn 在指定 (gene_idx, disease_idx) 坐标上评估。
    - predict_fn: callable，签名为 predict_fn(g_idx_tensor, d_idx_tensor) -> scores(float tensor)
    - data: 训练阶段使用的 HeteroData（含 data.label）
    - coords: list[np.array([g, d])]（如 data.val_set_coords_dict['val_true'] + ['val_false']）

    返回：字典，含 AUC、Average Precision（需要 sklearn 可用时生效；若不可用，仅返回简单命中率）
    """
    import numpy as np
    import torch

    pos = [c for c in coords if isinstance(c, np.ndarray)]
    g_idx = torch.tensor([int(c[0]) for c in pos], dtype=torch.long)
    d_idx = torch.tensor([int(c[1]) for c in pos], dtype=torch.long)

    # 标签：根据 data.label 中 -3/-4（或 -1/-2）来设定
    y = []
    for (g, d) in zip(g_idx.tolist(), d_idx.tolist()):
        val = data.label[g][d].item()
        if val in (-3, -1):  # 验证/测试的正样本标记
            y.append(1)
        elif val in (-4, -2):  # 验证/测试的负样本标记
            y.append(0)
        else:
            # 若传入混合坐标，这里退化为真实标签
            y.append(1 if val == 1 else 0)
    y = torch.tensor(y, dtype=torch.float32)

    # 预测分数（越大越可能为正）
    with torch.no_grad():
        scores = predict_fn(g_idx, d_idx).view(-1).float().cpu()

    # 计算指标
    metrics = {}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        metrics['AUC'] = float(roc_auc_score(y.numpy(), scores.numpy()))
        metrics['AP'] = float(average_precision_score(y.numpy(), scores.numpy()))
    except Exception:
        # 兜底：简单命中率（按0.5阈值）
        preds = (scores >= 0.5).float()
        acc = float((preds == y).float().mean().item())
        metrics['Hit@0.5'] = acc
    return metrics
# ========= 评估辅助函数结束 =========


# =================== 路径配置 ===================
mm_go_path = config.data_path + '/miRNA_functional_similarity.txt'
mm_seq_path = config.data_path + '/miRNA_sequence_similarity.txt'
mm_sim_path = config.data_path + '/miRNA_semantic_similarity.txt'
md_path = config.data_path + '/m_d_associations.txt'
dd_path = config.data_path + '/disease_semantic_similarity.txt'


# =================== 构建并保存 ===================
# 使用功能相似性数据
print("正在构建并保存功能相似性数据...")
go_databuilder = Databuilder(mm_go_path, md_path, dd_path)
go_databuilder.data_save(config.data_save_path + '/data_go.pkl')
print("功能相似性数据保存完成。")

# 使用序列相似性数据
print("正在构建并保存序列相似性数据...")
seq_databuilder = Databuilder(mm_seq_path, md_path, dd_path)
seq_databuilder.data_save(config.data_save_path + '/data_seq.pkl')
print("序列相似性数据保存完成。")

# 使用语义相似性数据
print("正在构建并保存语义相似性数据...")
sim_databuilder = Databuilder(mm_sim_path, md_path, dd_path)
sim_databuilder.data_save(config.data_save_path + '/data_sim.pkl')
print("语义相似性数据保存完成。")
