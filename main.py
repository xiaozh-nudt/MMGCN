import os
import sys
import torch
import copy
import pickle
import argparse
import scipy.sparse as sp

from model2 import LrGNN
from logger import Logger
from configloader import Config
from statistic import statistic
from lossfunction import ProbabilityMatrixCrossEntropyLossFunc
from dataloader import DataLoader
from AUCscore import AUC_score

# ---------------------- 公共工具 ---------------------- #
# GPT运行
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # CPU运行
# def get_device():
#     return torch.device('cpu')

def build_data(device):
    """加载三路数据，并上设备"""
    dl = DataLoader()
    data_go, data_seq, data_sim = dl.get_data()
    data_go  = data_go.to(device)
    data_seq = data_seq.to(device)
    data_sim = data_sim.to(device)
    return data_go, data_seq, data_sim

def build_model(device):
    model = LrGNN().to(device)
    return model

def save_pickle(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(pickle.dumps(obj))

# ---------------------- 训练流程 ---------------------- #
def train_main():
    device = get_device()
    config = Config()

    # 数据
    data_go, data_seq, data_sim = build_data(device)

    # 只使用验证集做选模；严禁训练期使用测试集
    val_coords_dict = copy.deepcopy(getattr(data_go, 'val_set_coords_dict', {'val_true': [], 'val_false': []}))
    if len(val_coords_dict.get('val_true', [])) == 0 or len(val_coords_dict.get('val_false', [])) == 0:
        print("Warning: 验证集为空；请确认数据构建是否包含 val 划分。")

    # 清掉 test_set_coords_dict，防止训练阶段误用
    (data_go.test_set_coords_dict).clear()
    (data_seq.test_set_coords_dict).clear()
    (data_sim.test_set_coords_dict).clear()

    # 模型与优化
    model = build_model(device)
    if not os.path.exists(config.model_save_path):
        print('The model_save_path does not exist!')
        sys.exit(1)

    start_epoch = 0
    if config.continue_to_train:
        ckpt = os.path.join(config.model_save_path, f"train_{config.restart_from}epochs.model")
        if not os.path.exists(ckpt):
            print(f"The model file '{ckpt}' does not exist!")
            sys.exit(1)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        start_epoch = config.restart_from + 1

    logger = Logger(writetofile=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = ProbabilityMatrixCrossEntropyLossFunc()

    best_val_auc = 0.0
    best_snapshot_path = None
    model_num = 0
    model.train()

    for epoch in range(start_epoch, config.total_epoch):
        optimizer.zero_grad()

        # 前向
        predict = model(data_go, data_seq, data_sim)

        # 训练损失（忽略 <0 的 val/test 标签）
        loss = loss_fn(predict, data_go.label)  # 直接传 data_go.label，不需要映射

        # ========= 验证集 AUC（用于选模） =========
        # AUC_score 需要 {'test_true','test_false'} 结构，这里做临时映射
        val_as_test = {
            'test_true':  val_coords_dict.get('val_true', []),
            'test_false': val_coords_dict.get('val_false', [])
        }
        val_auc, val_label_dump, val_res_dump = AUC_score(predict, val_as_test)

        # 选模策略：验证 AUC 提升且超过阈值时保存
        if val_auc > 0.92 and val_auc > best_val_auc:
            model_num += 1
            snap_path = os.path.join(config.model_save_path, f'{epoch}_epochs_valAUC{val_auc:.6f}.model')
            torch.save(model.state_dict(), snap_path)
            best_snapshot_path = snap_path

            save_pickle(os.path.join(config.res_save_path, f'{epoch}epochs_valAUC{val_auc:.6f}_label.pkl'), val_label_dump)
            save_pickle(os.path.join(config.res_save_path, f'{epoch}epochs_valAUC{val_auc:.6f}_res.pkl'),   val_res_dump)

            if model_num > 20:
                print("模型保存次数超过 20 次，提前停止。")
                break

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        logger.debug('training~epoch:', epoch, ' loss =', float(loss), ' ValAUC = ', val_auc, ' BestVal = ', best_val_auc)

        loss.backward()
        optimizer.step()

        # ========= 每 10 轮：输出验证命中率统计 =========
        if epoch % 10 == 0 and epoch != 0:
            model.eval()
            coo = sp.coo_matrix((predict.detach().cpu()).numpy())
            logger.debug('predict:', epoch, '\n', predict, '\nnonzero elements:', len(coo.data))

            val_correct_pos = 0
            val_correct_neg = 0
            for g, d in val_coords_dict.get('val_true', []):
                if predict[g][d] >= 0.5:
                    val_correct_pos += 1
            for g, d in val_coords_dict.get('val_false', []):
                if predict[g][d] < 0.5:
                    val_correct_neg += 1

            val_stat = statistic(val_correct_pos, val_correct_neg,
                                 len(val_coords_dict.get('val_true', [])),
                                 len(val_coords_dict.get('val_false', [])))
            logger.info(epoch, '@', 'VAL', val_stat)
            model.train()

    print("[TRAIN] Finished. Best Val AUC =", best_val_auc)
    if best_snapshot_path:
        print("[TRAIN] Best snapshot:", best_snapshot_path)

if __name__ == "__main__":
    # 直接运行训练，无需命令行参数
    train_main()
