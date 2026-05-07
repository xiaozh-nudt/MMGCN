import os
from configparser import ConfigParser, NoOptionError, NoSectionError
import sys

def _parse_bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Illegal boolean value: {v}")

class Config(object):
    def __init__(self):
        # ---- 先给出安全默认值（若 config.ini 缺失键，至少不至于 None） ----
        self.data_path = './data'
        self.data_save_path = './datasets'
        self.model_save_path = './models'
        self.ROC_save_path = './roc'
        self.res_save_path = './results'

        # 读取配置文件
        conn = ConfigParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError(config_path)

        conn.read(config_path)

        # ---- 路径相关 ----
        self.data_path = conn.get('path', 'data_path', fallback=self.data_path)
        self.mm_go_path = conn.get('path', 'mm_go_path', fallback=os.path.join(self.data_path, 'miRNA_functional_similarity.txt'))
        self.data_save_path = conn.get('path', 'data_save_path', fallback=self.data_save_path)
        self.model_save_path = conn.get('path', 'model_save_path', fallback=self.model_save_path)
        self.ROC_save_path = conn.get('path', 'ROC_save_path', fallback=self.ROC_save_path)
        self.res_save_path = conn.get('path', 'res_save_path', fallback=self.res_save_path)

        # ---- 数据规模/通道 ----
        self.gene = int(conn.get('dataset', 'gene_num'))
        self.gene_feature_channels = int(conn.get('dataset', 'gene_feature_channels'))
        self.disease = int(conn.get('dataset', 'disease_num'))
        self.disease_feature_channels = int(conn.get('dataset', 'disease_feature_channels'))

        # 兼容旧配置：training_mask（若无此项则给默认 0 或按需调整）
        self.training_mask = int(conn.get('dataset', 'training_mask', fallback='0'))

        # 测试集大小（正样本对数量）
        self.test_set = int(conn.get('dataset', 'test_set'))

        # ✅ 新增：验证集大小（正样本对数量，**可选**）
        # 若缺失则置为 None，数据构建器会按 10% 正样本自动回退
        try:
            self.valid_set = int(conn.get('dataset', 'valid_set'))
            if self.valid_set <= 0:
                raise ValueError("`valid_set` must be a positive integer.")
        except (NoOptionError, NoSectionError, ValueError):
            self.valid_set = None  # 让数据构建器走自动比例

        # ---- 训练超参 ----
        self.learning_rate = float(conn.get('parameter', 'learning_rate'))
        self.weight_decay = float(conn.get('parameter', 'weight_decay'))
        self.total_epoch = int(conn.get('parameter', 'total_epoch'))
        self.heads = int(conn.get('parameter', 'heads'))
        self.hidden_channels = int(conn.get('parameter', 'hidden_channels'))
        self.output_channels = int(conn.get('parameter', 'output_channels'))

        # ---- 断点续训 ----
        raw_ctt = conn.get('continue to train', 'continue_to_train', fallback='False')
        try:
            self.continue_to_train = _parse_bool(raw_ctt)
        except ValueError:
            print(f"'continue_to_train' in [config.ini] is illegal: {raw_ctt}")
            sys.exit(1)

        if self.continue_to_train:
            try:
                self.restart_from = int(conn.get('continue to train', 'restart_from'))
            except Exception:
                print("When 'continue_to_train' is True, 'restart_from' must be provided and integer.")
                sys.exit(1)

        # ---- 创建必要目录，避免后续保存失败 ----
        for p in [self.data_save_path, self.model_save_path, self.ROC_save_path, self.res_save_path]:
            try:
                os.makedirs(p, exist_ok=True)
            except Exception as e:
                print(f"Failed to create directory '{p}': {e}")
                sys.exit(1)
