import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def AUC_score(predict, test_coords_dict):
    """
    计算二分类 AUC：
    - predict: torch.Tensor [G, D]，模型输出的分数矩阵
    - test_coords_dict: {'test_true': [(g,d), ...], 'test_false': [(g,d), ...]}
      坐标元素可为 list/tuple/np.array/torch.Tensor，内部会统一为 int

    返回:
        auc_score (float 或 np.nan), label(np.ndarray), res(np.ndarray)
    """
    # ---- 基本检查 ----
    if not isinstance(test_coords_dict, dict):
        raise TypeError("test_coords_dict 必须是 dict。")
    if 'test_true' not in test_coords_dict or 'test_false' not in test_coords_dict:
        raise KeyError("test_coords_dict 必须包含 'test_true' 和 'test_false' 两个键。")

    # ---- 预测转为 numpy ----
    if isinstance(predict, torch.Tensor):
        try:
            predict_np = predict.detach().cpu().numpy()
        except Exception:
            predict_np = predict.cpu().detach().numpy()
    else:
        predict_np = np.asarray(predict)
    if predict_np.ndim != 2:
        raise ValueError(f"predict 需要是二维矩阵，当前形状为 {predict_np.shape}")

    G, D = predict_np.shape

    # ---- 安全取值的辅助函数 ----
    def _to_int(x):
        # 兼容 torch/np/int
        if isinstance(x, (np.generic,)):
            return int(x.item())
        if isinstance(x, torch.Tensor):
            return int(x.item())
        return int(x)

    def _gather_scores(coords, label_value):
        scores, labels = [], []
        for c in coords:
            # c 可能是 [g, d] / (g,d) / np.array([...]) / torch.tensor([...])
            try:
                g = _to_int(c[0])
                d = _to_int(c[1])
            except Exception:
                # 坐标格式异常
                continue
            if g < 0 or g >= G or d < 0 or d >= D:
                # 越界坐标，跳过但提示
                # 也可以 print 警告，但这里静默跳过，避免训练时刷屏
                continue
            v = predict_np[g, d]
            scores.append(v)
            labels.append(label_value)
        return scores, labels

    # ---- 收集正负样本分数与标签 ----
    pos_scores, pos_labels = _gather_scores(test_coords_dict.get('test_true', []), 1)
    neg_scores, neg_labels = _gather_scores(test_coords_dict.get('test_false', []), 0)

    res = np.array(pos_scores + neg_scores, dtype=float)
    label = np.array(pos_labels + neg_labels, dtype=float)

    # ---- 去除 NaN/Inf ----
    mask = np.isfinite(res) & np.isfinite(label)
    res = res[mask]
    label = label[mask]

    # ---- 基本有效性检查 ----
    if res.size == 0 or label.size == 0:
        # 返回 nan，保持返回签名不变
        return float('nan'), label, res

    # ---- 只有单类时的容错 ----
    # roc_auc_score 在只有 0 或 1 时会报错，这里返回 nan 更符合评估语义
    uniq = np.unique(label)
    if uniq.size < 2:
        # 可选：返回 0.5 当作随机水平
        # return 0.5, label, res
        return float('nan'), label, res

    # ---- 计算 AUC ----
    auc_score = roc_auc_score(label, res)
    return float(auc_score), label, res
