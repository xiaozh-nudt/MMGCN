import torch
import torch.nn as nn
from typing import Optional

class ProbabilityMatrixCrossEntropyLossFunc(nn.Module):
    """
    二分类矩阵式损失（正样本：label==1；负样本：label==0）
    - 忽略所有 label < 0 的位置（验证/测试标记）
    - 对每个正样本 (g,d_pos) 在同一行 g 随机采一个负样本列 d_neg (label==0)
      计算 -log(p_pos) - log(1 - p_neg)，累加后返回（可按需改成 mean）
    - predict 会被 clamp 到 (1e-7, 1-1e-7)
    """
    def __init__(self):
        super(ProbabilityMatrixCrossEntropyLossFunc, self).__init__()

    @torch.no_grad()
    def _sample_neg_col(self, row_labels: torch.Tensor) -> Optional[int]:
        """
        在单行标签中随机采一个 label==0 的列；若不存在返回 None。
        row_labels: [D] on CPU/GPU
        """
        zero_cols = (row_labels == 0).nonzero(as_tuple=False).view(-1)
        if zero_cols.numel() == 0:
            return None
        idx = torch.randint(low=0, high=zero_cols.numel(), size=(1,), device=zero_cols.device)
        return int(zero_cols[idx].item())

    def forward(self, predict_adj_mats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        predict_adj_mats: [G, D]，模型输出在 [0,1] 的分数（会自动 clamp）
        labels:           [G, D]，值域 {1, 0, -1, -2, -3, -4}
                          - 1 : 训练正样本
                          - 0 : 未知/负样本（可被采为负）
                          - <0: 验证/测试集合标记，全部忽略
        """
        device = predict_adj_mats.device
        labels = labels.to(device)

        # 避免 log(0)
        pred = torch.clamp(predict_adj_mats, min=1e-7, max=1 - 1e-7)

        # 仅取训练正样本
        pos_coords = (labels == 1).nonzero(as_tuple=False)  # [N_pos, 2] -> (g, d_pos)
        if pos_coords.numel() == 0:
            # 没有训练正样本：返回 0（保持图形可微）
            return torch.zeros((), device=device)

        total_loss = torch.zeros((), device=device)

        # 遍历正样本，按行随机采一个负样本列
        for g, d_pos in pos_coords:
            row_labels = labels[g]  # [D]
            d_neg = self._sample_neg_col(row_labels)
            if d_neg is None:
                # 该行没有可采负样本（全为正或 <0），跳过
                continue

            p_pos = pred[g, d_pos]
            p_neg = pred[g, d_neg]
            total_loss = total_loss - torch.log(p_pos) - torch.log(1.0 - p_neg)

        # 如需平均： return total_loss / max(1, pos_coords.size(0))
        return total_loss
