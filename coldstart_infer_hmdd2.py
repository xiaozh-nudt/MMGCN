# -*- coding: utf-8 -*-
"""
coldstart_infer_hmdd2.py
- 只基于 HMDD 2.0
- 只加载已训练模型
- 对指定疾病做“冷启动”：掩蔽其标签 + 删除 gene<->disease 边
- 推理并导出 miRNA 排名（全量 & Top-K），可选合并功能注释
"""

import os
import sys
import csv
import json
import re
import math
import unicodedata
from typing import List, Optional, Dict

import numpy as np
import torch
from configparser import ConfigParser

# 你的工程内模块
from model2 import LrGNN
from dataloader import DataLoader
from configloader import Config as TrainConfig

# ------------------ 基础工具 ------------------ #
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_miRNA_list_2(path_xlsx: str) -> List[str]:
    """HMDD 2.0 的 miRNA 行序（训练时使用），来自 miRNA_name.xlsx"""
    import pandas as pd
    df = pd.read_excel(path_xlsx, header=None)
    name_col = int(df.notna().sum().idxmax())
    return df[name_col].dropna().astype(str).str.strip().tolist()

# ------------------ 名称归一化（用于匹配疾病/miRNA） ------------------ #
def _strip_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("\u2010", "-").replace("\u2011", "-")
           .replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-"))
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def normalize_mirna(name: str,
                    *,
                    casefold: bool = True,
                    unify_hsa_prefix: bool = True,
                    drop_species_prefix: bool = False,
                    drop_asterisk: bool = True,
                    unify_mir_token: bool = True,
                    collapse_3p_5p: bool = False) -> str:
    s = _strip_unicode(name)
    if casefold:
        s = s.lower()
    if drop_asterisk:
        s = s.replace("*", "")
    if unify_mir_token:
        s = re.sub(r"\bmirna\b", "mir", s)
        s = re.sub(r"\bmir\b", "mir", s)
        s = re.sub(r"\bhsa\-mir\b", "hsa-mir", s)
    if unify_hsa_prefix:
        s = re.sub(r"^(hsa[_\-:\s]*)?(mir)", r"hsa-mir", s)
        s = re.sub(r"hsa[\s_:]*mir", "hsa-mir", s)
    if drop_species_prefix:
        s = re.sub(r"^hsa\-", "", s)
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    if collapse_3p_5p:
        s = re.sub(r"-(3p|5p)\b", "", s)
    return s

def normalize_disease(name: str,
                      *,
                      casefold: bool = True,
                      strip_punct: bool = True) -> str:
    s = _strip_unicode(name)
    if casefold:
        s = s.lower()
    if strip_punct:
        s = re.sub(r"[.,;:]+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s

def build_normalized_index(names: List[str], *, is_mirna: bool, **norm_kwargs):
    norm2idx: Dict[str, int] = {}
    idx2norm: Dict[int, str] = {}
    for i, nm in enumerate(names):
        if is_mirna:
            norm = normalize_mirna(nm, **norm_kwargs)
        else:
            norm = normalize_disease(nm, **norm_kwargs)
        if norm not in norm2idx:
            norm2idx[norm] = i
            idx2norm[i] = norm
    return norm2idx, idx2norm

NORM_OPTS_MIR = dict(
    unify_hsa_prefix=True,
    drop_species_prefix=False,
    drop_asterisk=True,
    unify_mir_token=True,
    collapse_3p_5p=False,
)
NORM_OPTS_DIS = dict(strip_punct=True)

# ------------------ 删边（围绕目标疾病） ------------------ #
def remove_md_edges_around_disease(data, d_col: int):
    """删除异质图里所有 gene<->disease 涉及 d_col 的边"""
    if not hasattr(data, "edge_index_dict"):
        return
    eid = data.edge_index_dict
    for k in list(eid.keys()):
        if len(k) != 3:
            continue
        s, rel, t = k
        edge = eid[k]
        if s == 'gene' and t == 'disease':
            keep = (edge[1] != d_col)
            eid[k] = edge[:, keep]
        elif s == 'disease' and t == 'gene':
            keep = (edge[0] != d_col)
            eid[k] = edge[:, keep]

# ------------------ 主流程 ------------------ #
def main():
    # 读取配置
    cp = ConfigParser()
    cfg_path = os.path.join(os.path.dirname(__file__), "config.ini")
    if not os.path.isfile(cfg_path):
        print(f"[ERROR] 未找到配置文件：{cfg_path}")
        sys.exit(1)
    cp.read(cfg_path, encoding="utf-8")

    # 兼容 TrainConfig（如果你的工程需要 res/model 的路径等）
    tcfg = TrainConfig()

    # [inference]
    disease_name   = cp.get("inference", "disease_name", fallback="").strip()
    mirna_2_list   = cp.get("inference", "mirna_2_list", fallback="").strip()
    disease_2_list = cp.get("inference", "disease_2_list", fallback="").strip()
    model_filename = cp.get("inference", "model_filename", fallback="").strip()
    output_dir     = cp.get("inference", "output_dir", fallback="").strip()
    topk           = cp.getint("inference", "topk", fallback=50)
    normalize_flag = cp.getboolean("inference", "normalize_names", fallback=True)
    include_ann    = cp.getboolean("inference", "include_annotations", fallback=False)
    ann_file       = cp.get("inference", "annotations_file", fallback="").strip()

    if not disease_name:
        print("[ERROR] 请在 [inference] 中提供 disease_name。")
        sys.exit(1)
    if not (os.path.isfile(mirna_2_list) and os.path.isfile(disease_2_list)):
        print("[ERROR] HMDD 2.0 名单路径有误（mirna_2_list / disease_2_list）。")
        sys.exit(1)
    if not model_filename:
        print("[ERROR] model_filename 未提供。")
        sys.exit(1)

    # 模型路径解析：支持纯文件名（拼接到 TrainConfig.model_save_path），或直接给出完整路径
    if os.path.isabs(model_filename) or os.path.isfile(model_filename):
        model_path = model_filename
    else:
        # 优先 [path].model_save_path（若你在 TrainConfig 里也有同名字段，它们通常一致）
        base_dir = getattr(tcfg, "model_save_path", None) or cp.get("path", "model_save_path", fallback=".")
        model_path = os.path.join(base_dir, model_filename)
    if not os.path.isfile(model_path):
        print(f"[ERROR] 模型文件不存在：{model_path}")
        sys.exit(1)

    # ----------- 修改后的 output_dir 逻辑 -----------
    if (not output_dir) or (output_dir.lower() == "auto"):
        safe_disease = disease_name.replace(" ", "_")
        output_dir = os.path.join("./result/case_report", f"coldstart_{safe_disease}")
    ensure_dir(output_dir)
    # -------------------------------------------

    # 读取 HMDD2.0 名单
    mi2_names = load_miRNA_list_2(mirna_2_list)
    di2_names = read_lines(disease_2_list)
    D = len(di2_names)

    # 疾病列号（规范化匹配优先）
    if normalize_flag:
        di2_norm_map, _ = build_normalized_index(di2_names, is_mirna=False, **NORM_OPTS_DIS)
        dn_norm = normalize_disease(disease_name, **NORM_OPTS_DIS)
        if dn_norm not in di2_norm_map:
            print(f"[ERROR] disease_name 规范化后在 2.0 疾病清单中找不到：{disease_name}")
            sys.exit(1)
        d_col = di2_norm_map[dn_norm]
        disease_name_out = di2_names[d_col]  # 回写为 2.0 原名
    else:
        try:
            d_col = di2_names.index(disease_name)
        except ValueError:
            print(f"[ERROR] disease_name 在 2.0 疾病清单中找不到：{disease_name}")
            sys.exit(1)
        disease_name_out = disease_name

    print(f"[INFO] 目标疾病: '{disease_name_out}' (列号: {d_col})")

    # 载入数据与设备
    device = get_device()
    print(f"[INFO] 使用设备: {device}")
    dl = DataLoader()
    data_go, data_seq, data_sim = dl.get_data()

    # 冷启动：屏蔽标签 + 删边（按你原有规则）
    for data in [data_go, data_seq, data_sim]:
        if hasattr(data, "label"):
            data.label[:, d_col] = -9
        remove_md_edges_around_disease(data, d_col)

    # 上设备
    data_go  = data_go.to(device)
    data_seq = data_seq.to(device)
    data_sim = data_sim.to(device)

    # 推理
    model = LrGNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(data_go, data_seq, data_sim)  # [G, D]
        # 某些模型可能输出 float16；为了稳定 CSV 导出，转 float32
        pred = pred.float()

    # 取该疾病列的分数并排序
    if pred.dim() != 2 or pred.size(1) != D:
        print(f"[WARN] 预测维度与疾病数不一致：pred.shape={tuple(pred.shape)}, D={D}")
    scores = pred[:, d_col].detach().cpu().numpy().reshape(-1)
    order = np.argsort(-scores)  # 降序
    G = len(order)

    # 构建导出表
    full_rows = []
    for r, idx in enumerate(order, 1):
        mi_name = mi2_names[idx] if 0 <= idx < len(mi2_names) else f"miRNA_{idx}"
        full_rows.append([disease_name_out, int(idx), mi_name, float(scores[idx])])

    # 写 full_rank.csv
    full_csv = os.path.join(output_dir, "full_rank.csv")
    with open(full_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["disease_name", "miRNA_index", "miRNA_name", "score"])
        w.writerows(full_rows)
    print(f"[INFO] 导出完毕：{full_csv}")

    # 写 topk_rank.csv（如需要）
    topk_csv = None
    if isinstance(topk, int) and topk > 0:
        topk_eff = max(1, min(topk, G))
        topk_rows = full_rows[:topk_eff]
        topk_csv = os.path.join(output_dir, "topk_rank.csv")
        with open(topk_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["disease_name", "miRNA_index", "miRNA_name", "score"])
            w.writerows(topk_rows)
        print(f"[INFO] 导出完毕：{topk_csv}")

    # （可选）合并功能注释
    ann_csv = None
    if include_ann:
        if not ann_file or not os.path.isfile(ann_file):
            print(f"[WARN] include_annotations=true，但未提供有效 annotations_file，跳过合并。")
        else:
            try:
                import pandas as pd
                # 读排名
                df_rank = pd.read_csv(full_csv)
                # 读注释：根据后缀自动判断分隔符
                sep = "\t" if ann_file.lower().endswith((".tsv", ".txt")) else ","
                df_ann = pd.read_csv(ann_file, sep=sep)

                if "miRNA_name" not in df_ann.columns:
                    # 尝试从长表转宽表：要求 miRNA_name, attribute, value
                    required = {"miRNA_name", "attribute", "value"}
                    if required.issubset(set(df_ann.columns)):
                        df_ann = (
                            df_ann
                            .groupby(["miRNA_name", "attribute"])["value"]
                            .apply(lambda s: ";".join(map(str, s)))
                            .unstack("attribute")
                            .reset_index()
                        )
                    else:
                        raise ValueError("注释文件缺少 'miRNA_name' 或（'miRNA_name','attribute','value'）列")

                df_merged = df_rank.merge(df_ann, on="miRNA_name", how="left")
                ann_csv = os.path.join(output_dir, "functional_annotations.csv")
                df_merged.to_csv(ann_csv, index=False)
                print(f"[INFO] 功能注释已合并：{ann_csv}")
            except Exception as e:
                print(f"[WARN] 合并注释时发生异常，已跳过：{e}")

    # 摘要
    summary = {
        "disease_name": disease_name_out,
        "disease_index": int(d_col),
        "model_path": model_path,
        "n_miRNA": int(G),
        "outputs": {
            "full_rank_csv": full_csv,
            "topk_rank_csv": topk_csv,
            "annotations_csv": ann_csv
        }
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[DONE] 冷启动推理完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
