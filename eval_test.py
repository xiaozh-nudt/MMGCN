import os
import sys
import re
import copy
import csv
import pickle
import argparse
from datetime import datetime
from typing import Optional, List, Tuple

import torch

from model2 import LrGNN
from dataloader import DataLoader
from configloader import Config
from AUCscore import AUC_score

# 可选：用于打印命中率的统计（如果项目里有）
try:
    from statistic import statistic
except Exception:
    statistic = None


def get_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def list_model_files(model_dir: str) -> List[str]:
    if not os.path.isdir(model_dir):
        return []
    return [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.lower().endswith(".model") and os.path.isfile(os.path.join(model_dir, f))
    ]


def filter_and_sort(files: List[str], pattern: Optional[str], sort_key: str) -> List[str]:
    # 先按正则过滤
    if pattern:
        rx = re.compile(pattern)
        files = [p for p in files if rx.search(os.path.basename(p))]

    # 再排序：mtime(修改时间) 或 auc(按文件名中的 valAUC 数值)
    if sort_key == "auc":
        auc_pat = re.compile(r"valAUC([0-9]*\.[0-9]+|[0-9]+)")
        def auc_or_neg(p):
            m = auc_pat.search(os.path.basename(p))
            try:
                return float(m.group(1)) if m else -1.0
            except Exception:
                return -1.0
        files.sort(key=lambda p: auc_or_neg(p), reverse=True)
    else:
        # mtime
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def save_pickle(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(pickle.dumps(obj))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def eval_model(model_path: str,
               model: LrGNN,
               device: torch.device,
               data_triplet,
               test_coords_dict,
               threshold: Optional[float] = None,
               dump_each: bool = False,
               res_dir: Optional[str] = None):
    """对单个模型进行评测，返回 (auc, info_dict)。发生异常时 auc=None 并返回错误信息"""
    data_go, data_seq, data_sim = data_triplet
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            predict = model(data_go, data_seq, data_sim)

        auc, label_dump, res_dump = AUC_score(predict, test_coords_dict)

        info = {
            "auc": auc,
            "tp": None,
            "tn": None,
            "p_total": len(test_coords_dict.get("test_true", [])),
            "n_total": len(test_coords_dict.get("test_false", [])),
            "acc_at_threshold": None,
            "error": ""
        }

        # 阈值统计可选
        if threshold is not None:
            thr = float(threshold)
            tp = 0
            tn = 0
            for g, d in test_coords_dict.get("test_true", []):
                if predict[g][d] >= thr:
                    tp += 1
            for g, d in test_coords_dict.get("test_false", []):
                if predict[g][d] < thr:
                    tn += 1
            info["tp"] = tp
            info["tn"] = tn
            total = max(1, info["p_total"] + info["n_total"])
            info["acc_at_threshold"] = (tp + tn) / total

        # 每个模型是否单独 dump
        if dump_each and res_dir:
            base = os.path.splitext(os.path.basename(model_path))[0]
            save_pickle(os.path.join(res_dir, f"EVAL_{base}_label.pkl"), label_dump)
            save_pickle(os.path.join(res_dir, f"EVAL_{base}_res.pkl"),   res_dump)

        return auc, info
    except Exception as e:
        return None, {
            "auc": None,
            "tp": None,
            "tn": None,
            "p_total": None,
            "n_total": None,
            "acc_at_threshold": None,
            "error": f"{type(e).__name__}: {e}"
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ALL models under config.model_save_path")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu (default auto)")
    parser.add_argument("--pattern", type=str, default="", help="Regex to filter model filenames")
    parser.add_argument("--sort", choices=["mtime", "auc"], default="mtime",
                        help="Sort models by 'mtime' (default) or by 'auc' parsed from filename (valAUC...)")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N models (0 means no limit)")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold for hit-rate stats")
    parser.add_argument("--dump_each", action="store_true", help="Dump per-model label/res pickle")
    args = parser.parse_args()

    device = get_device(args.device)
    config = Config()

    # 准备模型列表
    all_models = list_model_files(config.model_save_path)
    if not all_models:
        print(f"[ERROR] No .model files found in: {config.model_save_path}")
        sys.exit(1)

    models = filter_and_sort(all_models, args.pattern or None, args.sort)
    if args.limit and args.limit > 0:
        models = models[:args.limit]

    print(f"[INFO] Found {len(models)} model(s) to evaluate under: {config.model_save_path}")

    # 载入数据一次复用
    dl = DataLoader()
    data_go, data_seq, data_sim = dl.get_data()
    test_coords_dict = copy.deepcopy(getattr(data_go, "test_set_coords_dict", {}))
    if len(test_coords_dict.get("test_true", [])) == 0 or len(test_coords_dict.get("test_false", [])) == 0:
        print("[WARN] Empty test set. Check your data building pipeline.")

    data_go = data_go.to(device)
    data_seq = data_seq.to(device)
    data_sim = data_sim.to(device)

    model = LrGNN().to(device)

    # 结果输出目录与 CSV
    ensure_dir(config.res_save_path)
    csv_path = os.path.join(config.res_save_path, "EVAL_summary.csv")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 写表头
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["evaluated_at",
                        "model_path",
                        "auc",
                        "p_total",
                        "n_total",
                        "threshold",
                        "tp",
                        "tn",
                        "acc_at_threshold",
                        "notes_or_error"])
        # 逐个评测
        for i, mp in enumerate(models, 1):
            print(f"[{i}/{len(models)}] Evaluating: {mp}")
            auc, info = eval_model(
                model_path=mp,
                model=model,
                device=device,
                data_triplet=(data_go, data_seq, data_sim),
                test_coords_dict=test_coords_dict,
                threshold=args.threshold,
                dump_each=args.dump_each,
                res_dir=config.res_save_path,
            )
            w.writerow([
                now_str,
                mp,
                f"{info['auc']:.6f}" if info["auc"] is not None else "",
                info["p_total"],
                info["n_total"],
                args.threshold if args.threshold is not None else "",
                info["tp"] if info["tp"] is not None else "",
                info["tn"] if info["tn"] is not None else "",
                f"{info['acc_at_threshold']:.6f}" if info["acc_at_threshold"] is not None else "",
                info["error"]
            ])
            # 也把结果同步打印到控制台
            if info["error"]:
                print(f"  -> ERROR: {info['error']}")
            else:
                line = f"  -> AUC={info['auc']:.6f}"
                if args.threshold is not None:
                    line += f", TP={info['tp']}/{info['p_total']}, TN={info['tn']}/{info['n_total']}, ACC={info['acc_at_threshold']:.4f}"
                print(line)

    print(f"\n[DONE] Summary written to: {csv_path}")


if __name__ == "__main__":
    main()
