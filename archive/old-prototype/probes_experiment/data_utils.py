"""Shared data loading utilities."""
import json, os
import numpy as np
import pandas as pd
import torch
from config import LAYERS


def load_hidden_states(pt_dir: str, split: str, n: int) -> np.ndarray:
    """Returns [n, n_layers, d_model] float32."""
    data = []
    for i in range(n):
        path = os.path.join(pt_dir, f"{split}_{i}.pt")
        d = torch.load(path, map_location="cpu", weights_only=True)
        data.append(torch.stack([d[l] for l in LAYERS], dim=0).float().numpy())
    return np.stack(data)


def load_vu_from_judge(judge_path: str, n: int) -> np.ndarray:
    """Mean of 10 judge scores per question → [n] float32."""
    with open(judge_path) as f:
        judge = json.load(f)
    return np.array([np.mean(scores) for scores in judge[:n]], dtype=np.float32)


def load_acc(acc_json: str, n: int) -> np.ndarray:
    """Accuracy labels → [n] int32 (1=correct, 0=hallucinated)."""
    with open(acc_json) as f:
        d = json.load(f)
    if isinstance(d, dict):
        keys = sorted(d.keys(), key=lambda k: int(k.split("_")[1]))
        return np.array([d[k] for k in keys[:n]], dtype=np.int32)
    return np.array(d[:n], dtype=np.int32)


def load_all(train_csv, test_csv, train_pt_dir, test_pt_dir,
             train_vu_judge, test_vu_judge,
             train_acc_json, test_acc_json,
             n_train=500, n_test=500, verbose=True):
    """Load everything needed for experiments. Returns dict."""

    def p(msg):
        if verbose: print(msg)

    p("Loading CSVs...")
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    p("Loading hidden states...")
    hs_tr = load_hidden_states(train_pt_dir, "train", n_train)
    hs_te = load_hidden_states(test_pt_dir,  "test",  n_test)
    p(f"  hs_train={hs_tr.shape}  hs_test={hs_te.shape}")

    p("Loading SE from CSV...")
    se_tr = df_tr["sentence_semantic_entropy"].values.astype(np.float32)
    se_te = df_te["sentence_semantic_entropy"].values.astype(np.float32)

    p("Loading VU...")
    vu_tr = df_tr["verbal_uncertainty"].values.astype(np.float32)
    vu_te = df_te["verbal_uncertainty"].values.astype(np.float32)

    p("Loading accuracy / hallucination labels...")
    acc_tr = load_acc(train_acc_json, n_train)
    acc_te = load_acc(test_acc_json,  n_test)
    hal_tr = 1 - acc_tr
    hal_te = 1 - acc_te
    p(f"  hal_rate train={hal_tr.mean():.3f}  test={hal_te.mean():.3f}")

    return dict(
        hs_tr=hs_tr, hs_te=hs_te,
        se_tr=se_tr, se_te=se_te,
        vu_tr=vu_tr, vu_te=vu_te,
        hal_tr=hal_tr, hal_te=hal_te,
    )
