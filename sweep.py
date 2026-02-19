#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv,
    global_mean_pool, global_add_pool
)

# GPSConv is optional (needs torch_geometric>=2.3 typically)
try:
    from torch_geometric.nn import GPSConv
    _HAS_GPS = True
except Exception:
    _HAS_GPS = False

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from scipy import sparse
from scipy.sparse.linalg import eigsh


# ----------------------------
# Utilities
# ----------------------------

def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def pick_device(device: str) -> torch.device:
    device = device.lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ----------------------------
# Your SLA/FLA evaluation core (LogReg + RBF-SVM)
# ----------------------------

def class_weights(y):
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}

@dataclass
class ModelSpec:
    name: str
    estimator: object
    needs_scaling: bool

def make_models(seed: int, n_classes: int):
    models = []
    models.append(ModelSpec(
        "LogReg",
        LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        ),
        True,
    ))
    models.append(ModelSpec(
        "SVM_rbf",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        ),
        True,
    ))
    return models

def wrap_model(spec: ModelSpec):
    if spec.needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", spec.estimator)])
    return spec.estimator

@dataclass
class FoldResult:
    acc: float

def eval_one_fold(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    return FoldResult(acc=float(accuracy_score(y_te, y_hat)))

def run_seed(X, y, seed, n_splits=10):
    set_global_seed(seed)
    models = make_models(seed, n_classes=len(np.unique(y)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    out = {}
    for spec in models:
        model = wrap_model(spec)
        folds = []
        for tr_idx, te_idx in skf.split(X, y):
            folds.append(eval_one_fold(
                model,
                X[tr_idx], y[tr_idx],
                X[te_idx], y[te_idx]
            ))
        accs = np.array([f.acc for f in folds])
        out[spec.name] = {
            "summary": {
                "acc_mean": float(accs.mean()),
                "acc_std":  float(accs.std(ddof=1)),
            }
        }
    return out

def _nan_to_num(v):
    v = np.asarray(v, dtype=float)
    v[~np.isfinite(v)] = 0.0
    return v

def _moments(arr):
    if arr.size == 0:
        return [0,0,0,0]
    m = arr.mean()
    s = arr.std()
    if s < 1e-12:
        return [m,s,0,0]
    z = (arr-m)/s
    return [m,s,(z**3).mean(),(z**4).mean()-3]

def _quantiles(arr, qs=(0.05,0.25,0.5,0.75,0.95)):
    if arr.size == 0:
        return [0]*len(qs)
    return [float(np.quantile(arr,q)) for q in qs]

def betti_profile_degree(G, T=10):
    degs = np.array([d for _,d in G.degree()])
    if degs.size == 0:
        return np.zeros(T), np.zeros(T)
    taus = np.quantile(degs, np.linspace(0.05,0.95,T))
    b0, b1 = [], []
    for tau in taus:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for u,v in G.edges():
            if max(G.degree(u),G.degree(v)) <= tau:
                H.add_edge(u,v)
        c = nx.number_connected_components(H)
        n = H.number_of_nodes()
        m = H.number_of_edges()
        b0.append(c)
        b1.append(max(0, m - n + c))
    return np.array(b0), np.array(b1)

def topk_laplacian_eigs(G, k=4):
    n = G.number_of_nodes()
    if n<=1:
        return np.zeros(k)
    A = nx.to_scipy_sparse_array(G,format="csr").astype(np.float64)
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv = np.power(d,-0.5,where=d>0)
    D = sparse.diags(d_inv, format="csr").astype(np.float64)
    L = (sparse.eye(n, format="csr").astype(np.float64) - D @ A @ D).astype(np.float64)
    vals = eigsh(L, k=min(k+1,n-1), which="SM", return_eigenvectors=False)
    vals = np.sort(vals)
    vals = vals[1:k+1] if vals[0]<1e-6 else vals[:k]
    return np.pad(vals,(0,k-len(vals)))

def topk_adj_eigs(G, k=4):
    n = G.number_of_nodes()
    if n<=1:
        return np.zeros(k)
    A = nx.to_scipy_sparse_array(G,format="csr").astype(np.float64)
    vals = eigsh(A,k=min(k,n-1),which="LA",return_eigenvectors=False)
    vals = np.sort(vals)[::-1][:k]
    return np.pad(vals,(0,k-len(vals)))

def struct_stats(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    degs = np.array([d for _, d in G.degree()])
    clust = np.array(list(nx.clustering(G).values())) if n > 0 else np.zeros(1)
    tri = np.array(list(nx.triangles(G).values())) if n > 0 else np.zeros(1)
    density = 2*m/(n*(n-1)) if n > 1 else 0
    return np.array([
        n, m, density,
        *_moments(degs), *_quantiles(degs),
        *_moments(clust), *_moments(tri)
    ])

def _get_graph_label(data: Data) -> int:
    return int(data.y.item())

def StructSummary(dataset: Sequence[Data]):
    X, y = [], []
    for data in dataset:
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        ei = to_undirected(data.edge_index).cpu().numpy()
        if ei.ndim == 2 and ei.shape[0] == 2:
            edges = ei.T.tolist()
        elif ei.ndim == 2 and ei.shape[1] == 2:
            edges = ei.tolist()
        else:
            edges = []
        G.add_edges_from(edges)
        b0, b1 = betti_profile_degree(G)
        lap = topk_laplacian_eigs(G)
        adj = topk_adj_eigs(G)
        stats = struct_stats(G)
        X.append(np.concatenate([b0, b1, lap, adj, stats]))
        y.append(_get_graph_label(data))
    return _nan_to_num(np.vstack(X)), np.array(y)

def FeatSummary(dataset: Sequence[Data]):
    X, y = [], []
    for data in dataset:
        X.append(data.x.mean(dim=0).cpu().numpy())
        y.append(_get_graph_label(data))
    return _nan_to_num(np.vstack(X)), np.array(y)

def run_SLA_FLA(dataset: Sequence[Data], seeds=range(5)):
    Xs, y = StructSummary(dataset)
    Xf, _ = FeatSummary(dataset)

    SLA, FLA = {}, {}
    for seed in seeds:
        SLA[seed] = run_seed(Xs, y, seed)
        FLA[seed] = run_seed(Xf, y, seed)

    return {
        "SLA": SLA,
        "FLA": FLA,
        "meta": {
            "n_graphs": len(dataset),
            "n_classes": int(len(np.unique(y))),
            "seeds": list(seeds)
        }
    }

def aggregate_sla_fla(res: dict) -> dict:
    """
    Produces:
      SLA_LogReg_mean, SLA_SVM_mean, SLA_avg_mean
      FLA_LogReg_mean, FLA_SVM_mean, FLA_avg_mean
    where means are averaged across seeds of the per-seed 10-fold acc_mean.
    """
    def _agg_block(block: dict, model_name: str) -> Tuple[float, float]:
        vals = []
        for seed, payload in block.items():
            vals.append(payload[model_name]["summary"]["acc_mean"])
        vals = np.array(vals, dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    out = {}
    for side in ["SLA", "FLA"]:
        m1_mean, m1_std = _agg_block(res[side], "LogReg")
        m2_mean, m2_std = _agg_block(res[side], "SVM_rbf")
        out[f"{side}_LogReg_mean"] = m1_mean
        out[f"{side}_LogReg_std"]  = m1_std
        out[f"{side}_SVM_mean"]    = m2_mean
        out[f"{side}_SVM_std"]     = m2_std
        out[f"{side}_avg_mean"]    = float(0.5 * (m1_mean + m2_mean))
        out[f"{side}_avg_std"]     = float(0.5 * math.sqrt(m1_std**2 + m2_std**2))
    return out


# ----------------------------
# Synthetic dataset generator
# ----------------------------

@dataclass
class SynthConfig:
    n_graphs: int
    n_classes: int
    n_nodes_mean: int
    n_nodes_std: int
    feat_dim: int
    node_var: float

    struct_sep: float   # in [0,1]
    feat_sep: float     # in [0,1]

    edge_noise: float   # fraction of edges rewired (0..1)
    shared_p: float     # baseline ER p for "noise" graphs (label-independent)

    feat_signal: float  # scale of class means (bigger => easier FLA when feat_sep=1)

    min_nodes: int = 8
    max_nodes: int = 400

def sample_num_nodes(rng: np.random.RandomState, mean: int, std: int, min_n: int, max_n: int) -> int:
    n = int(round(rng.normal(mean, std)))
    n = max(min_n, min(max_n, n))
    return n

def class_struct_params(n_classes: int) -> List[dict]:
    """
    Each class gets a distinct structural generator "template".
    Mix of ER / BA / WS / SBM-like (2-block) graphs.
    This diversity helps structural summaries discriminate labels when struct_sep=1.
    """
    params = []
    for c in range(n_classes):
        t = c % 4
        if t == 0:
            params.append({"type": "ER", "p": 0.05 + 0.02 * (c % 5)})
        elif t == 1:
            params.append({"type": "BA", "m": 2 + (c % 4)})
        elif t == 2:
            params.append({"type": "WS", "k": 4 + 2 * (c % 4), "beta": 0.05 + 0.1 * ((c % 3) / 2.0)})
        else:
            # 2-block SBM with class-specific p_in/p_out
            pin = 0.10 + 0.04 * (c % 4)
            pout = 0.01 + 0.01 * ((c % 3))
            params.append({"type": "SBM2", "p_in": pin, "p_out": pout})
    return params

def gen_graph_from_template(rng: np.random.RandomState, n: int, template: dict) -> nx.Graph:
    t = template["type"]
    if t == "ER":
        p = float(template["p"])
        G = nx.erdos_renyi_graph(n, p, seed=rng.randint(1_000_000_000))
    elif t == "BA":
        m = int(template["m"])
        m = max(1, min(m, max(1, n-1)))
        G = nx.barabasi_albert_graph(n, m, seed=rng.randint(1_000_000_000))
    elif t == "WS":
        k = int(template["k"])
        k = max(2, min(k, n-1))
        if k % 2 == 1:
            k += 1
            k = min(k, n-1)
        beta = float(template["beta"])
        G = nx.watts_strogatz_graph(n, k, beta, seed=rng.randint(1_000_000_000))
    elif t == "SBM2":
        p_in = float(template["p_in"])
        p_out = float(template["p_out"])
        sizes = [n // 2, n - (n // 2)]
        probs = [[p_in, p_out], [p_out, p_in]]
        G = nx.stochastic_block_model(sizes, probs, seed=rng.randint(1_000_000_000))
    else:
        raise ValueError(f"Unknown template type: {t}")
    return nx.Graph(G)

def rewire_edges(rng: np.random.RandomState, G: nx.Graph, frac: float) -> nx.Graph:
    """
    Simple edge-noise: remove a fraction of edges and add same number of random edges.
    """
    frac = clamp01(frac)
    if frac <= 0.0 or G.number_of_edges() == 0 or G.number_of_nodes() <= 1:
        return G

    n = G.number_of_nodes()
    m = G.number_of_edges()
    k = int(round(frac * m))
    if k <= 0:
        return G

    edges = list(G.edges())
    rng.shuffle(edges)
    to_remove = edges[:k]
    G.remove_edges_from(to_remove)

    # add random edges
    added = 0
    attempts = 0
    max_attempts = 20 * k + 100
    while added < k and attempts < max_attempts:
        u = rng.randint(0, n)
        v = rng.randint(0, n)
        attempts += 1
        if u == v:
            continue
        if G.has_edge(u, v):
            continue
        G.add_edge(u, v)
        added += 1

    return G

def make_class_means(rng: np.random.RandomState, n_classes: int, feat_dim: int, feat_signal: float) -> np.ndarray:
    """
    Random class mean vectors; normalized and scaled.
    """
    M = rng.normal(size=(n_classes, feat_dim))
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return feat_signal * M

def gen_one_pyg_graph(
    rng: np.random.RandomState,
    cfg: SynthConfig,
    class_templates: List[dict],
    class_means: np.ndarray,
    label: int
) -> Data:
    n = sample_num_nodes(rng, cfg.n_nodes_mean, cfg.n_nodes_std, cfg.min_nodes, cfg.max_nodes)

    # ---- Structure: label-dependent with prob struct_sep, else shared noise ----
    if rng.rand() < cfg.struct_sep:
        G = gen_graph_from_template(rng, n, class_templates[label])
    else:
        # shared ER noise (label-independent)
        G = nx.erdos_renyi_graph(n, cfg.shared_p, seed=rng.randint(1_000_000_000))
        G = nx.Graph(G)

    # Edge noise / rewiring
    G = rewire_edges(rng, G, cfg.edge_noise)

    # Ensure node indexing 0..n-1
    G.add_nodes_from(range(n))

    # ---- Features: label-dependent with prob feat_sep, else shared noise ----
    x = rng.normal(loc=0.0, scale=math.sqrt(cfg.node_var), size=(n, cfg.feat_dim)).astype(np.float32)
    if rng.rand() < cfg.feat_sep:
        x += class_means[label].astype(np.float32)[None, :]

    # Convert to PyG Data
    edge_index = np.array(list(G.edges()), dtype=np.int64)
    if edge_index.size == 0:
        edge_index = edge_index.reshape(0, 2)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long) if edge_index.shape[0] > 0 else torch.empty((2,0), dtype=torch.long)
    edge_index = to_undirected(edge_index)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long)
    )
    return data

def generate_dataset(cfg: SynthConfig, seed: int) -> List[Data]:
    rng = np.random.RandomState(seed)
    class_templates = class_struct_params(cfg.n_classes)
    class_means = make_class_means(rng, cfg.n_classes, cfg.feat_dim, cfg.feat_signal)

    # balanced label allocation
    labels = np.arange(cfg.n_graphs) % cfg.n_classes
    rng.shuffle(labels)

    ds = []
    for y in labels:
        ds.append(gen_one_pyg_graph(rng, cfg, class_templates, class_means, int(y)))
    return ds


# ----------------------------
# Basic GNNs + training (graph classification)
# ----------------------------

class GNN(nn.Module):
    def __init__(self, kind: str, in_ch: int, hidden: int, out_ch: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.5, pool: str = "mean"):
        super().__init__()
        kind = kind.lower()
        self.kind = kind
        self.dropout = dropout
        self.pool = global_mean_pool if pool == "mean" else global_add_pool

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        def make_conv(i_in: int, i_out: int):
            if kind == "gcn":
                return GCNConv(i_in, i_out)
            if kind == "sage":
                return SAGEConv(i_in, i_out)
            if kind == "gat":
                # keep output dim stable: heads * (i_out/heads)
                h = max(1, heads)
                per_head = max(1, i_out // h)
                return GATConv(i_in, per_head, heads=h, concat=True)
            raise ValueError(f"Unknown GNN kind: {kind}")

        # layer 1
        self.convs.append(make_conv(in_ch, hidden))
        self.norms.append(nn.BatchNorm1d(hidden))
        # subsequent layers
        for _ in range(max(1, num_layers) - 1):
            self.convs.append(make_conv(hidden, hidden))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.lin = nn.Linear(hidden, out_ch)

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = self.pool(x, batch)
        return self.lin(g)

class GPS(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int, out_dim:int,
                 num_layers:int=2, heads:int=8, dropout:float=0.5, pool:str='mean'):
        super().__init__()
        if not _HAS_GPS:
            raise RuntimeError("GPSConv is not available in this environment. Install/upgrade torch-geometric.")
        self.in_proj = nn.Identity() if in_channels == hidden_dim else nn.Linear(in_channels, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        for _ in range(max(1, num_layers)):
            local_mpnn = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(GPSConv(channels=hidden_dim, conv=local_mpnn, heads=heads, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.in_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, edge_index, batch=batch)
            out = norm(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            x = x + out
        return self.linear(self.pool(x, batch))

def build_model(model_name, in_ch, h_ch, out_ch, num_layers=2, heads=8, dropout=0.5, pool='mean'):
    model_name = model_name.lower()
    if model_name in ['gcn', 'sage', 'gat']:
        return GNN(model_name, in_ch, h_ch, out_ch, num_layers=num_layers, heads=heads, dropout=dropout, pool=pool)
    elif model_name == 'gps':
        return GPS(in_ch, h_ch, out_ch, num_layers=num_layers, heads=heads, dropout=dropout, pool=pool)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def split_dataset_adapted(ds: Sequence[Data], splits=(0.6, 0.2, 0.2), seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    n = len(ds)
    n_tr = int(round(splits[0]*n))
    n_va = int(round(splits[1]*n))
    tr = [ds[i] for i in idx[:n_tr]]
    va = [ds[i] for i in idx[n_tr:n_tr+n_va]]
    te = [ds[i] for i in idx[n_tr+n_va:]]
    return tr, va, te

def remap_labels_inplace(ds: Sequence[Data]):
    labels = sorted({int(d.y.item()) for d in ds})
    mapping = {old: i for i, old in enumerate(labels)}
    for d in ds:
        d.y = torch.tensor([mapping[int(d.y.item())]], dtype=torch.long)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
    return correct / total if total > 0 else 0.0

def train_and_score_adapted(model_name: str,
                            dataset_source: Sequence[Data],
                            seed: int = 42,
                            epochs: int = 100,
                            batch_size: int = 32,
                            h_ch: int = 64,
                            pool: str = 'mean',
                            device: torch.device = torch.device("cpu"),
                            num_layers: int = 2,
                            heads: int = 8,
                            dropout: float = 0.5) -> float:

    set_global_seed(seed)

    train_list, val_list, test_list = split_dataset_adapted(dataset_source, splits=(0.6, 0.2, 0.2), seed=seed)
    remap_labels_inplace(train_list)
    remap_labels_inplace(val_list)
    remap_labels_inplace(test_list)

    if len(train_list) == 0:
        raise RuntimeError("Empty train set after filtering.")

    in_ch = train_list[0].x.size(1)
    num_classes = int(max(int(d.y.item()) for d in train_list + val_list + test_list) + 1)

    model = build_model(model_name, in_ch, h_ch, num_classes,
                        num_layers=num_layers, heads=heads, dropout=dropout, pool=pool).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_test_acc = 0.0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y.long())
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return float(best_test_acc)


# ----------------------------
# Sweep driver
# ----------------------------

def parse_list(s: str) -> List[float]:
    # accepts "0,0.1,0.2" or "linspace(0,1,21)"
    s = s.strip()
    if s.startswith("linspace(") and s.endswith(")"):
        inner = s[len("linspace("):-1]
        a, b, n = inner.split(",")
        a = float(a); b = float(b); n = int(n)
        return [float(x) for x in np.linspace(a, b, n)]
    return [float(x) for x in s.split(",") if x.strip() != ""]

def main():
    p = argparse.ArgumentParser("Synthetic SLA/FLA sweep for graph classification.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--csv_name", type=str, default="sweep_results.csv")

    # dataset scale
    p.add_argument("--n_graphs", type=int, default=240)
    p.add_argument("--n_classes", type=int, default=3)
    p.add_argument("--n_nodes_mean", type=int, default=60)
    p.add_argument("--n_nodes_std", type=int, default=10)
    p.add_argument("--feat_dim", type=int, default=32)
    p.add_argument("--node_var", type=float, default=1.0)

    # generator knobs
    p.add_argument("--shared_p", type=float, default=0.04)
    p.add_argument("--edge_noise", type=float, default=0.02)
    p.add_argument("--feat_signal", type=float, default=3.0)

    # sweep grids
    p.add_argument("--struct_grid", type=str, default="linspace(0,1,21)")
    p.add_argument("--feat_grid", type=str, default="linspace(0,1,21)")
    p.add_argument("--jitter_reps", type=int, default=2)
    p.add_argument("--jitter_std", type=float, default=0.015)

    # SLA/FLA evaluation seeds
    p.add_argument("--sla_fla_seeds", type=str, default="0,1,2,3,4")

    # GNN training
    p.add_argument("--gnn_model", type=str, default="gcn", choices=["gcn","sage","gat","gps"])
    p.add_argument("--gnn_seeds", type=str, default="0,1,2")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--pool", type=str, default="mean", choices=["mean","add"])
    p.add_argument("--device", type=str, default="auto")

    # persistence
    p.add_argument("--save_datasets", action="store_true",
                   help="If set, saves each generated dataset as a .pt file under out_dir/datasets/")

    args = p.parse_args()

    ensure_dir(args.out_dir)
    ds_dir = os.path.join(args.out_dir, "datasets")
    if args.save_datasets:
        ensure_dir(ds_dir)

    struct_vals = parse_list(args.struct_grid)
    feat_vals = parse_list(args.feat_grid)
    sla_fla_seeds = [int(x) for x in parse_list(args.sla_fla_seeds)]
    gnn_seeds = [int(x) for x in parse_list(args.gnn_seeds)]

    device = pick_device(args.device)

    if args.gnn_model == "gps" and not _HAS_GPS:
        raise RuntimeError("You selected --gnn_model gps, but GPSConv is unavailable. Use gcn/sage/gat or install GPSConv.")

    # CSV header
    import csv
    out_csv = os.path.join(args.out_dir, args.csv_name)

    fieldnames = [
        # generator parameters
        "struct_sep_param", "feat_sep_param", "struct_sep_used", "feat_sep_used",
        "n_graphs", "n_classes", "n_nodes_mean", "n_nodes_std", "feat_dim", "node_var",
        "shared_p", "edge_noise", "feat_signal",
        "chance_acc",

        # measured SLA/FLA
        "SLA_LogReg_mean", "SLA_LogReg_std",
        "SLA_SVM_mean", "SLA_SVM_std",
        "SLA_avg_mean", "SLA_avg_std",
        "FLA_LogReg_mean", "FLA_LogReg_std",
        "FLA_SVM_mean", "FLA_SVM_std",
        "FLA_avg_mean", "FLA_avg_std",

        # GNN
        "gnn_model", "gnn_epochs", "gnn_hidden", "gnn_num_layers", "gnn_heads", "gnn_dropout", "gnn_pool",
        "GNN_test_mean", "GNN_test_std",
        "dataset_path"
    ]

    rng_master = np.random.RandomState(12345)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_runs = len(struct_vals) * len(feat_vals) * max(1, args.jitter_reps)
        run_id = 0

        for s0 in struct_vals:
            for f0 in feat_vals:
                for jr in range(max(1, args.jitter_reps)):
                    run_id += 1

                    # jitter to densify the plane without changing the nominal grid resolution
                    s = clamp01(float(s0 + rng_master.normal(0, args.jitter_std)))
                    ff = clamp01(float(f0 + rng_master.normal(0, args.jitter_std)))

                    cfg = SynthConfig(
                        n_graphs=args.n_graphs,
                        n_classes=args.n_classes,
                        n_nodes_mean=args.n_nodes_mean,
                        n_nodes_std=args.n_nodes_std,
                        feat_dim=args.feat_dim,
                        node_var=args.node_var,
                        struct_sep=s,
                        feat_sep=ff,
                        edge_noise=args.edge_noise,
                        shared_p=args.shared_p,
                        feat_signal=args.feat_signal,
                    )

                    # dataset seed: deterministic per grid cell + jitter rep
                    ds_seed = int(10_000_000 * s0 + 1_000_000 * f0 + 10_000 * jr) % 2_000_000_000
                    dataset = generate_dataset(cfg, seed=ds_seed)

                    dataset_path = ""
                    if args.save_datasets:
                        dataset_path = os.path.join(ds_dir, f"ds_struct{float(s0):.3f}_feat{float(f0):.3f}_jr{jr}.pt")
                        torch.save(dataset, dataset_path)

                    # SLA/FLA
                    sla_fla_res = run_SLA_FLA(dataset, seeds=sla_fla_seeds)
                    agg = aggregate_sla_fla(sla_fla_res)

                    # GNN scores across seeds
                    gnn_scores = []
                    for gseed in gnn_seeds:
                        acc = train_and_score_adapted(
                            args.gnn_model,
                            dataset_source=dataset,
                            seed=gseed,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            h_ch=args.hidden,
                            pool=args.pool,
                            device=device,
                            num_layers=args.num_layers,
                            heads=args.heads,
                            dropout=args.dropout
                        )
                        gnn_scores.append(acc)
                    gnn_scores = np.array(gnn_scores, dtype=float)
                    gnn_mean = float(gnn_scores.mean())
                    gnn_std  = float(gnn_scores.std(ddof=1) if len(gnn_scores) > 1 else 0.0)

                    row = {
                        "struct_sep_param": float(s0),
                        "feat_sep_param": float(f0),
                        "struct_sep_used": float(s),
                        "feat_sep_used": float(ff),

                        "n_graphs": args.n_graphs,
                        "n_classes": args.n_classes,
                        "n_nodes_mean": args.n_nodes_mean,
                        "n_nodes_std": args.n_nodes_std,
                        "feat_dim": args.feat_dim,
                        "node_var": args.node_var,
                        "shared_p": args.shared_p,
                        "edge_noise": args.edge_noise,
                        "feat_signal": args.feat_signal,
                        "chance_acc": float(1.0 / args.n_classes),

                        **agg,

                        "gnn_model": args.gnn_model,
                        "gnn_epochs": args.epochs,
                        "gnn_hidden": args.hidden,
                        "gnn_num_layers": args.num_layers,
                        "gnn_heads": args.heads,
                        "gnn_dropout": args.dropout,
                        "gnn_pool": args.pool,
                        "GNN_test_mean": gnn_mean,
                        "GNN_test_std": gnn_std,
                        "dataset_path": dataset_path
                    }

                    writer.writerow(row)

                    # keep memory stable
                    del dataset
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    print(f"Wrote CSV: {out_csv}")
    if args.save_datasets:
        print(f"Saved datasets under: {ds_dir}")

if __name__ == "__main__":
    main()
