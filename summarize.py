from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from itertools import combinations, product

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import NormalizeFeatures

import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh

# =========================
# Utilities
# =========================
def _safe_float(x: float) -> float:
    try:
        xf = float(x)
        if math.isfinite(xf):
            return xf
        return 0.0
    except Exception:
        return 0.0

def _nan_to_num(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v[~np.isfinite(v)] = 0.0
    return v

def _quantiles(arr: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> List[float]:
    if arr.size == 0:
        return [0.0] * len(qs)
    return [float(np.quantile(arr, q)) for q in qs]

def _moments(arr: np.ndarray) -> List[float]:
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    m = float(np.mean(arr))
    s = float(np.std(arr))
    if s <= 1e-12:
        return [m, s, 0.0, 0.0]
    z = (arr - m) / (s + 1e-12)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    return [m, s, skew, kurt]

# =========================
# Betti profiles via degree sublevel filtration
# =========================
def _betti_profile_degree_filtration(G: nx.Graph, T: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if G.number_of_nodes() == 0:
        return np.zeros(T), np.zeros(T)

    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size == 0:
        return np.zeros(T), np.zeros(T)

    qs = np.linspace(0.05, 0.95, T)
    taus = np.quantile(degs, qs)

    b0, b1 = [], []
    for tau in taus:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for u, v in G.edges():
            if max(G.degree(u), G.degree(v)) <= tau:
                H.add_edge(u, v)
        c = nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0
        n = H.number_of_nodes()
        m = H.number_of_edges()
        b0.append(float(c))
        b1.append(float(max(0, m - n + c)))
    return np.array(b0, dtype=float), np.array(b1, dtype=float)

# =========================
# Spectral features
# =========================
def _topk_eigs_norm_laplacian(G: nx.Graph, k: int = 16) -> np.ndarray:
    n = G.number_of_nodes()
    if n == 0 or k <= 0:
        return np.zeros(max(0, k))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    d = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    kk = min(max(1, k + 1), max(1, min(n - 1, 2 * k)))
    try:
        vals = eigsh(L, k=kk, which='SM', return_eigenvectors=False)
        vals = np.sort(np.real(vals))
    except Exception:
        vals = np.linalg.eigvalsh(L.toarray())
        vals = np.sort(np.real(vals))

    if vals.size > 0 and abs(vals[0]) < 1e-8:
        vals = vals[1:]
    vals = vals[:k]
    if vals.size < k:
        vals = np.pad(vals, (0, k - vals.size))
    return _nan_to_num(vals)

def _topk_eigs_adjacency(G: nx.Graph, k: int = 16) -> np.ndarray:
    n = G.number_of_nodes()
    if n == 0 or k <= 0:
        return np.zeros(max(0, k))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    if k >= n or n <= 2:
        try:
            vals = np.linalg.eigvalsh(A.toarray())
            vals = np.sort(np.real(vals))[::-1]
        except Exception:
            vals = np.zeros(0)
    else:
        try:
            vals = eigsh(A, k=k, which='LA', return_eigenvectors=False)
            vals = np.sort(np.real(vals))[::-1]
        except Exception:
            vals = np.linalg.eigvalsh(A.toarray())
            vals = np.sort(np.real(vals))[::-1]
    vals = vals[:k]
    if vals.size < k:
        vals = np.pad(vals, (0, k - vals.size))
    return _nan_to_num(vals)

# =========================
# Structural statistics
# =========================
def _struct_stats(G: nx.Graph) -> List[float]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return [0.0] * 16

    degs = np.array([d for _, d in G.degree()], dtype=float)
    deg_mom = _moments(degs)
    deg_qs = _quantiles(degs)

    try:
        clust = list(nx.clustering(G).values())
        cl_mom = _moments(np.array(clust, dtype=float))
    except Exception:
        cl_mom = [0.0, 0.0, 0.0, 0.0]

    try:
        tri = list(nx.triangles(G).values())
        tri_mom = _moments(np.array(tri, dtype=float))
    except Exception:
        tri_mom = [0.0, 0.0, 0.0, 0.0]

    density = 0.0
    if n > 1:
        density = 2.0 * m / (n * (n - 1))

    return [
        float(n), float(m), float(density),
        *deg_mom, *deg_qs,
        *cl_mom, *tri_mom,
    ]

# =========================
# Build graphs -> components
# =========================
def graph_to_components(data: Data, k_eigs: int, betti_T: int) -> Dict[str, np.ndarray]:
    num_nodes = int(data.num_nodes)
    if data.edge_index is None:
        G = nx.empty_graph(num_nodes)
    else:
        ei = to_undirected(data.edge_index)
        edges = ei.t().detach().cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges.tolist())

    b0, b1 = _betti_profile_degree_filtration(G, T=betti_T)
    leigs = _topk_eigs_norm_laplacian(G, k=k_eigs)
    aeigs = _topk_eigs_adjacency(G, k=k_eigs)
    stats = np.array(_struct_stats(G), dtype=float)

    comps = {
        "betti": np.concatenate([b0, b1]),
        "lap": leigs,
        "adj": aeigs,
        "stats": stats,
    }
    return {k: _nan_to_num(v) for k, v in comps.items()}

def assemble_S_from_components(comp_rows: List[Dict[str, np.ndarray]],
                               use_keys: List[str]) -> np.ndarray:
    mats = []
    for row in comp_rows:
        parts = [row[k] for k in use_keys]
        mats.append(np.concatenate(parts))
    return _nan_to_num(np.vstack(mats))

def compute_F(data: Data, method:str="centroid") -> np.ndarray:
    if method == "centroid":
        return data.x.numpy().mean(axis=0)
    elif method == "meanStd":
        mean = data.x.numpy().mean(axis=0)
        std = data.x.numpy().std(axis=0)
        return np.concatenate([mean, std], axis=0)
    else:
        raise NotImplementedError("Other methods not implemented yet.")
    
def compute_prototypes(F_list: List[np.ndarray], y_list):
    labels = np.unique(y_list)
    prototypes = {}
    for label in labels:
        Fs = [F for F, y in zip(F_list, y_list) if y == label]
        prototypes[label] = np.mean(Fs, axis=0)
    return prototypes

def compute_distances(F_list: List[np.ndarray], prototypes: Dict[int, np.ndarray]) -> np.ndarray:
    distances = []
    for F in F_list:
        for proto in prototypes.values():
            dist = np.linalg.norm(F - proto)
            distances.append(dist)
            cos_dist = 1 - (np.dot(F, proto) / (np.linalg.norm(F) * np.linalg.norm(proto) + 1e-10))
            distances.append(cos_dist)
    
    return np.array(distances).reshape(len(F_list), -1)



def StructSummary(ds):
    y = np.array([int(ds[i].y.item()) for i in range(len(ds))])

    comp_rows: List[Dict[str, np.ndarray]] = []
    for data in ds:
        comps = graph_to_components(data, k_eigs=4, betti_T=10)
        comp_rows.append({k: comps[k] for k in ("betti", "lap", "adj", "stats")})

    Xs = assemble_S_from_components(comp_rows, use_keys=["betti", "lap", "adj", "stats"])

    return Xs, y


def FeatSummary(ds):
    y = np.array([int(ds[i].y.item()) for i in range(len(ds))])

    F_list: List[np.ndarray] = []
    F_list = [compute_F(data, method="centroid") for data in ds]
    
    # Optional prototype distances/learning
    # proto = compute_prototypes(F_list, y)
    # D = compute_distances(F_list, proto)
    # F_list = D

    Xf = _nan_to_num(np.vstack(F_list))

    return Xf, y