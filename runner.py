import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures

from score import run_seed
from summarize import StructSummary, FeatSummary

REPORT_KEYS = ["acc_mean", "bal_acc_mean", "macro_f1_mean"]

def get_vals(results_by_seed: dict, model_name: str, key: str) -> list[float]:
    return [float(results_by_seed[s][model_name]["summary"][key]) for s in results_by_seed.keys()]

def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"

def main(root, name):
    ds = TUDataset(root=str(root), name=name, use_node_attr=True, transform=NormalizeFeatures())

    # Ensure all graphs are non-empty (Handle fingerprint), throw out the graphs that are empty
    non_empty_indices = [i for i in range(len(ds)) if ds[i].num_nodes > 0]
    ds = ds.index_select(non_empty_indices)

    # Does the dataset have node attributes?
    has_node_attr = ds.num_node_features > 0

    # Get labels
    y = ds.data.y.numpy()

    # Compute Struct vectors
    X_S, _ = StructSummary(ds)

    if has_node_attr:
        # Get node attributes and compute Feature vectors
        X_F, _ = FeatSummary(ds)
    

    SLA_results, FLA_results = {}, {}
    for seed in range(10):
        struct_res = run_seed(X_S, y, seed, n_splits=10, shuffle=True, use_class_weights=True)
        SLA_results[seed] = struct_res

        if has_node_attr:
            feature_res = run_seed(X_F, y, seed, n_splits=10, shuffle=True, use_class_weights=True)
            FLA_results[seed] = feature_res
    
    # Report average results per model:
    print(f"Results for dataset: {name}")

    print("Structure Label Alignment (SLA) Results:")
    for model_name in SLA_results[0].keys():
        parts = []
        for key in REPORT_KEYS:
            vals = get_vals(SLA_results, model_name, key)
            parts.append(f"{key.replace('_mean','')}: {fmt(float(np.mean(vals)), float(np.std(vals, ddof=1)))}")
        print(f"  {model_name}: " + " | ".join(parts))

    if has_node_attr:
        print("Feature Label Alignment (FLA) Results:")
        for model_name in FLA_results[0].keys():
            parts = []
            for key in REPORT_KEYS:
                vals = get_vals(FLA_results, model_name, key)
                parts.append(f"{key.replace('_mean','')}: {fmt(float(np.mean(vals)), float(np.std(vals, ddof=1)))}")
            print(f"  {model_name}: " + " | ".join(parts))
    


if __name__ == "__main__":
    # for ds in ["AIDS", "BZR", "COX2", "FRANKENSTEIN", "PROTEINS", "PROTEINS_full"]:
    #     main(root="data/TUDataset", name=ds)
    
    # for ds in ["COIL-DEL", "COIL-RAG", "Fingerprint", "Letter-high"]:
    #     main(root="data/TUDataset", name=ds)
    
    for ds in ["COLORS-3", "Synthie"]:
        main(root="data/TUDataset", name=ds)