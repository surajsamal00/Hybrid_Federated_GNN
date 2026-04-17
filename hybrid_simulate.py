"""
hybrid_simulate.py
------------------
Main entry point for the HYBRID federated simulation.
XGBoost (local) + GraphSAGE (federated) with dynamic number of banks.
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from hybrid_client import HybridFederatedClient
from server import CentralServer
from stream import StreamQueue

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── CLI args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hybrid XGBoost+GraphSAGE Federated Simulation")
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--rounds", type=int, default=30)
parser.add_argument("--local_epochs", type=int, default=1)
parser.add_argument("--mu", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=75)
parser.add_argument("--stream_frac", type=float, default=0.15)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--replay_frac", type=float, default=0.2)
parser.add_argument("--use_fedadam", action="store_true", default=True)
parser.add_argument("--no_fedadam", dest="use_fedadam", action="store_false")
parser.add_argument("--server_lr", type=float, default=1e-3)
parser.add_argument("--max_rows", type=int, default=None)
parser.add_argument("--out_dir", type=str, default="./hybrid_fed_results")
parser.add_argument("--num_banks", type=int, default=4)
parser.add_argument("--k_default", type=int, default=15)  
parser.add_argument("--sim_threshold", type=float, default=0.5)  # threshold

# XGBoost-specific params
parser.add_argument("--xgb_n_estimators", type=int, default=100)
parser.add_argument("--xgb_max_depth", type=int, default=6)
parser.add_argument("--xgb_lr", type=float, default=0.1)

args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(args.csv)
df["label"] = df["loan_status"].isin([
    "Charged Off", "Default", "Late (31-120 days)"
]).astype(int)
df.drop(columns=["loan_status"], inplace=True)

print(f"Total rows: {len(df)} | Positives: {df['label'].sum()}")

# ─── Optional subsample ──────────────────────────────────────────────────────
if args.max_rows is not None and args.max_rows < len(df):
    from sklearn.model_selection import train_test_split
    df, _ = train_test_split(
        df,
        train_size=args.max_rows,
        stratify=df["label"],
        random_state=SEED
    )
    df = df.reset_index(drop=True)

# ─── Split stream vs banks ───────────────────────────────────────────────────
stream_size = int(len(df) * args.stream_frac)
df_stream = df.sample(n=stream_size, random_state=SEED)
df_banks = df.drop(df_stream.index).reset_index(drop=True)
df_stream = df_stream.reset_index(drop=True)

# ─── Split into N banks ──────────────────────────────────────────────────────
df_banks = df_banks.sample(frac=1, random_state=SEED).reset_index(drop=True)
splits = np.array_split(np.arange(len(df_banks)), args.num_banks)
bank_dfs = [df_banks.iloc[idx].reset_index(drop=True) for idx in splits]

for i, bdf in enumerate(bank_dfs):
    print(f"Bank {i}: {len(bdf)} rows")
print(f"Stream: {len(df_stream)} rows")

# ─── Device ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ─── XGBoost params ──────────────────────────────────────────────────────────
xgb_params = {
    'n_estimators': args.xgb_n_estimators,
    'max_depth': args.xgb_max_depth,
    'learning_rate': args.xgb_lr,
}

# ─── Clients (HYBRID) ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Initializing HYBRID clients (XGBoost + GraphSAGE) …")
print(f"{'='*60}\n")

clients = [
    HybridFederatedClient(
        bank_id=i,
        df=bank_dfs[i],
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        lr=args.lr,
        replay_frac=args.replay_frac,
        k_default=args.k_default,
        xgb_params=xgb_params,
        device=device,
    )
    for i in range(args.num_banks)
]

print(f"\n{'='*60}")
print("XGBoost models trained (local, frozen)")
print("GraphSAGE + Fusion will be federated")
print(f"{'='*60}\n")

# ─── Server ──────────────────────────────────────────────────────────────────
server = CentralServer(
    clients[0].get_weights(),
    use_fedadam=args.use_fedadam,
    server_lr=args.server_lr,
)

# Broadcast initial weights
init_w = server.get_global_weights()
for c in clients:
    c.set_weights(init_w)

# ─── Stream ──────────────────────────────────────────────────────────────────
stream = StreamQueue(df_stream, n_banks=args.num_banks, batch_size=args.batch_size)

# ─── History ─────────────────────────────────────────────────────────────────
history = {
    "round": [],
    "global_auc": [],
    "global_ap": [],
    "global_xgb_auc": []  # Track XGBoost-only performance
}
for i in range(args.num_banks):
    history[f"bank{i}_auc"] = []
    history[f"bank{i}_ap"] = []
    history[f"bank{i}_xgb_auc"] = []

# ─── Evaluation helper ───────────────────────────────────────────────────────
def evaluate_all():
    results = [c.evaluate() for c in clients]
    
    # Hybrid metrics
    aucs = [r["AUC"] for r in results if r["AUC"] is not None]
    aps = [r["AP"] for r in results if r["AP"] is not None]
    
    # XGBoost-only metrics
    xgb_aucs = [r["XGB_AUC"] for r in results if r["XGB_AUC"] is not None]
    
    g_auc = float(np.mean(aucs)) if aucs else None
    g_ap = float(np.mean(aps)) if aps else None
    g_xgb_auc = float(np.mean(xgb_aucs)) if xgb_aucs else None
    
    return results, g_auc, g_ap, g_xgb_auc

# ─── Plot helper ─────────────────────────────────────────────────────────────
def save_plot(round_num):
    rounds = history["round"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # AUC (Hybrid)
    for i in range(args.num_banks):
        axes[0].plot(rounds, history[f"bank{i}_auc"], marker="o", label=f"Bank {i}")
    axes[0].plot(rounds, history["global_auc"], linestyle="--", linewidth=2, label="Global Hybrid", color="black")
    axes[0].legend(); axes[0].grid(True); axes[0].set_title("Hybrid AUC")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("AUC")
    
    # AP (Hybrid)
    for i in range(args.num_banks):
        axes[1].plot(rounds, history[f"bank{i}_ap"], marker="o", label=f"Bank {i}")
    axes[1].plot(rounds, history["global_ap"], linestyle="--", linewidth=2, label="Global Hybrid", color="black")
    axes[1].legend(); axes[1].grid(True); axes[1].set_title("Hybrid AP")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("AP")
    
    # Comparison: Hybrid vs XGBoost-only
    axes[2].plot(rounds, history["global_auc"], marker="o", label="Hybrid (XGB+SAGE)", linewidth=2)
    axes[2].plot(rounds, history["global_xgb_auc"], marker="s", label="XGBoost-only", linewidth=2, linestyle="--")
    axes[2].legend(); axes[2].grid(True); axes[2].set_title("Hybrid vs XGBoost-only")
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("AUC")
    
    plt.tight_layout()
    path = os.path.join(args.out_dir, f"round_{round_num:03d}.png")
    plt.savefig(path, dpi=100)
    plt.close()

# ─── Initial evaluation (Round 0) ────────────────────────────────────────────
print("\n" + "="*60)
print("Round 0 (Before Federation)")
print("="*60)
results, g_auc, g_ap, g_xgb_auc = evaluate_all()

history["round"].append(0)
history["global_auc"].append(g_auc)
history["global_ap"].append(g_ap)
history["global_xgb_auc"].append(g_xgb_auc)

for i, r in enumerate(results):
    history[f"bank{i}_auc"].append(r["AUC"])
    history[f"bank{i}_ap"].append(r["AP"])
    history[f"bank{i}_xgb_auc"].append(r["XGB_AUC"])
    print(f"Bank {i}: Hybrid AUC={r['AUC']:.4f}, AP={r['AP']:.4f} | XGB-only AUC={r['XGB_AUC']:.4f}")

print(f"\nGlobal: Hybrid AUC={g_auc:.4f}, AP={g_ap:.4f} | XGB-only AUC={g_xgb_auc:.4f}")
print(f"Hybrid improvement over XGB: {(g_auc - g_xgb_auc)*100:.2f}%")

save_plot(0)

# ─── Federation loop ─────────────────────────────────────────────────────────
for rnd in range(1, args.rounds + 1):
    print(f"\n{'='*60}")
    print(f"Round {rnd}/{args.rounds}")
    print("="*60)
    
    # Stream ingestion
    if not stream.is_empty():
        batches = stream.pop_batch()
        for i, (c, b) in enumerate(zip(clients, batches)):
            if not b.empty:
                c.add_new_data(b)
    
    # Broadcast
    global_w = server.get_global_weights()
    for c in clients:
        c.set_weights(global_w)
    
    # Local training (only GraphSAGE + Fusion, XGBoost frozen)
    for i, c in enumerate(clients):
        loss = c.local_train(global_w, mu=args.mu, epochs=args.local_epochs)
        print(f"Bank {i} loss: {loss:.4f}")
    
    # Aggregate
    w_list = [c.get_weights() for c in clients]
    n_list = [c.get_num_samples() for c in clients]
    server.aggregate(w_list, n_list)
    
    # Evaluate
    global_w = server.get_global_weights()
    for c in clients:
        c.set_weights(global_w)
    
    results, g_auc, g_ap, g_xgb_auc = evaluate_all()
    
    history["round"].append(rnd)
    history["global_auc"].append(g_auc)
    history["global_ap"].append(g_ap)
    history["global_xgb_auc"].append(g_xgb_auc)
    
    for i, r in enumerate(results):
        history[f"bank{i}_auc"].append(r["AUC"])
        history[f"bank{i}_ap"].append(r["AP"])
        history[f"bank{i}_xgb_auc"].append(r["XGB_AUC"])
        print(f"Bank {i}: Hybrid AUC={r['AUC']:.4f}, AP={r['AP']:.4f} | XGB-only AUC={r['XGB_AUC']:.4f}")
    
    print(f"\nGlobal: Hybrid AUC={g_auc:.4f}, AP={g_ap:.4f} | XGB-only AUC={g_xgb_auc:.4f}")
    improvement = (g_auc - g_xgb_auc) * 100
    print(f"Hybrid improvement over XGB: {improvement:+.2f}%")
    
    save_plot(rnd)

# ─── Final summary ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FEDERATION COMPLETE")
print("="*60)

final_hybrid_auc = history["global_auc"][-1]
final_hybrid_ap = history["global_ap"][-1]
final_xgb_auc = history["global_xgb_auc"][-1]

print(f"\nFinal Global Metrics:")
print(f"  Hybrid (XGB+SAGE) AUC: {final_hybrid_auc:.4f}")
print(f"  Hybrid (XGB+SAGE) AP:  {final_hybrid_ap:.4f}")
print(f"  XGBoost-only AUC:      {final_xgb_auc:.4f}")
print(f"  Improvement:           {(final_hybrid_auc - final_xgb_auc)*100:+.2f}%")

print("\nPer-Bank Final Results:")
for i in range(args.num_banks):
    hybrid_auc = history[f"bank{i}_auc"][-1]
    hybrid_ap = history[f"bank{i}_ap"][-1]
    xgb_auc = history[f"bank{i}_xgb_auc"][-1]
    print(f"  Bank {i}: Hybrid AUC={hybrid_auc:.4f}, AP={hybrid_ap:.4f} | "
          f"XGB AUC={xgb_auc:.4f} | Δ={hybrid_auc - xgb_auc:+.4f}")

# ─── Save metrics ────────────────────────────────────────────────────────────
import json
with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\nResults saved to {args.out_dir}/")
print("Done.")
