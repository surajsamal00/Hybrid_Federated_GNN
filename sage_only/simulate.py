"""
simulate.py
-----------
Main entry point for the federated simulation (dynamic number of banks).
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

from client import FederatedClient
from server import CentralServer
from stream import StreamQueue

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── CLI args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Federated GraphSAGE simulation")
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--local_epochs", type=int, default=5)
parser.add_argument("--mu", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=75)
parser.add_argument("--stream_frac", type=float, default=0.15)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--replay_frac", type=float, default=0.2)
parser.add_argument("--use_fedadam", action="store_true", default=True)
parser.add_argument("--no_fedadam", dest="use_fedadam", action="store_false")
parser.add_argument("--server_lr", type=float, default=1e-2)
parser.add_argument("--max_rows", type=int, default=None)
parser.add_argument("--out_dir", type=str, default="./fed_results")
parser.add_argument("--num_banks", type=int, default=3)
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

# ─── Clients ─────────────────────────────────────────────────────────────────
print("Initializing clients …")
clients = [
    FederatedClient(
        bank_id=i,
        df=bank_dfs[i],
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        lr=args.lr,
        replay_frac=args.replay_frac,
        device=device,
    )
    for i in range(args.num_banks)
]

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
history = {"round": [], "global_auc": [], "global_ap": []}
for i in range(args.num_banks):
    history[f"bank{i}_auc"] = []
    history[f"bank{i}_ap"] = []

# ─── Evaluation helper ───────────────────────────────────────────────────────
def evaluate_all():
    results = [c.evaluate() for c in clients]
    aucs = [r["AUC"] for r in results if r["AUC"] is not None]
    aps = [r["AP"] for r in results if r["AP"] is not None]
    g_auc = float(np.mean(aucs)) if aucs else None
    g_ap = float(np.mean(aps)) if aps else None
    return results, g_auc, g_ap

# ─── Plot helper ─────────────────────────────────────────────────────────────
def save_plot(round_num):
    rounds = history["round"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AUC
    for i in range(args.num_banks):
        axes[0].plot(rounds, history[f"bank{i}_auc"], marker="o", label=f"Bank {i}")
    axes[0].plot(rounds, history["global_auc"], linestyle="--", label="Global")
    axes[0].legend(); axes[0].grid(True); axes[0].set_title("AUC")

    # AP
    for i in range(args.num_banks):
        axes[1].plot(rounds, history[f"bank{i}_ap"], marker="o", label=f"Bank {i}")
    axes[1].plot(rounds, history["global_ap"], linestyle="--", label="Global")
    axes[1].legend(); axes[1].grid(True); axes[1].set_title("AP")

    path = os.path.join(args.out_dir, f"round_{round_num:03d}.png")
    plt.savefig(path)
    plt.close()

# ─── Federation loop ─────────────────────────────────────────────────────────
for rnd in range(1, args.rounds + 1):
    print(f"\nRound {rnd}/{args.rounds}")

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

    # Local training
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

    results, g_auc, g_ap = evaluate_all()

    history["round"].append(rnd)
    history["global_auc"].append(g_auc)
    history["global_ap"].append(g_ap)

    for i, r in enumerate(results):
        history[f"bank{i}_auc"].append(r["AUC"])
        history[f"bank{i}_ap"].append(r["AP"])
        print(f"Bank {i}: AUC={r['AUC']:.4f}, AP={r['AP']:.4f}")

    print(f"Global: AUC={g_auc:.4f}, AP={g_ap:.4f}")

    save_plot(rnd)

# ─── Save metrics ────────────────────────────────────────────────────────────
import json
with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
    json.dump(history, f, indent=2)

print("Done.")
