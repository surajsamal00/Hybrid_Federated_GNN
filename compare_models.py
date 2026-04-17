"""
compare_models.py
-----------------
Compare Hybrid vs XGBoost-only vs GraphSAGE-only results.

Usage:
    python compare_models.py \
        --hybrid_metrics ./hybrid_fed_results/metrics.json \
        --sage_metrics ./fed_results/metrics.json \
        --out_plot ./comparison.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Compare Hybrid vs Single Models")
parser.add_argument("--hybrid_metrics", type=str, required=True,
                    help="Path to hybrid model metrics.json")
parser.add_argument("--sage_metrics", type=str, default=None,
                    help="Path to GraphSAGE-only metrics.json (optional)")
parser.add_argument("--out_plot", type=str, default="./model_comparison.png",
                    help="Output plot path")
args = parser.parse_args()

# Load hybrid results
with open(args.hybrid_metrics, "r") as f:
    hybrid = json.load(f)

# Load SAGE-only results if provided
sage = None
if args.sage_metrics:
    with open(args.sage_metrics, "r") as f:
        sage = json.load(f)

# Extract data
rounds = hybrid["round"]
hybrid_auc = hybrid["global_auc"]
xgb_auc = hybrid["global_xgb_auc"]

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ─── Plot 1: AUC Comparison ──────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(rounds, hybrid_auc, marker='o', label='Hybrid (XGB+SAGE)', 
        linewidth=2, markersize=5, color='green')
ax.plot(rounds, xgb_auc, marker='s', label='XGBoost-only', 
        linewidth=2, markersize=5, linestyle='--', color='blue')
if sage:
    ax.plot(sage["round"], sage["global_auc"], marker='^', 
            label='GraphSAGE-only', linewidth=2, markersize=5, 
            linestyle=':', color='orange')

ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("Model Comparison: AUC Over Rounds", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ─── Plot 2: AP Comparison ───────────────────────────────────────────────────
ax = axes[0, 1]
hybrid_ap = hybrid["global_ap"]
ax.plot(rounds, hybrid_ap, marker='o', label='Hybrid (XGB+SAGE)', 
        linewidth=2, markersize=5, color='green')
if sage:
    ax.plot(sage["round"], sage["global_ap"], marker='^', 
            label='GraphSAGE-only', linewidth=2, markersize=5, 
            linestyle=':', color='orange')

ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("AP", fontsize=12)
ax.set_title("Model Comparison: AP Over Rounds", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ─── Plot 3: Improvement Over XGBoost ────────────────────────────────────────
ax = axes[1, 0]
improvement = [(h - x) * 100 for h, x in zip(hybrid_auc, xgb_auc)]
ax.plot(rounds, improvement, marker='o', linewidth=2, markersize=5, color='green')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("AUC Improvement (%)", fontsize=12)
ax.set_title("Hybrid Improvement Over XGBoost-Only", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Fill positive improvements
positive_mask = np.array(improvement) > 0
if positive_mask.any():
    ax.fill_between(rounds, 0, improvement, where=positive_mask, 
                     alpha=0.3, color='green', label='Improvement')
ax.legend(fontsize=10)

# ─── Plot 4: Final Performance Bar Chart ─────────────────────────────────────
ax = axes[1, 1]

models = ['XGBoost\nOnly', 'Hybrid\n(XGB+SAGE)']
final_aucs = [xgb_auc[-1], hybrid_auc[-1]]
colors = ['blue', 'green']

if sage:
    models.insert(1, 'GraphSAGE\nOnly')
    final_aucs.insert(1, sage["global_auc"][-1])
    colors.insert(1, 'orange')

bars = ax.bar(models, final_aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, auc in zip(bars, final_aucs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{auc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel("Final AUC", fontsize=12)
ax.set_title("Final Model Performance", fontsize=14, fontweight='bold')
ax.set_ylim([min(final_aucs) - 0.02, max(final_aucs) + 0.02])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(args.out_plot, dpi=150, bbox_inches='tight')
print(f"\n{'='*60}")
print("COMPARISON SAVED")
print(f"{'='*60}")
print(f"Plot saved to: {args.out_plot}\n")

# ─── Print Summary Statistics ────────────────────────────────────────────────
print(f"{'='*60}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*60}\n")

print(f"{'Model':<20} {'Final AUC':>12} {'Final AP':>12} {'Δ vs XGB':>12}")
print("-" * 60)

xgb_final = xgb_auc[-1]
hybrid_final = hybrid_auc[-1]
hybrid_ap_final = hybrid_ap[-1]

print(f"{'XGBoost-only':<20} {xgb_final:>12.4f} {'N/A':>12} {'baseline':>12}")
print(f"{'Hybrid (XGB+SAGE)':<20} {hybrid_final:>12.4f} {hybrid_ap_final:>12.4f} {f'+{(hybrid_final-xgb_final)*100:.2f}%':>12}")

if sage:
    sage_final = sage["global_auc"][-1]
    sage_ap_final = sage["global_ap"][-1]
    print(f"{'GraphSAGE-only':<20} {sage_final:>12.4f} {sage_ap_final:>12.4f} {f'{(sage_final-xgb_final)*100:+.2f}%':>12}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60 + "\n")

delta = (hybrid_final - xgb_final) * 100

if delta > 2.0:
    print("✅ STRONG RESULT: Hybrid significantly outperforms XGBoost")
    print(f"   → {delta:.2f}% improvement demonstrates value of graph structure")
elif delta > 0.5:
    print("✅ GOOD RESULT: Hybrid improves over XGBoost baseline")
    print(f"   → {delta:.2f}% improvement shows complementary signals")
elif delta > -0.5:
    print("⚠️  MARGINAL: Hybrid ≈ XGBoost (within measurement noise)")
    print("   → Consider tuning fusion layer or graph construction")
else:
    print("❌ CONCERN: Hybrid underperforms XGBoost")
    print("   → Debug: Check graph connectivity, fusion layer learning")

if sage:
    sage_delta = (hybrid_final - sage_final) * 100
    print(f"\nHybrid vs GraphSAGE-only: {sage_delta:+.2f}%")
    if sage_delta > 2.0:
        print("   → XGBoost contribution is significant")
    elif abs(sage_delta) < 1.0:
        print("   → GraphSAGE dominates, XGBoost adds little")

print("\n" + "="*60)
print("CONVERGENCE ANALYSIS")
print("="*60 + "\n")

# Check convergence
early_auc = hybrid_auc[min(5, len(rounds)-1)]
mid_auc = hybrid_auc[len(rounds)//2]
final_auc = hybrid_auc[-1]

if final_auc > mid_auc > early_auc:
    print("✅ Monotonic improvement — model is learning steadily")
elif final_auc - early_auc > 0.01:
    print("✅ Overall improvement despite fluctuations")
else:
    print("⚠️  Limited learning — model plateaued early")
    print("   → Consider: more rounds, higher learning rate, or architecture changes")

print(f"\nRound 5 AUC:   {early_auc:.4f}")
print(f"Round {len(rounds)//2} AUC:  {mid_auc:.4f}")
print(f"Final AUC:     {final_auc:.4f}")
print(f"Total gain:    {(final_auc - early_auc)*100:+.2f}%")

print("\n" + "="*60)
