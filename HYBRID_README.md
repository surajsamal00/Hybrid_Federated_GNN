# Hybrid XGBoost + GraphSAGE Federated Learning

## 🎯 Architecture Overview

This implementation combines **XGBoost** (tabular learning) with **GraphSAGE** (graph learning) in a federated setting for credit default prediction.

### Two-Branch Design

```
Customer Data (57 features + 1 density)
         |
         ├─────────────────┬─────────────────┐
         |                 |                 |
    [XGBoost]         [GraphSAGE]            |
    (LOCAL)           (FEDERATED)            |
         |                 |                 |
    Risk Score        Graph Embedding        |
    (1 value)         (128-dim)              |
         |                 |                 |
         └─────────────────┴─────────────────┘
                           |
                    [Fusion Layer]
                     (FEDERATED)
                           |
                    Final Prediction
```

### Key Components

1. **XGBoost Branch (Local)**
   - Trains on 57 tabular features (no graph structure)
   - Captures individual customer risk
   - **Stays local** — never shared across banks
   - Provides 1 risk score per customer

2. **GraphSAGE Branch (Federated)**
   - Uses 58 features (57 + density)
   - Leverages graph structure (customer similarity)
   - Captures network/contagion effects
   - Returns 128-dimensional embedding

3. **Fusion Layer (Federated)**
   - Combines XGBoost score + GraphSAGE embedding
   - Learnable MLP: (128 + 1) → 64 → 1
   - Learns optimal weighting between branches

## 📊 What Gets Federated?

| Component | Shared Across Banks | Training |
|-----------|---------------------|----------|
| **XGBoost models** | No (privacy is preserved) | Once, frozen |
| **GraphSAGE parameters** | ✅ Yes (via FedProx + FedAdam) | Every round |
| **Fusion layer** | ✅ Yes (via FedProx + FedAdam) | Every round |

## 🔧 Usage

### Basic Run (4 banks, 40 rounds)

```bash
python hybrid_simulate.py \
  --csv /path/to/lending_club.csv \
  --num_banks 4 \
  --rounds 40 \
  --hidden_dim 128 \
  --emb_dim 128 \
  --mu 0.01 \
  --out_dir ./hybrid_results
```

### Full Configuration

```bash
python hybrid_simulate.py \
  --csv /path/to/lending_club.csv \
  --num_banks 4 \
  --rounds 40 \
  --local_epochs 10 \
  --mu 0.01 \
  --batch_size 75 \
  --stream_frac 0.15 \
  --hidden_dim 128 \
  --emb_dim 128 \
  --dropout 0.3 \
  --lr 1e-3 \
  --server_lr 1e-3 \
  --use_fedadam \
  --xgb_n_estimators 100 \
  --xgb_max_depth 6 \
  --xgb_lr 0.1 \
  --out_dir ./hybrid_results
```

### Key Parameters

**Model Architecture:**
- `--hidden_dim`: GraphSAGE hidden dimension (default: 128)
- `--emb_dim`: GraphSAGE output embedding dimension (default: 128)
- `--dropout`: Dropout rate (default: 0.3)

**XGBoost Settings:**
- `--xgb_n_estimators`: Number of trees (default: 100)
- `--xgb_max_depth`: Tree depth (default: 6)
- `--xgb_lr`: XGBoost learning rate (default: 0.1)

**Federation:**
- `--num_banks`: Number of banks (default: 4)
- `--rounds`: Number of federation rounds (default: 30)
- `--mu`: FedProx regularization strength (default: 0.01)
- `--server_lr`: FedAdam server learning rate (default: 1e-3)

## 📈 Expected Performance

Based on your setup (4 banks, 128→128 architecture):

| Model | Expected AUC | Expected AP |
|-------|--------------|-------------|
| **XGBoost-only** | 0.68-0.70 | 0.24-0.26 |
| **GraphSAGE-only** | 0.66-0.68 | 0.22-0.24 |
| **Hybrid (XGB+SAGE)** | **0.70-0.73** | **0.26-0.28** |

**Why Hybrid Works:**
- XGBoost excels at individual risk (credit score, income, DTI)
- GraphSAGE captures contagion (neighbors' default patterns)
- Fusion layer learns optimal combination

## 🔍 Output Analysis

The script generates:

1. **Plots** (`round_XXX.png`):
   - Panel 1: Hybrid AUC per bank + global
   - Panel 2: Hybrid AP per bank + global
   - Panel 3: **Hybrid vs XGBoost-only comparison**

2. **Metrics** (`metrics.json`):
   - `global_auc`: Hybrid model AUC
   - `global_xgb_auc`: XGBoost-only baseline
   - Per-bank metrics for both models

3. **Console Output**:
   ```
   Bank 0: Hybrid AUC=0.7123, AP=0.2678 | XGB-only AUC=0.6845
   Global: Hybrid AUC=0.7089, AP=0.2654 | XGB-only AUC=0.6798
   Hybrid improvement over XGB: +2.91%
   ```

## 🧪 Ablation Study

To understand component contributions:

```bash
# 1. XGBoost-only baseline (already computed in hybrid run)
# Check metrics.json -> global_xgb_auc

# 2. GraphSAGE-only (use original simulate.py)
python simulate.py --csv data.csv --num_banks 4 --rounds 30

# 3. Hybrid (this script)
python hybrid_simulate.py --csv data.csv --num_banks 4 --rounds 30
```

Compare final AUCs:
- If Hybrid > XGB and Hybrid > SAGE → **Strong result**
- Typical gap: Hybrid should be 2-5% better than best single model

## 🔬 Technical Details

### Training Flow

**Round 0 (Initialization):**
1. Each bank trains local XGBoost on 57 tabular features
2. XGBoost generates risk scores for all customers
3. GraphSAGE + Fusion initialized randomly
4. Evaluate: Get baseline XGB-only AUC

**Rounds 1-30 (Federation):**
1. Broadcast global GraphSAGE + Fusion weights
2. Each bank:
   - Forward pass: GraphSAGE embedding + frozen XGBoost scores
   - Loss: BCE + FedProx penalty
   - Backprop: Update only GraphSAGE + Fusion (XGB frozen)
3. Server aggregates using FedAdam
4. Evaluate: Compare Hybrid vs XGB-only

### Leakage Prevention

All leakage fixes from original `client.py` are preserved:

✅ **Train/test split BEFORE scaling**
✅ **Scaler fit on train only**
✅ **Graph built on scaled features**
✅ **Density computed on full graph, not just training**
✅ **No leaky columns** (57 features from 41 base + 16 engineered)
✅ **Streaming data properly integrated**

### Privacy Guarantees

- **XGBoost models never leave the bank** ✅
- Only XGBoost *predictions* (scalars) are used, not model weights
- GraphSAGE + Fusion parameters aggregated via FedProx + FedAdam
- Each bank maintains its own credit policy (XGB) while benefiting from cross-bank graph structure (SAGE)

## 📝 Model Overview

> "We propose a hybrid federated learning approach combining XGBoost for individual risk assessment with GraphSAGE for network contagion detection. Local XGBoost models capture bank-specific credit policies and remain private, while federated GraphSAGE leverages cross-bank customer similarity graphs to propagate default risk signals. Our fusion layer learns to optimally combine individual and network risk, achieving [X]% improvement over single-model baselines while preserving data privacy across 4 simulated financial institutions."

**Key Factors:**
1. **Privacy-preserving hybrid architecture** — XGB local, SAGE federated
2. **Complementary risk signals** — Individual (XGB) + Contagion (SAGE)
3. **Empirical validation** — Show Hybrid > XGB-only and Hybrid > SAGE-only
4. **Scalability** — Demonstrate with 4 banks, ~35k customers each

## 🐛 Troubleshooting

**Issue: XGBoost AUC much higher than Hybrid**
- Check fusion layer learning rate (might be too high/low)
- Verify XGBoost scores are properly passed to forward()
- Ensure pos_weight is set correctly in BCE loss

**Issue: Hybrid ≈ XGBoost (no improvement)**
- GraphSAGE may not be learning useful patterns
- Check graph connectivity (avg degree > 5)
- Try increasing hidden_dim or emb_dim
- Verify graph edges are being created

**Issue: GPU memory error**
- Reduce batch_size
- Reduce hidden_dim or emb_dim
- Use smaller XGBoost (fewer trees)

## 📚 Files

- `hybrid_client.py` — HybridFederatedClient class
- `hybrid_simulate.py` — Main training script
- `server.py` — FedAdam aggregation
- `stream.py` — Streaming data 
- `loan_cleaned.csv` — Customer data (to be downloaded from lending club website)
