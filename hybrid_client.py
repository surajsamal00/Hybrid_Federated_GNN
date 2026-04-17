import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree, subgraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ─── FAISS (GPU → CPU → sklearn fallback) ────────────────────────────────────
_HAS_FAISS = False
_HAS_FAISS_GPU = False
try:
    import faiss
    _HAS_FAISS = True
    try:
        if hasattr(faiss, "StandardGpuResources"):
            _HAS_FAISS_GPU = True
    except Exception:
        pass
except Exception:
    pass

if not _HAS_FAISS:
    from sklearn.neighbors import NearestNeighbors

# ─── Constants ───────────────────────────────────────────────────────────────
LEAKY = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "last_credit_pull_d", "last_fico_range_low",
    "last_fico_range_high", "collections_12_mths_ex_med",
    "acc_now_delinq", "chargeoff_within_12_mths", "delinq_amnt",
    "mths_since_recent_bc", "mths_since_recent_inq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "hardship_flag",
    "debt_settlement_flag", "disbursement_method", "url", "title",
    "pymnt_plan", "loan_status",
]
DATE_COLS = ["issue_d", "earliest_cr_line", "hardship_flag", "debt_settlement_flag"]


# ─── Feature Engineering ─────────────────────────────────────────────────────
def add_features(df):
    """
    16 engineered features from the 41 surviving numeric columns.
    All derived from application-time data only — zero leakage risk.

    Groups:
      A. Affordability   — joint income / debt stress
      B. Credit quality  — FICO encoded more directly
      C. Utilisation     — revolving credit behaviour
      D. Derogatory      — adverse event accumulation
      E. Loan structure  — rate / size interactions
    """
    d = df.copy()

    # ── A. Affordability ─────────────────────────────────────────────────────
    d["monthly_inc"]      = d["annual_inc"] / 12.0
    d["install_to_inc"]   = d["installment"] / (d["monthly_inc"] + 1.0)
    d["dti_x_rate"]       = d["dti"] * d["int_rate"]
    d["loan_to_income"]   = d["loan_amnt"] / (d["annual_inc"] + 1.0)
    d["fund_ratio"]       = d["funded_amnt_inv"] / (d["funded_amnt"] + 1.0)

    # ── B. Credit quality ────────────────────────────────────────────────────
    d["fico_avg"]         = (d["fico_range_low"] + d["fico_range_high"]) / 2.0
    d["fico_deficit"]     = d["fico_avg"] - 700.0
    fico_norm             = d["fico_deficit"].clip(-200, 200) / 200.0
    d["fico_dti_stress"]  = (1.0 - fico_norm) * d["dti"]

    # ── C. Utilisation ───────────────────────────────────────────────────────
    d["revol_stress"]     = (d["revol_util"].clip(0, 200) / 100.0) * np.log1p(d["revol_bal"])
    d["open_acc_ratio"]   = d["open_acc"] / (d["total_acc"] + 1.0)
    d["recent_acc_ratio"] = d["acc_open_past_24mths"] / (d["total_acc"] + 1.0)

    # ── D. Derogatory ────────────────────────────────────────────────────────
    d["adverse_score"]    = (
        d["delinq_2yrs"].clip(0, 10)             * 2.0
        + d["inq_last_6mths"].clip(0, 10)        * 1.0
        + d["pub_rec"].clip(0, 5)                * 3.0
        + d["pub_rec_bankruptcies"].clip(0, 3)   * 5.0
        + d["num_accts_ever_120_pd"].clip(0, 20) * 2.0
    )
    d["delinq_rate"]      = d["num_accts_ever_120_pd"] / (d["total_acc"] + 1.0)
    d["dlq_exposure"]     = 1.0 - (d["pct_tl_nvr_dlq"].clip(0, 100) / 100.0)

    # ── E. Loan structure ────────────────────────────────────────────────────
    d["total_interest_proxy"] = (d["int_rate"] / 100.0) * d["loan_amnt"]
    d["install_to_loan"]  = d["installment"] / (d["loan_amnt"] + 1.0)

    new_cols = [
        "monthly_inc", "install_to_inc", "dti_x_rate", "loan_to_income",
        "fund_ratio", "fico_avg", "fico_deficit", "fico_dti_stress",
        "revol_stress", "open_acc_ratio", "recent_acc_ratio",
        "adverse_score", "delinq_rate", "dlq_exposure",
        "total_interest_proxy", "install_to_loan",
    ]
    d[new_cols] = d[new_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return d


# ─── GraphSAGE model (original) ──────────────────────────────────────────────
class SAGEClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn2   = nn.BatchNorm1d(hidden_dim // 2)

        self.conv3 = SAGEConv(hidden_dim // 2, out_dim)
        self.bn3   = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.dropout(F.relu(self.bn3(self.conv3(x, edge_index))))
        return self.head(x).squeeze(1)

    @torch.no_grad()
    def get_embeddings(self, x, edge_index):
        self.eval()
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        return x


# ─── HYBRID MODEL: XGBoost + GraphSAGE ───────────────────────────────────────
class HybridSAGEClassifier(nn.Module):
    """
    Combines XGBoost (tabular) + GraphSAGE (graph) predictions.
    
    Architecture:
    - GraphSAGE: Returns embeddings (out_dim dimensional)
    - XGBoost: Pre-computed risk score (1 dimensional)
    - Fusion: Learnable combination layer
    
    Forward flow:
    1. GraphSAGE processes graph → embedding (N, out_dim)
    2. XGBoost score (pre-computed) → (N, 1)
    3. Concatenate → (N, out_dim + 1)
    4. Fusion MLP → (N, 1) final prediction
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, dropout=0.3):
        super().__init__()
        
        # GraphSAGE branch (same as original but returns embeddings)
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = SAGEConv(hidden_dim, out_dim)
        self.bn3   = nn.BatchNorm1d(out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Fusion layer: combines GraphSAGE embedding + XGBoost score
        self.fusion = nn.Sequential(
            nn.Linear(out_dim + 1, 64),  # out_dim (SAGE) + 1 (XGB) → 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, edge_index, xgb_scores=None):
        """
        Parameters
        ----------
        x          : Tensor (N, in_dim) - node features
        edge_index : Tensor (2, E) - graph edges
        xgb_scores : Tensor (N,) or None - XGBoost predictions
                     If None, fusion layer uses zero for XGB component
        
        Returns
        -------
        logits : Tensor (N,) - final predictions
        """
        # GraphSAGE embedding
        h = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        h = self.dropout(F.relu(self.bn2(self.conv2(h, edge_index))))
        h = self.dropout(F.relu(self.bn3(self.conv3(h, edge_index))))  # (N, out_dim)
        
        if xgb_scores is None:
            # Fallback: use zeros (GraphSAGE-only mode)
            xgb_scores = torch.zeros(h.size(0), 1, device=h.device)
        else:
            # Ensure XGB scores are (N, 1)
            if xgb_scores.dim() == 1:
                xgb_scores = xgb_scores.unsqueeze(1)
        
        # Concatenate GraphSAGE embedding + XGBoost score
        combined = torch.cat([h, xgb_scores], dim=1)  # (N, out_dim + 1)
        
        # Fusion layer
        logits = self.fusion(combined).squeeze(1)  # (N,)
        
        return logits
    
    @torch.no_grad()
    def get_embeddings(self, x, edge_index):
        """Get GraphSAGE embeddings without classification head"""
        self.eval()
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        return h


# ─── FAISS kNN ────────────────────────────────────────────────────
def build_knn_graph(X, y=None, k_default=15, sim_threshold=0.5, batch_size=4096, make_undirected=True):
    """
    Cosine-similarity kNN graph using FAISS.
    - Uniform k for all nodes
    - Semi-mutual filtering for noise reduction
    """

    n, d = X.shape

    # --- Normalize ---
    Xn = X.astype(np.float32).copy()
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn /= norms

    # --- Storage for neighbors ---
    neighbors = [set() for _ in range(n)]
    sims_dict = [{} for _ in range(n)]

    # ================= FAISS =================
    if _HAS_FAISS:
        index_cpu = faiss.IndexFlatIP(d)

        if _HAS_FAISS_GPU:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        else:
            index = index_cpu

        index.add(Xn)
        max_k = k_default + 1

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            q = Xn[start:end]

            D, I = index.search(q, max_k)

            for li in range(q.shape[0]):
                i = start + li

                nids = I[li, 1:k_default+1]     # skip self
                nsims = D[li, 1:k_default+1]

                for j, sim in zip(nids, nsims):
                    if sim >= sim_threshold:
                        j = int(j)
                        neighbors[i].add(j)
                        sims_dict[i][j] = float(sim)

    # ================= sklearn fallback =================
    else:
        nbrs = NearestNeighbors(n_neighbors=k_default+1, metric="cosine", n_jobs=-1)
        nbrs.fit(Xn)
        distances, indices = nbrs.kneighbors(Xn)

        for i in range(n):
            nids = indices[i, 1:k_default+1]
            sims = 1.0 - distances[i, 1:k_default+1]

            for j, sim in zip(nids, sims):
                if sim >= sim_threshold:
                    j = int(j)
                    neighbors[i].add(j)
                    sims_dict[i][j] = float(sim)

    # ================= Semi-mutual filtering =================
    edges_src, edges_dst = [], []
    strong_threshold = sim_threshold + 0.1

    for i in range(n):
        for j in neighbors[i]:
            sim = sims_dict[i][j]

            cond1 = i in neighbors[j]        # mutual
            cond2 = sim >= strong_threshold # strong similarity fallback

            if cond1 or cond2:
                edges_src.append(i)
                edges_dst.append(j)

                if make_undirected:
                    edges_src.append(j)
                    edges_dst.append(i)

    # ================= Final =================
    print("Avg degree:", len(edges_src) / n) 
    print("Normal-> 5-20")
    if not edges_src:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.stack([
        torch.tensor(edges_src, dtype=torch.long),
        torch.tensor(edges_dst, dtype=torch.long)
    ], dim=0)


# ─── HYBRID FEDERATED CLIENT ─────────────────────────────────────────────────
class HybridFederatedClient:
    """
    Federated client with XGBoost + GraphSAGE hybrid model.
    
    Training Strategy:
    1. Train XGBoost locally (once, frozen)
    2. Use XGBoost predictions as input to GraphSAGE fusion layer
    3. Federate only GraphSAGE + fusion parameters (XGBoost stays local)
    
    Privacy Preservation:
    - XGBoost models never leave the bank
    - Only neural network parameters are aggregated
    """
    
    def __init__(self, bank_id, df, hidden_dim=128, emb_dim=128, dropout=0.3,
                 lr=1e-3, replay_frac=0.2, k_default=15,
                 xgb_params=None, device=None):
        
        self.bank_id     = bank_id
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr          = lr
        self.replay_frac = replay_frac
        self.k           = k_default
        
        # ── 1. Clean + engineer ───────────────────────────────────────────────
        X_df_cleaned       = self._clean_df(df)
        self.feature_names = X_df_cleaned.columns.tolist()
        X_raw              = X_df_cleaned.values.astype(np.float32)
        self.y             = df["label"].astype(int).values
        
        print(f"[Bank {bank_id}] Feature dim: {X_raw.shape[1]} "
              f"(41 base + 16 engineered = 57; +1 density = 58 total)")
        
        # ── 2. Train / test split BEFORE scaling ──────────────────────────────
        idx = np.arange(len(self.y))
        self.train_idx, self.test_idx = train_test_split(
            idx, test_size=0.2, stratify=self.y, random_state=42
        )
        
        # ── 3. Scaler fit on train only, transform all ────────────────────────
        self.scaler = StandardScaler()
        self.scaler.fit(X_raw[self.train_idx])
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)
        
        # ── 4. Single unified label-free graph on scaled features ─────────────
        full_edge_index_raw = build_knn_graph(X_scaled, y=None, k_default= self.k)
        
        # ── 5. Density for ALL nodes (train + test) ───────────────────────────
        deg_all     = degree(full_edge_index_raw[0], num_nodes=len(self.y))
        density_all = (deg_all / (deg_all.max() + 1e-6)).cpu().numpy().reshape(-1, 1)
        self.X      = np.hstack([X_scaled, density_all])
        
        avg_deg = deg_all.mean().item()
        print(f"[Bank {bank_id}] Avg neighbors/node: {avg_deg:.1f}"
              + ("  ⚠ sparse — lower sim_threshold" if avg_deg < 5 else "  ✓"))
        
        # ── 6. Training subgraph with relabelled local indices ─────────────────
        train_mask = torch.zeros(self.X.shape[0], dtype=torch.bool)
        train_mask[self.train_idx] = True
        self.train_edge_index, _ = subgraph(
            train_mask, full_edge_index_raw,
            relabel_nodes=True, num_nodes=self.X.shape[0]
        )
        
        # ── 7. Full inference graph on final 58-feature self.X ────────────────
        self.full_edge_index = build_knn_graph(self.X, y=None, k_default=self.k)
        
        # ── 8. XGBoost Model (LOCAL ONLY) ─────────────────────────────────────
        print(f"[Bank {bank_id}] Training local XGBoost...")
        
        # Default XGBoost params (can override)
        default_xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': -1
        }
        if xgb_params:
            default_xgb_params.update(xgb_params)
        
        # Calculate class weight for imbalance
        pos_count = self.y[self.train_idx].sum()
        neg_count = len(self.train_idx) - pos_count
        scale_pos_weight = neg_count / (pos_count + 1e-9)
        default_xgb_params['scale_pos_weight'] = min(scale_pos_weight, 20.0)
        
        # Train XGBoost on tabular features ONLY (no density, no graph)
        self.xgb_model = xgb.XGBClassifier(**default_xgb_params)
        self.xgb_model.fit(
            X_scaled[self.train_idx],  # 57 features (no density)
            self.y[self.train_idx],
            verbose=False
        )
        
        # Pre-compute XGBoost scores for ALL data (train + test)
        self._update_xgb_scores()
        
        xgb_train_auc = roc_auc_score(
            self.y[self.train_idx],
            self.xgb_train_scores
        )
        print(f"[Bank {bank_id}] XGBoost train AUC: {xgb_train_auc:.4f}")
        
        # ── 9. Hybrid Model (FEDERATED) ───────────────────────────────────────
        self.model = HybridSAGEClassifier(
            in_dim=self.X.shape[1],      # 58 (57 + density)
            hidden_dim=hidden_dim,
            out_dim=emb_dim,
            dropout=dropout,
        ).to(self.device)

        sage_params = []
        for name, param in self.model.named_parameters():
            if 'conv' in name or 'bn' in name:
                sage_params.append(param)

        fusion_params = list(self.model.fusion.parameters())

        self.optimizer = optim.Adam([
            {'params': sage_params, 'lr': self.lr * 2.0},    # GraphSAGE: 2x faster
            {'params': fusion_params, 'lr': self.lr * 0.5}   # Fusion: 2x slower
        ], lr=self.lr)
        print(f"[Bank {bank_id}] SAGE LR={self.lr * 2.0:.5f}, Fusion LR={self.lr * 0.5:.5f}")
    
    def _clean_df(self, df):
        """Remove leaky columns and engineer features"""
        d = df.copy()
        d = d[[c for c in d.columns if c not in LEAKY and c != "label"]]
        
        for c in DATE_COLS:
            if c in d.columns:
                d.drop(columns=[c], inplace=True)
        
        d = add_features(d)
        
        # Drop non-numeric
        num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        d = d[num_cols]
        
        # Fill missing
        d.fillna(0, inplace=True)
        
        return d
    
    def _update_xgb_scores(self):
        """
        Pre-compute XGBoost scores for all data.
        CRITICAL: Use only 57 tabular features (no density feature).
        """
        X_tabular = self.X[:, :-1]  # Remove density column
        
        # Get probabilities for all nodes
        xgb_probs_all = self.xgb_model.predict_proba(X_tabular)[:, 1]
        
        # Split into train/test for easy access
        self.xgb_train_scores = xgb_probs_all[self.train_idx]
        self.xgb_test_scores  = xgb_probs_all[self.test_idx]
        self.xgb_all_scores   = xgb_probs_all
    
    # ── Federated API ─────────────────────────────────────────────────────────
    def set_weights(self, global_state_dict):
        """Load global model weights (GraphSAGE + fusion only)"""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))
    
    def get_weights(self):
        """Return current model weights (GraphSAGE + fusion only)"""
        return copy.deepcopy(self.model.state_dict())
    
    def get_num_samples(self):
        """Number of training samples"""
        return len(self.train_idx)
    
    def local_train(self, global_state_dict, mu=0.01, epochs=5):
        """
        Local training with FedProx.
        XGBoost frozen, only GraphSAGE + fusion updated.
        """
        self.model.load_state_dict(copy.deepcopy(global_state_dict))
        self.model.train()
        
        x_train = torch.tensor(self.X[self.train_idx]).to(self.device)
        y_train = torch.tensor(
            self.y[self.train_idx], dtype=torch.float32
        ).to(self.device)
        
        # XGBoost scores for training data
        xgb_train_tensor = torch.tensor(
            self.xgb_train_scores, dtype=torch.float32
        ).to(self.device)
        
        # pos_weight from train labels only
        pos = y_train.sum()
        neg = len(y_train) - pos
        pw  = (neg / (pos + 1e-9)).clamp(max=20.0)
        
        global_params = [p.data.clone() for p in self.model.parameters()]
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass with XGBoost scores
            logits = self.model(
                x_train,
                self.train_edge_index.to(self.device),
                xgb_scores=xgb_train_tensor
            )
            
            task_loss = F.binary_cross_entropy_with_logits(
                logits, y_train,
                pos_weight=torch.as_tensor(pw, device=self.device)
            )
            
            # FedProx proximal regularisation
            prox = sum(
                ((p - g.to(self.device)) ** 2).sum()
                for p, g in zip(self.model.parameters(), global_params)
            )
            loss = task_loss + (mu / 2.0) * prox
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on test set using XGBoost + GraphSAGE hybrid"""
        self.model.eval()
        
        x_all = torch.tensor(self.X).to(self.device)
        xgb_all_tensor = torch.tensor(
            self.xgb_all_scores, dtype=torch.float32
        ).to(self.device)
        
        logits = self.model(
            x_all,
            self.full_edge_index.to(self.device),
            xgb_scores=xgb_all_tensor
        )
        
        probs = torch.sigmoid(logits[self.test_idx]).cpu().numpy()
        y_true = self.y[self.test_idx]
        
        # Also evaluate XGBoost-only for comparison
        xgb_only_auc = roc_auc_score(y_true, self.xgb_test_scores)
        
        hybrid_auc = roc_auc_score(y_true, probs)
        hybrid_ap = average_precision_score(y_true, probs)
        
        return {
            "AUC": hybrid_auc,
            "AP": hybrid_ap,
            "XGB_AUC": xgb_only_auc,
            "report": classification_report(
                y_true, 
                (probs >= 0.5).astype(int),
                zero_division=0  # Suppress ill-defined precision warning
            ),
        }
    
    def add_new_data(self, new_df, update_labels=True,
                     recompute_density=True, max_train_size=50000):
        """
        Integrates streaming customers.
        Updates both XGBoost scores and GraphSAGE graph.
        """
        X_new_cleaned = self._clean_df(new_df)
        X_new_cleaned = X_new_cleaned.reindex(columns=self.feature_names, fill_value=0)
        X_new_raw     = X_new_cleaned.values.astype(np.float32)
        X_new_scaled  = self.scaler.transform(X_new_raw).astype(np.float32)
        
        n_new = X_new_scaled.shape[0]
        n_old = self.X.shape[0]
        
        if recompute_density:
            # Combine old + new (excluding old density column)
            X_combined = np.vstack([self.X[:, :-1], X_new_scaled])

            # Build graph with uniform k
            tmp_graph = build_knn_graph(X_combined, k_default=self.k)

            deg_all = degree(tmp_graph[0], num_nodes=n_old + n_new)

            # Normalize safely
            max_deg = deg_all.max().clamp(min=1)

            density_all = (deg_all / max_deg).cpu().numpy().reshape(-1, 1)

            density_old = density_all[:n_old]
            density_new = density_all[n_old:]

            # Update stored density
            self.X[:, -1:] = density_old

        else:
            # Simple fallback (no class-awareness anymore)
            # density_new = np.full((n_new, 1), 0.5, dtype=np.float32)
            density_new = np.full(
                    (n_new, 1),
                    self.X[:, -1].mean(),
                    dtype=np.float32
                )
        
        self.X = np.vstack([self.X, np.hstack([X_new_scaled, density_new])])
        
        if update_labels and "label" in new_df.columns:
            new_y          = new_df["label"].astype(int).values
            new_indices    = np.arange(n_old, n_old + n_new)
            self.y         = np.concatenate([self.y, new_y])
            self.train_idx = np.concatenate([self.train_idx, new_indices])
            
            if len(self.train_idx) > max_train_size:
                self.train_idx = self.train_idx[-max_train_size:]
                print(f"[Bank {self.bank_id}] Train set capped at {max_train_size}")
            
            train_mask = torch.zeros(self.X.shape[0], dtype=torch.bool)
            train_mask[self.train_idx] = True
            tmp_graph = build_knn_graph(self.X[:, :-1], y=None, k_default=self.k)
            self.train_edge_index, _ = subgraph(
                train_mask, tmp_graph,
                relabel_nodes=True, num_nodes=self.X.shape[0]
            )
        
        growth_ratio = n_new / max(n_old, 1)
        if growth_ratio > 0.05 or not hasattr(self, "_last_graph_size"):
            self.full_edge_index  = build_knn_graph(self.X, y=None, k_default=self.k)
            self._last_graph_size = self.X.shape[0]
        else:
            new_edges            = build_knn_graph(self.X[n_old:], y=None, k_default=self.k)
            new_edges            = new_edges + n_old
            combined             = torch.cat([self.full_edge_index, new_edges], dim=1)
            self.full_edge_index = torch.unique(combined, dim=1)
        
        # ── Update XGBoost scores for new data ────────────────────────────────
        self._update_xgb_scores()
        
        # ── Get hybrid predictions for new data ───────────────────────────────
        self.model.eval()
        with torch.no_grad():
            x_t   = torch.from_numpy(self.X).to(self.device)
            xgb_t = torch.from_numpy(self.xgb_all_scores).float().to(self.device)
            logits = self.model(x_t, self.full_edge_index.to(self.device), xgb_t)
            probs  = torch.sigmoid(logits[n_old:]).cpu().numpy()
        
        print(f"[Bank {self.bank_id}] +{n_new} customers | "
              f"Total: {self.X.shape[0]} | Train: {len(self.train_idx)} | "
              f"Avg default prob: {probs.mean():.4f}")
        return probs
