"""
server.py
---------
CentralServer: aggregates client weights using FedAdam (default)
or plain weighted FedAvg.

FedAdam (Reddi et al., 2021):
  Instead of directly replacing the global model with the weighted
  average of client weights, the server treats the difference between
  the weighted average and the current global model as a pseudo-gradient,
  then applies server-side Adam to update the global model adaptively.

  delta_t  = weighted_avg(w_clients) - w_global   ← pseudo-gradient
  m_t      = beta1 * m_t  +  (1 - beta1) * delta_t
  v_t      = beta2 * v_t  +  (1 - beta2) * delta_t²
  w_global = w_global  +  server_lr * m_t / (sqrt(v_t) + tau)

Why FedAdam over plain FedAvg?
  - Adapts the step size per parameter — more stable convergence
  - Handles non-IID data better than FedAvg when combined with FedProx
  - Proven to converge faster in heterogeneous settings

FIX #3 — server_lr / tau tuning
  Original: server_lr=1e-2, tau=1e-3  → ratio = 10
  Fixed:    server_lr=1e-3, tau=1e-3  → ratio = 1

  With only 3 heterogeneous banks, v_hat starts near zero in early rounds.
  A high server_lr/tau ratio makes the effective step size very large then
  (server_lr / tau ≈ 10), which causes oscillation and unstable convergence
  before the second moment has had time to warm up.

  Lowering server_lr to 1e-3 (matching tau) brings the early-round step
  size in line with the original FedAdam paper's recommended configuration
  for heterogeneous settings, without sacrificing asymptotic performance.
"""

import copy
import torch


class CentralServer:
    """
    Parameters
    ----------
    initial_state_dict : OrderedDict – starting weights (e.g. from one client)
    use_fedadam        : bool  – True = FedAdam, False = plain weighted FedAvg
    server_lr          : float – Adam step size on the server (η_s)
                                 FIX: changed default from 1e-2 → 1e-3
    beta1              : float – first moment decay  (default 0.9)
    beta2              : float – second moment decay (default 0.99)
    tau                : float – numerical stability term (default 1e-3)
                                 Ratio server_lr/tau should be ≈ 1 for stable
                                 early-round behaviour with few clients.
    """

    def __init__(self, initial_state_dict,
                 use_fedadam=True,
                 server_lr=1e-3,   # FIX #3: was 1e-2; reduced to match tau
                 beta1=0.9,
                 beta2=0.99,
                 tau=1e-3):

        self.global_state = copy.deepcopy(initial_state_dict)
        self.use_fedadam  = use_fedadam
        self.server_lr    = server_lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.tau          = tau
        self._t           = 0  # round counter for bias correction

        # Initialise Adam moment buffers (zeros, same shape as model)
        self._m = {k: torch.zeros_like(v.float())
                   for k, v in self.global_state.items()}
        self._v = {k: torch.zeros_like(v.float())
                   for k, v in self.global_state.items()}

    # ── API ───────────────────────────────────────────────────────────────────

    def get_global_weights(self):
        """Broadcast current global weights to all clients."""
        return copy.deepcopy(self.global_state)

    def aggregate(self, client_weights_list, client_sample_counts):
        """
        Aggregate client weights and update the global model.

        Parameters
        ----------
        client_weights_list  : list[OrderedDict]
        client_sample_counts : list[int]
        """
        total = sum(client_sample_counts)

        # ── Weighted average of client weights ────────────────────────────────
        avg_state = copy.deepcopy(client_weights_list[0])
        for key in avg_state:
            avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)

        for state, n in zip(client_weights_list, client_sample_counts):
            w = n / total
            for key in avg_state:
                avg_state[key] += w * state[key].float()

        if not self.use_fedadam:
            # Plain FedAvg — just replace global with weighted average
            self.global_state = avg_state
            return copy.deepcopy(self.global_state)

        # ── FedAdam update ────────────────────────────────────────────────────
        self._t += 1

        new_state = copy.deepcopy(self.global_state)
        for key in new_state:
            # Pseudo-gradient: direction from global toward weighted average
            delta = avg_state[key].float() - self.global_state[key].float()

            # First moment
            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * delta

            # Second moment
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * delta ** 2

            # Bias-corrected moments
            m_hat = self._m[key] / (1 - self.beta1 ** self._t)
            v_hat = self._v[key] / (1 - self.beta2 ** self._t)

            # Adam update
            new_state[key] = (self.global_state[key].float()
                              + self.server_lr * m_hat / (torch.sqrt(v_hat) + self.tau))

        self.global_state = new_state
        return copy.deepcopy(self.global_state)