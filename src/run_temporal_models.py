"""Temporal Deep Learning Models for CS2 Early Round Outcome Prediction.

Implements:
1. LSTMClassifier: 2-layer causal LSTM with dropout and classification head
2. GRUClassifier: 2-layer causal GRU with dropout and classification head
3. TemporalConvNet: Dilated causal 1D-CNN with residual blocks

Strict Causal Guard:
Sequence constructed strictly from snapshots tau <= t. Future ticks are never included.
Threshold tuned on validation set only; evaluated on held-out test rounds.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from cs2pred.modeling_extended import tune_threshold, evaluate_metrics

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        last_hidden = out[:, -1, :] # representation at final available cutoff step
        logits = self.fc(last_hidden).squeeze(-1)
        return logits

class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, hn = self.gru(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim: int, num_channels: list[int] = [32, 64], kernel_size: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_c = input_dim
        for i, out_c in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(CausalConv1d(in_c, out_c, kernel_size=kernel_size, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_c = out_c
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x is (batch, seq_len, feat_dim) -> transpose to (batch, feat_dim, seq_len)
        x_trans = x.transpose(1, 2)
        out = self.network(x_trans)
        last_step = out[:, :, -1]
        logits = self.fc(last_step).squeeze(-1)
        return logits

def train_temporal_model(model_name: str, X_seq_tr, y_tr, X_seq_va, y_va, X_seq_te, y_te,
                         batch_size: int = 256, epochs: int = 25, lr: float = 0.002, seed: int = 42):
    """Train sequence model with early stopping on validation BCE loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_seq_tr.shape[2]

    if model_name == "lstm":
        net = LSTMClassifier(input_dim=input_dim).to(device)
    elif model_name == "gru":
        net = GRUClassifier(input_dim=input_dim).to(device)
    elif model_name == "tcn":
        net = TemporalConvNet(input_dim=input_dim).to(device)
    else:
        raise ValueError(f"Unknown temporal model: {model_name}")

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_seq_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
                           batch_size=batch_size, shuffle=True)
    va_x = torch.tensor(X_seq_va, dtype=torch.float32).to(device)
    te_x = torch.tensor(X_seq_te, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_weights = None
    patience = 5
    patience_cnt = 0

    t0 = time.time()
    for ep in range(epochs):
        net.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = net(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            v_logits = net(va_x)
            v_loss = criterion(v_logits, torch.tensor(y_va, dtype=torch.float32).to(device)).item()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    train_time = time.time() - t0
    if best_weights:
        net.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    net.eval()
    with torch.no_grad():
        pva = torch.sigmoid(net(va_x)).cpu().numpy()
        pte = torch.sigmoid(net(te_x)).cpu().numpy()

    thr, val_mf1 = tune_threshold(y_va, pva)
    metrics = evaluate_metrics(y_te, pte, thr=thr)
    metrics["val_threshold"] = thr
    metrics["val_macro_f1"] = val_mf1
    metrics["train_time_s"] = train_time
    return metrics, pte
