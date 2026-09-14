"""AURA: causal temporal identification and self-supervised torque reconstruction.

Train with: python data/train_aura.py --csv-dir data/csv_aligned
Labels are optional simulation metadata, used only for evaluation. Checkpoints
use format version 2 and must be retrained after the temporal-encoder changes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .aura_data import (FEATURES, MOTOR_IDS, MOTOR_TYPES, TAU_MAX_VALUES, WINDOW,
                            load_recordings, split_recordings)
else:
    from aura_data import (FEATURES, MOTOR_IDS, MOTOR_TYPES, TAU_MAX_VALUES, WINDOW,
                           load_recordings, split_recordings)

N_MOTORS, FEAT_DIM = len(MOTOR_IDS), len(FEATURES)
TAU_MAX = torch.tensor(TAU_MAX_VALUES, dtype=torch.float32)
DATA_DIR = Path(__file__).resolve().parent


class TemporalConvEncoder(nn.Module):
    """Dilated causal convolutions with receptive field covering the full window."""

    def __init__(self, in_channels=4, hidden=64, out_channels=128, kernel_size=3, window=30):
        super().__init__()
        if window < 1 or kernel_size < 2:
            raise ValueError('window >= 1 and kernel_size >= 2 are required')
        n_layers = max(1, math.ceil(math.log2((window - 1) / (kernel_size - 1) + 1)))
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else hidden,
                      out_channels if i == n_layers - 1 else hidden,
                      kernel_size, dilation=2 ** i)
            for i in range(n_layers)
        ])
        self.receptive_field = 1 + (kernel_size - 1) * (2 ** n_layers - 1)

    def forward(self, x):
        for layer in self.layers:
            padding = (layer.kernel_size[0] - 1) * layer.dilation[0]
            x = F.elu(layer(F.pad(x, (padding, 0))))
        return x[:, :, -1]


class CrossMotorAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        attended, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class AURA(nn.Module):
    """[B, motors, history, 4] -> (alpha, kappa), each [B, motors]."""

    def __init__(self, n_motors=12, window=30, feat_dim=4, d_model=128, n_heads=4,
                 d_embed=32, tau_max=None):
        super().__init__()
        self.config = dict(n_motors=n_motors, window=window, feat_dim=feat_dim,
                           d_model=d_model, n_heads=n_heads, d_embed=d_embed)
        self.n_motors, self.window, self.feat_dim = n_motors, window, feat_dim
        self.d_model = d_model
        limits = TAU_MAX.clone() if tau_max is None else torch.as_tensor(tau_max, dtype=torch.float32)
        if limits.shape != (n_motors,) or not torch.isfinite(limits).all() or (limits <= 0).any():
            raise ValueError('Provide one finite positive torque limit per motor')
        self.register_buffer('tau_max', limits)
        self.temporal_enc = TemporalConvEncoder(feat_dim, out_channels=d_model, window=window)
        self.cross_motor = CrossMotorAttention(d_model, n_heads)
        # Identity must enter before mean pooling; otherwise swapping L/R
        # histories produces the same latent, regardless of output embeddings.
        self.token_embed = nn.Embedding(n_motors, d_model)
        self.motor_embed = nn.Embedding(n_motors, d_embed)
        self.predictor = nn.Sequential(nn.Linear(d_model + d_embed, 128), nn.ELU(),
                                       nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 2))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.n_motors, self.window, self.feat_dim):
            raise ValueError(f'Expected [B,{self.n_motors},{self.window},{self.feat_dim}]')
        batch = x.shape[0]
        temporal = x.permute(0, 1, 3, 2).reshape(batch * self.n_motors, self.feat_dim, self.window)
        h = self.temporal_enc(temporal).reshape(batch, self.n_motors, self.d_model)
        idx = torch.arange(self.n_motors, device=x.device)
        h = self.cross_motor(h + self.token_embed(idx).unsqueeze(0))
        latent = h.mean(dim=1).unsqueeze(1).expand(-1, self.n_motors, -1)
        embed = self.motor_embed(idx).unsqueeze(0).expand(batch, -1, -1)
        out = self.predictor(torch.cat((latent, embed), dim=-1))
        return torch.sigmoid(out[..., 0]), F.softplus(out[..., 1]) + 1e-6

    def reconstruct_torque(self, tau_cmd, alpha, kappa):
        return alpha * self.tau_max * torch.tanh(tau_cmd / (self.tau_max * kappa.clamp_min(1e-6)))

    def compensate(self, tau_des, alpha, kappa, eps_clip=0.02):
        if not 0 < eps_clip < 1:
            raise ValueError('eps_clip must be in (0,1)')
        if not all(torch.isfinite(v).all() for v in (tau_des, alpha, kappa)):
            raise ValueError('Compensation inputs must be finite')
        if (alpha <= 0).any() or (alpha > 1).any() or (kappa <= 0).any():
            raise ValueError('Require 0 < alpha <= 1 and kappa > 0')
        ratio = (tau_des / (alpha * self.tau_max)).clamp(-1 + eps_clip, 1 - eps_clip)
        command = self.tau_max * kappa * torch.atanh(ratio)
        return command.clamp(-self.tau_max, self.tau_max)


def normalization(records):
    """Fit statistics exclusively to training windows, accumulating in float64."""
    count = 0
    total = np.zeros((12, 4), dtype=np.float64)
    squared = np.zeros_like(total)
    for record in records:
        values = record['windows'].astype(np.float64)
        total += values.sum(axis=(0, 2))
        squared += (values ** 2).sum(axis=(0, 2))
        count += values.shape[0] * values.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - mean ** 2, 0)).clip(1e-6)
    return (torch.tensor(mean[None, :, None, :], dtype=torch.float32),
            torch.tensor(std[None, :, None, :], dtype=torch.float32))


def make_loader(records, mean, std, batch_size, shuffle=False):
    raw = torch.from_numpy(np.concatenate([r['windows'] for r in records]))
    labels = torch.from_numpy(np.concatenate([np.broadcast_to(r['labels'], (len(r['windows']), 12, 2))
                                              for r in records]))
    dataset = TensorDataset((raw - mean) / std, raw[:, :, -1, 0], raw[:, :, -1, 1], labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    squared = torch.zeros(12, dtype=torch.float64, device=device)
    fb_sum, fb_squared = squared.clone(), squared.clone()
    param_abs = torch.zeros((12, 2), dtype=torch.float64, device=device)
    param_count = torch.zeros_like(param_abs)
    count = 0
    for x, command, feedback, labels in loader:
        x, command, feedback, labels = [v.to(device) for v in (x, command, feedback, labels)]
        alpha, kappa = model(x)
        estimate = model.reconstruct_torque(command, alpha, kappa)
        squared += ((estimate - feedback).double() ** 2).sum(0)
        fb_sum += feedback.double().sum(0)
        fb_squared += (feedback.double() ** 2).sum(0)
        valid = torch.isfinite(labels)
        errors = (torch.stack((alpha, kappa), dim=-1) - labels).abs()
        param_abs += torch.where(valid, errors, 0).sum(0)
        param_count += valid.sum(0)
        count += len(x)
    if not count:
        raise ValueError('Empty evaluation set')
    sst = (fb_squared - fb_sum ** 2 / count).clamp_min(0)
    metrics = {'torque_rmse_nm': float((squared.sum() / (count * 12)).sqrt()), 'windows': count,
               'per_motor': {}}
    for i, mid in enumerate(MOTOR_IDS):
        metrics['per_motor'][mid] = {
            'torque_rmse_nm': float((squared[i] / count).sqrt()),
            'r2': float(1 - squared[i] / sst[i]) if sst[i] > 1e-12 else None,
            'alpha_mae': float(param_abs[i, 0] / param_count[i, 0]) if param_count[i, 0] else None,
            'kappa_mae': float(param_abs[i, 1] / param_count[i, 1]) if param_count[i, 1] else None,
        }
    return metrics


def train_aura(model, train_loader, val_loader, *, device='cpu', epochs=500, lr=1e-3, patience=120):
    """Pure reconstruction loss, minibatches, validation-selected checkpoint."""
    if epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)
    best_loss, best_state, stale = float('inf'), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for x, command, feedback, _ in train_loader:
            x, command, feedback = [v.to(device) for v in (x, command, feedback)]
            alpha, kappa = model(x)
            loss = F.mse_loss(model.reconstruct_torque(command, alpha, kappa), feedback)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        if not count:
            raise ValueError('Empty training set')
        val_mse = evaluate(model, val_loader, device)['torque_rmse_nm'] ** 2
        if not math.isfinite(val_mse):
            raise ValueError('Non-finite validation loss')
        scheduler.step(val_mse)
        history.append({'epoch': epoch, 'train_mse': total / count, 'val_mse': val_mse})
        if val_mse < best_loss:
            best_loss, stale = val_mse, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        print(f'Epoch {epoch}: train MSE={total / count:.6f}, val MSE={val_mse:.6f}')
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', nargs='?', type=Path, default=DATA_DIR)
    parser.add_argument('--csv-dir', type=Path, help='Default: <data_dir>/csv_aligned')
    parser.add_argument('--out-dir', type=Path, help='Default: <data_dir>/models/aura_retrained')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--window', type=int, default=WINDOW)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--max-gap-ms', type=float, default=50.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tau-max', type=float, nargs=12, default=TAU_MAX_VALUES)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    csv_dir = args.csv_dir or args.data_dir / 'csv_aligned'
    records = load_recordings(csv_dir, args.window, args.stride, args.max_gap_ms)
    splits = split_recordings(records, args.seed)
    mean, std = normalization(splits['train'])
    loaders = {name: make_loader(values, mean, std, args.batch_size, name == 'train')
               for name, values in splits.items()}
    model = AURA(window=args.window, tau_max=args.tau_max)
    history = train_aura(model, loaders['train'], loaders['val'], device=args.device, epochs=args.epochs)
    metrics = evaluate(model, loaders['test'], args.device)
    out_dir = args.out_dir or args.data_dir / 'models' / 'aura_retrained'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {name: [{'path': r['path'], 'session_id': r['session_id'], 'windows': len(r['windows'])}
                       for r in values] for name, values in splits.items()}
    model.cpu()
    torch.save({'format_version': 2, 'model': model.state_dict(), 'model_config': model.config,
                'x_mean': mean, 'x_std': std, 'motor_ids': MOTOR_IDS, 'features': FEATURES,
                'window': args.window, 'tau_max': model.tau_max, 'seed': args.seed,
                'splits': manifest, 'history': history}, out_dir / 'aura.pt')
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (out_dir / 'splits.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f"Held-out recording torque RMSE: {metrics['torque_rmse_nm']:.4f} Nm")
    print(f'Checkpoint: {out_dir / "aura.pt"}')


if __name__ == '__main__':
    main()

