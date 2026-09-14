"""Small dependency-light checks for the public AURA release."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aura_model import AURA

CHECKPOINT = ROOT / "checkpoints" / "aura.pt"


def test_network_forward_contract():
    model = AURA(n_motors=12, window=30, feat_dim=4,
                 d_model=128, n_heads=4, d_embed=32,
                 tau_max=[360., 128., 128., 360., 101., 130.] * 2)
    model.eval()
    with torch.no_grad():
        alpha, kappa = model(torch.zeros(2, 12, 30, 4))
    assert alpha.shape == (2, 12)
    assert kappa.shape == (2, 12)
    assert torch.all((alpha > 0) & (alpha < 1))
    assert torch.all(kappa > 0)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not packaged")
def test_packaged_checkpoint_loads():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert state["format_version"] == 2
    assert state["motor_ids"] and len(state["motor_ids"]) == 12
    assert state["features"] == ["cmd_effort", "fb_effort", "position", "velocity"]
    model = AURA(**state["model_config"], tau_max=state["tau_max"])
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        alpha, kappa = model(torch.zeros(1, 12, state["window"], 4))
    assert torch.isfinite(alpha).all() and torch.isfinite(kappa).all()


def test_torque_reconstruction_and_compensation():
    model = AURA(tau_max=[10.] * 12)
    alpha = torch.full((1, 12), 0.7)
    kappa = torch.full((1, 12), 0.8)
    command = torch.zeros(1, 12)
    feedback = model.reconstruct_torque(command, alpha, kappa)
    assert torch.allclose(feedback, torch.zeros_like(feedback))
