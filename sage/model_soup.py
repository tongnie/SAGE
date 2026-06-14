"""Weight-space interpolation utilities used by SAGE."""

from __future__ import annotations

from copy import deepcopy


def create_souped_model(model1, model2, w1: float, w2: float, device):
    """Return a deep-copied model whose parameters are w1 * model1 + w2 * model2."""
    souped_model = deepcopy(model1)
    souped_model.to(device)
    params1 = model1.state_dict()
    params2 = model2.state_dict()
    souped_params = souped_model.state_dict()
    for name in souped_params.keys():
        if name in params1 and name in params2:
            souped_params[name].data.copy_(w1 * params1[name].data + w2 * params2[name].data)
    souped_model.load_state_dict(souped_params)
    return souped_model
