import torch

from sage.model_soup import create_souped_model


def test_create_souped_model_interpolates_weights():
    model1 = torch.nn.Linear(2, 1, bias=False)
    model2 = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model1.weight.fill_(1.0)
        model2.weight.fill_(3.0)

    souped = create_souped_model(model1, model2, 0.25, 0.75, torch.device("cpu"))
    assert torch.allclose(souped.weight, torch.full_like(souped.weight, 2.5))
