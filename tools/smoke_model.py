"""Exercise the selected model on synthetic tensors, with no training data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))

import torch
import main as entrypoint
import infer_test_monai
from train_test_monai_semi import CONFIG, LABELED_DEVICE
from daoct.models.build import make_model, count_params


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    assert LABELED_DEVICE == "Topcon_Maestro2"
    assert CONFIG["pretrained_trunk"] is None
    model = make_model(dict(CONFIG)).cpu().train()
    out = model(torch.randn(1, 1, 64, 64))
    expected = {"seg": (1, 10, 64, 64), "sdm": (1, 1, 64, 64),
                "surf_logits": (1, 9, 64, 64), "surf_rows": (1, 9, 64)}
    for key, shape in expected.items():
        assert tuple(out[key].shape) == shape, (key, out[key].shape)
        assert torch.isfinite(out[key]).all(), key
    sum(value.square().mean() for value in out.values()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    print(f"PASS: entry-point imports, all output shapes and finite CPU gradients; {count_params(model):,} parameters.")
    print("This check does not validate full training, prediction quality or the server environment.")


if __name__ == "__main__":
    main()
