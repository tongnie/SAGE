"""Train the realism HGPO expert used by SAGE."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sage.hgpo_training import main


if __name__ == "__main__":
    main(
        default_adversarial_weight=1.0,
        default_realism_weight=10.0,
        default_save_path="./advgen/finetuned/hgpo_finetuned_model_real.bin",
        default_run_name="hgpo_real",
    )
