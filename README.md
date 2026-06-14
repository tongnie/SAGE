# SAGE

Official code release for **Steerable Adversarial Scenario Generation through Test-Time Preference Alignment** (ICLR 2026).

SAGE treats adversarial scenario generation as multi-objective preference alignment. It trains two HGPO/DPO experts with opposing preferences, then steers test-time generation by interpolating their weights.

- Paper: https://openreview.net/forum?id=lYNsZdKn5R
- Project page: https://tongnie.github.io/SAGE/
- Checkpoints: see [docs/checkpoints.md](docs/checkpoints.md)

## What Is Included

- HGPO/DPO fine-tuning scripts for the adversarial and realism experts.
- Shared reward, feasibility, realism, and distributional metrics.
- SAGE weight-interpolation generation and evaluation script.
- Closed-loop downstream RL training script.
- Scenario split metadata and WOMD/CAT preprocessing utilities.

Raw WOMD/CAT scenarios, training logs, TensorBoard runs, debug checkpoints, and full ablation artifacts are intentionally not included.

## Installation

```bash
conda create -n sage python=3.9 -y
conda activate sage
pip install -e .
```

The editable install builds `advgen.utils_cython` from source. If the extension build fails, install a local C/C++ build toolchain and rerun `pip install -e .`.

## Data And Checkpoints

Prepare processed MetaDrive/WOMD scenarios by following the CAT data preparation flow, or use:

```bash
python scripts/prepare_waymo.py --help
python scripts/select_cases.py --help
```

Expected local layout:

```text
raw_scenes_500/
raw_scenes_370/
advgen/pretrained/densetnt.bin
advgen/finetuned/grpo_finetuned_model_adv_best.bin
advgen/finetuned/grpo_finetuned_model_real_best.bin
```

Checkpoint filenames, SHA256 values, and Google Drive placeholders are listed in [docs/checkpoints.md](docs/checkpoints.md).

## Train HGPO Experts

Adversarial expert:

```bash
python scripts/train_hgpo_adv.py \
  --data_directory ./raw_scenes_500 \
  --base_model_path ./advgen/pretrained/densetnt.bin \
  --split_file ./configs/splits/sage_womd_500.json
```

Realism expert:

```bash
python scripts/train_hgpo_real.py \
  --data_directory ./raw_scenes_500 \
  --base_model_path ./advgen/pretrained/densetnt.bin \
  --split_file ./configs/splits/sage_womd_500.json
```

For a quick smoke test, add `--epochs 1 --max_scenarios 1 --log_dir ./tmp/runs`.

## Evaluate SAGE Generation

```bash
python scripts/eval_sage.py \
  --data_directory ./raw_scenes_500 \
  --adv_model_path ./advgen/finetuned/grpo_finetuned_model_adv_best.bin \
  --real_model_path ./advgen/finetuned/grpo_finetuned_model_real_best.bin \
  --w_adv 0.5 \
  --max_scenarios 10
```

The vendored MetaDrive copy is trimmed for headless training and evaluation. Use `--use_render` only after restoring the corresponding MetaDrive rendering assets/plugins.

## Downstream RL Training

```bash
python scripts/train_rl_sage.py \
  --data_directory ./raw_scenes_370 \
  --adversarial_model_path ./advgen/finetuned/grpo_finetuned_model_adv_best.bin \
  --realism_model_path ./advgen/finetuned/grpo_finetuned_model_real_best.bin
```

For a smoke test, use `--max_timesteps 100 --eval_freq 50 --start_timesteps 10`.

## Citation

```bibtex
@inproceedings{nie2026sage,
  title={Steerable Adversarial Scenario Generation through Test-Time Preference Alignment},
  author={Nie, Tong and Mei, Yuewen and Tang, Yihong and He, Junlin and Sun, Jie and Shi, Haotian and Ma, Wei and Sun, Jian},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=lYNsZdKn5R}
}
```

## License And Acknowledgements

This repository is released under the MIT License. It builds on the CAT codebase, DenseTNT-style motion forecasting components, a modified MetaDrive environment, and the Waymo Open Motion Dataset. See [NOTICE.md](NOTICE.md) for attribution notes.
