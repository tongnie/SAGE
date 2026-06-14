# Reproduction Commands

Train the two experts:

```bash
python scripts/train_hgpo_adv.py --data_directory ./raw_scenes_500
python scripts/train_hgpo_real.py --data_directory ./raw_scenes_500
```

Evaluate test-time steerability:

```bash
python scripts/eval_sage.py --data_directory ./raw_scenes_500 --w_adv 0.0
python scripts/eval_sage.py --data_directory ./raw_scenes_500 --w_adv 0.5
python scripts/eval_sage.py --data_directory ./raw_scenes_500 --w_adv 1.0
```

Run downstream closed-loop RL training:

```bash
python scripts/train_rl_sage.py --data_directory ./raw_scenes_370
```

For smoke tests, add small caps such as `--max_scenarios 1`, `--epochs 1`, or `--max_timesteps 100`.
