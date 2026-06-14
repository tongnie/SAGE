# Data Preparation

SAGE expects processed MetaDrive scenarios derived from WOMD/CAT data. The raw and processed scenarios are not redistributed in this repository.

Expected local directories:

```text
raw_scenes_500/
raw_scenes_370/
```

The public split metadata is stored in `configs/splits/sage_womd_500.json`. The split file contains the train/eval ranges and skip IDs used by the release scripts. The companion summary file `configs/splits/sage_autopilot_summary.csv` records the scenario-level replay/autopilot scores used by the HGPO scripts.

Useful entry points:

```bash
python scripts/prepare_waymo.py --help
python scripts/select_cases.py --help
```

If a script reports that the data directory is missing, create the expected directory by following the CAT/WOMD preprocessing workflow, then rerun the command with `--data_directory`.
