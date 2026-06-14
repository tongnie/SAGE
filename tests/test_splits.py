from sage.splits import filter_ids_by_summary, scenario_ids, scenario_ids_from_csv


def test_sage_split_filters_known_skip_ids():
    ids = scenario_ids("configs/splits/sage_womd_500.json", split="train", max_scenarios=10)
    assert ids[:5] == [0, 1, 2, 5, 6]
    assert 3 not in ids


def test_scenario_summary_csv_loader():
    ids = scenario_ids_from_csv("configs/splits/sage_autopilot_summary.csv")
    assert ids[:3] == [0, 1, 2]
    assert filter_ids_by_summary([0, 1, 400], "configs/splits/sage_autopilot_summary.csv") == [0, 1]
