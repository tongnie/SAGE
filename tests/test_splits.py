from sage.splits import scenario_ids


def test_sage_split_filters_known_skip_ids():
    ids = scenario_ids("configs/splits/sage_womd_500.json", split="train", max_scenarios=10)
    assert ids[:5] == [0, 1, 2, 5, 6]
    assert 3 not in ids
