from src.baseline_plans import BASELINE_SHA256, baseline_plan_hash, load_baseline_plans


def test_historical_baseline_is_byte_validated():
    assert baseline_plan_hash(".") == BASELINE_SHA256
    plans = load_baseline_plans(".")
    assert len(plans) == 100
    assert [plan.id for plan in plans] == list(range(100))
