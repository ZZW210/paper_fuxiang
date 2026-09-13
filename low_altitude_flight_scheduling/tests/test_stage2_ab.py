from types import SimpleNamespace

from run_stage2_ab import origins


def conflict(a, b, cell=(1, 2, 3), indices=(4, 5)):
    return SimpleNamespace(plan_a=a, plan_b=b, cell=cell, idx_a=indices[0], idx_b=indices[1])


def test_origin_partition_and_reversed_pairs():
    before = [conflict(1, 2), conflict(2, 3)]
    after = [conflict(2, 1, indices=(5, 4)), conflict(2, 3, cell=(9, 2, 3)), conflict(7, 8)]
    assert origins(before, after) == dict(exact_survivors=1, moved_same_pair=1,
        new_conflict_pairs=1, new_conflict_points=1, resolved_original_conflicts=1)


def test_empty_final_resolves_all_original_events():
    stats = origins([conflict(1, 2)], [])
    assert stats['resolved_original_conflicts'] == 1
    assert stats['exact_survivors'] == stats['new_conflict_pairs'] == 0
