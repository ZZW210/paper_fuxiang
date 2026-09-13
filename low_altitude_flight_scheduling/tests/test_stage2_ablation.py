from copy import deepcopy
from types import SimpleNamespace

from run_stage2_ablation import fixed_delay, fixed_speed, local_improved, raw_units
from src.config import load_config
from src.conflict_detection import Conflict, ConflictSegment


def conflict(a=1,b=2,gap=-5):
    return Conflict(a,b,(1,2,3),4,5,10,15,gap,30,'uncertain')


def test_raw_units_never_merge_continuous_points():
    conflicts = [conflict(),conflict()]
    units = raw_units(conflicts)
    assert len(units)==2
    assert units[0].conflicts==[conflicts[0]]
    assert units[0].cells==[conflicts[0].cell]


def test_fixed_candidates_do_not_read_time_information():
    cfg = load_config()
    plans = [SimpleNamespace(id=i,etd=900,delay=0,speed_profile=[10]) for i in (1,2)]
    # No time attributes or required_shift exist on this unit.
    unit = SimpleNamespace(plan_a=1,plan_b=2)
    assert fixed_delay(plans,unit,cfg,20)
    assert fixed_speed(plans,unit,cfg,20)
    for fn in (fixed_delay,fixed_speed):
        forward = ConflictSegment(1,2,[conflict(gap=-500)])
        reverse = ConflictSegment(1,2,[conflict(gap=500)])
        assert fn(plans,forward,cfg,20)==fn(plans,reverse,cfg,20)


def test_local_acceptance_ignores_global_worsening():
    old = [conflict()]
    new = [conflict(3,4),conflict(5,6)]
    assert local_improved(old,new,(1,2))
    assert not local_improved(old,[conflict(),conflict(3,4)],(1,2))
