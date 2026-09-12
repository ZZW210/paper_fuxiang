from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.config import load_config, apply_quick_overrides
from src.run_archive import RunArchive, generate_run_id, validate_run_id


def archive(tmp_path, run_id=None, overwrite=False):
    cfg=apply_quick_overrides(load_config("nonexistent.yaml"))
    args=SimpleNamespace(run_id=run_id,overwrite_run=overwrite,seed=2025,quick=True,config="config.yaml")
    return RunArchive(Path(__file__).resolve().parents[1],tmp_path,cfg,args)


def finish(run):
    with run:
        pd.DataFrame([dict(initial_conflicts_uncertain=130,final_conflicts_one_stage=98,final_conflicts_two_stage=40,
                           seed=2025,scheduler_mode="paper_strict")]).to_csv(run.directory/"metrics_summary.csv",index=False)
        pd.DataFrame(dict(global_generation=[1],fitness=[1])).to_csv(run.directory/"paper_convergence.csv",index=False)
        pd.DataFrame(dict(x=[1])).to_csv(run.directory/"logs"/"nested.csv",index=False)
        figure,axis=plt.subplots()
        axis.plot([0,1],[0,1])
        figure.savefig(run.directory/"test.png")
        plt.close(figure)
        print("Stage1 generation 1/50")


def test_run_id_auto_generation():
    value=generate_run_id("paper_strict","raw_equation",2025,"a83fd21")
    validate_run_id(value)
    assert value and "seed2025" in value and "paper_strict" in value and "raw_equation" in value
    assert not any(c in value for c in ':\\/')
    assert len({generate_run_id("paper_strict","raw_equation",2025) for _ in range(100)})==100


@pytest.mark.parametrize("value",["../outside","a/b","a\\b","a:b","CON","run.",""])
def test_invalid_run_id_rejected(value):
    with pytest.raises(ValueError):
        validate_run_id(value)


def test_run_directory_isolated(tmp_path):
    a,b=archive(tmp_path,"run1"),archive(tmp_path,"run2")
    finish(a)
    original=(a.directory/"metrics_summary.csv").read_bytes()
    finish(b)
    assert original==(a.directory/"metrics_summary.csv").read_bytes()
    assert (tmp_path/"latest_run.txt").read_text()=="run2"
    assert pd.read_csv(tmp_path/"run_index.csv").run_id.tolist()==["run1","run2"]


def test_csv_contains_run_id(tmp_path):
    run=archive(tmp_path,"csv_test")
    finish(run)
    for filename in run.directory.rglob("*.csv"):
        frame=pd.read_csv(filename)
        assert frame.columns[0]=="run_id" and frame.run_id.eq(run.run_id).all()
    assert (run.directory/"figures"/"test.png").exists()
    assert not (run.directory/"test.png").exists()
    assert "[run_id=csv_test] Stage1 generation 1/50" in (run.directory/"stdout.log").read_text(encoding="utf-8")


def test_manifest_matches_run(tmp_path):
    run=archive(tmp_path,"manifest_test")
    finish(run)
    manifest=json.loads((run.directory/"run_manifest.json").read_text(encoding="utf-8"))
    summary=pd.read_csv(run.directory/"metrics_summary.csv").iloc[0]
    assert manifest["run_id"]==summary.run_id and manifest["status"]=="completed"
    assert manifest["initial_conflicts"]==130 and manifest["final_conflicts"]==40
    assert manifest["NP_stage1"]==20 and manifest["Ngen_stage1"]==50
    assert abs(manifest["elapsed_seconds"]-summary.runtime_total)<1e-10
    cfg=load_config(run.directory/"config_snapshot.yaml")
    assert cfg["run"]["run_id"]==run.run_id and cfg["fata"]["NP"]==20


def test_existing_run_id_refuses_overwrite(tmp_path):
    a=archive(tmp_path,"existing")
    (a.directory/"marker.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        archive(tmp_path,"existing")
    assert (a.directory/"marker.txt").read_text()=="preserve"
    b=archive(tmp_path,"existing",overwrite=True)
    assert not (b.directory/"marker.txt").exists()


def test_failed_run_not_added_to_latest_or_index(tmp_path):
    run=archive(tmp_path,"failed")
    with pytest.raises(RuntimeError):
        with run:
            raise RuntimeError("expected failure")
    manifest=json.loads((run.directory/"run_manifest.json").read_text())
    assert manifest["status"]=="failed" and not (tmp_path/"latest_run.txt").exists()
    assert not (tmp_path/"run_index.csv").exists()


def test_png_is_saved_with_run_label(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    original=Figure.savefig
    labels=[]
    def inspect_save(figure,*args,**kwargs):
        labels.extend(t.get_text() for t in figure.texts)
        return original(figure,*args,**kwargs)
    monkeypatch.setattr(Figure,"savefig",inspect_save)
    finish(archive(tmp_path,"png_test"))
    assert "Run ID: png_test" in labels
