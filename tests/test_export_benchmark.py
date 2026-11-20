"""eval/export_benchmark.py and the benchmark.json it writes for the dashboard. The full export needs data that isn't
in git (answers, outcomes, the embedding model); these test its pieces and the committed file."""
import json

import pytest

from app.registry import PROJECT_ROOT
from eval import export_benchmark as eb
from eval.routing_sim import EvalRow

PHASE_3 = (PROJECT_ROOT / "reports" / "PHASE_3.md").read_text(encoding="utf-8")


def rows():
    """Three prompts: one the cheap tier passes, one only medium, one only high."""
    def make(i, label, passed):
        return EvalRow(id=f"p{i}", task_type="qa", source="own", label=label, passed=passed,
                       cost={"low": 1.0, "medium": 2.0, "high": 4.0}, latency_ms={t: 100.0 for t in eb.TIERS})
    return [make(0, "low", {"low": True, "medium": True, "high": True}),
            make(1, "medium", {"low": False, "medium": True, "high": True}),
            make(2, "high", {"low": False, "medium": False, "high": True})]


# ---------- the classifier table ----------

def test_the_phase_3_classifier_table_is_read_with_c_chosen():
    table = eb.parse_classifier_table(PHASE_3)
    assert len(table) == 6 and [row["name"] for row in table if row["chosen"]] == ["C: both, tuned logistic regression"]
    chosen = next(row for row in table if row["chosen"])
    assert (chosen["auc_cheap"], chosen["auc_medium"], chosen["brier_cheap"], chosen["brier_medium"]) == \
        (0.672, 0.667, 0.224, 0.111)
    assert table[0] == {"name": "Task type only (baseline)", "auc_cheap": 0.581, "auc_medium": 0.613,
                        "brier_cheap": 0.233, "brier_medium": 0.112, "chosen": False}


@pytest.mark.parametrize("markdown", [
    "# nothing here\n",
    "## Classifier comparison\n\n| Model | AUC | b | c | d |\n|---|---|---|---|---|\n| x | 1 | 2 | 3 | 4 |\n",
    "## Classifier comparison\n\n| Classifier | Cheap ROC AUC | Medium ROC AUC | Cheap Brier | Medium Brier |\n"
    "|---|---|---|---|---|\n| x | 1 | 2 | 3 | 4 |\n",                                           # no chosen row
    "## Classifier comparison\n\n| Classifier | Cheap ROC AUC | Medium ROC AUC | Cheap Brier | Medium Brier |\n"
    "|---|---|---|---|---|\n| **x** | 1 | n/a | 3 | 4 |\n",                                     # not a number
])
def test_a_changed_table_stops_the_export(markdown):
    with pytest.raises(eb.ExportError):
        eb.parse_classifier_table(markdown)


# ---------- curve, references, reproduction ----------

def test_the_curve_runs_from_all_cheap_to_all_strong():
    data = rows()
    probs = {"p0": (0.9, 0.95), "p1": (0.2, 0.9), "p2": (0.1, 0.3)}
    points = eb.curve(data, probs)
    assert [p["tau"] for p in points] == eb.TAUS and len(points) == 101
    assert points[0]["mix"] == {"low": 1.0, "medium": 0.0, "high": 0.0} and points[0]["correct"] == 1
    assert points[-1]["mix"] == {"low": 0.0, "medium": 0.0, "high": 1.0} and points[-1]["cost_saved"] == 0.0
    at_half = next(p for p in points if p["tau"] == 0.5)                 # p0 cheap, p1 medium, p2 high: perfect
    assert at_half["correct"] == 3 and at_half["cost_per_1k_usd"] == pytest.approx(1000 * 7 / 3, abs=1e-3)


def test_references():
    refs = eb.references(rows())
    assert refs["always_cheapest"]["correct"] == 1 and refs["always_strongest"]["quality_kept"] == 1.0
    assert refs["oracle"]["correct"] == 3 and refs["oracle"]["cost_saved"] == pytest.approx(1 - 7 / 12, abs=1e-6)
    assert str(refs["always_strongest"]["cost_saved"]) == "0.0"                                 # never "-0.0"


def saved_like(p):
    return {"quality": p["correct"] / p["n"], "n": p["n"], "cost_saved": p["cost_saved"],
            "quality_retained": p["quality_kept"], "cost_per_1k": p["cost_per_1k_usd"], "mix": dict(p["mix"])}


def test_the_export_stops_unless_it_reproduces_the_pre_registered_result():
    data = rows()
    refs = eb.references(data)
    headline = refs["oracle"]
    saved = {"predict_only_balanced": saved_like(headline), "always_medium": saved_like(refs["always_medium"]),
             "always_high": saved_like(refs["always_strongest"]), "oracle": saved_like(refs["oracle"])}
    eb.check_reproduces(headline, refs, saved)                                          # identical: fine
    for change in ({"quality": 0.0}, {"cost_saved": 0.5}, {"mix": {"low": 0.0, "medium": 0.0, "high": 1.0}}):
        broken = {**saved, "predict_only_balanced": {**saved["predict_only_balanced"], **change}}
        with pytest.raises(eb.ExportError, match="predict_only_balanced"):
            eb.check_reproduces(headline, refs, broken)


# ---------- the committed file ----------

@pytest.fixture(scope="module")
def committed():
    return json.loads(eb.OUT_PATH.read_text(encoding="utf-8"))


def test_the_committed_file_has_every_section(committed):
    assert committed["schema"] == eb.SCHEMA and committed["classifier"] == "classifier_c_logreg_cv v2"
    assert set(committed) == {"schema", "classifier", "split", "headline", "vs_always_cheapest", "references",
                              "presets", "tier_mix_default", "curve", "classifiers", "judge", "dataset", "caveats"}
    assert set(committed["references"]) == {"always_cheapest", "always_medium", "always_strongest", "oracle"}
    assert set(committed["presets"]) == {"cheap", "balanced", "high"}
    assert committed["split"]["n"] == committed["dataset"]["test_labeled"] == 149


def test_the_committed_headline_is_the_pre_registered_result(committed):
    phase4 = json.loads((PROJECT_ROOT / "reports" / "final_test_phase4.json").read_text(encoding="utf-8"))
    saved = phase4["real_prices"]["predict_only_balanced"]
    h = committed["headline"]
    assert h["tau"] == 0.86 and h["correct"] == saved["correct"] == 144
    assert h["cost_saved"] == pytest.approx(saved["cost_saved"], abs=1e-6)
    assert h["provenance"].startswith("pre-registered") and committed["presets"]["balanced"]["provenance"] == h["provenance"]
    assert committed["tier_mix_default"] == h["mix"]


def test_the_committed_curve_is_labeled_for_reporting_only(committed):
    assert committed["curve"]["provenance"] == eb.CURVE_PROVENANCE and "not used for tuning" in eb.CURVE_PROVENANCE
    points = committed["curve"]["points"]
    assert [p["tau"] for p in points] == eb.TAUS
    assert all(abs(sum(p["mix"].values()) - 1) < 1e-5 for p in points)
    for name in ("cheap", "high"):
        assert committed["presets"][name]["provenance"] == eb.CURVE_PROVENANCE
    assert "-0.0" not in eb.OUT_PATH.read_text(encoding="utf-8")


def test_the_report_copy_matches_the_dashboard_file(committed):
    copy = json.loads(eb.REPORT_COPY.read_text(encoding="utf-8"))
    assert copy == {k: committed[k] for k in copy} and "curve" in copy
