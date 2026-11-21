"""Regenerate the figures used in CASE_STUDY.md from the committed JSON reports.

    python scripts/case_study_figures.py

Every number plotted is read from a file in reports/ or models/; nothing is typed in by hand.
Output: PNGs in docs/case-study/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
OUT = ROOT / "docs" / "case-study"

# Light theme; categorical hues in a fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
TIER_COLORS = {"low": AQUA, "medium": BLUE, "high": VIOLET}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def caption(fig, text: str) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=9, color=INK_2, style="italic")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def cost_vs_quality() -> None:
    d = load(REPORTS / "benchmark_v2_test_curve.json")
    pts = d["curve"]["points"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot([p["cost_per_1k_usd"] for p in pts], [100 * p["quality_kept"] for p in pts],
            color=BLUE, lw=2, label="Triage, classifier v2 (τ swept 0 → 1)", zorder=3)
    refs = d["references"]
    marks = [("always_cheapest", "Always cheapest", "v", AQUA),
             ("always_medium", "Always medium", "s", ORANGE),
             ("always_strongest", "Always strongest", "^", VIOLET),
             ("oracle", "Perfect router (oracle)", "*", YELLOW)]
    for key, label, m, c in marks:
        r = refs[key]
        ax.scatter(r["cost_per_1k_usd"], 100 * r["quality_kept"], marker=m, s=160 if m == "*" else 90,
                   color=c, edgecolor=INK, linewidth=0.8, zorder=4,
                   label=f"{label}: {r['correct']}/{r['n']}")
    offsets = {"cheap": (55, -36), "balanced": (-30, -42), "high": (10, -62)}
    for name, p in d["presets"].items():
        x, y = p["cost_per_1k_usd"], 100 * p["quality_kept"]
        ax.scatter(x, y, s=70, color=INK, zorder=5)
        ax.annotate(f"{name} (τ {p['tau']})\n{100 * p['quality_kept']:.1f}% kept, "
                    f"{100 * p['cost_saved']:.0f}% saved", (x, y), textcoords="offset points",
                    xytext=offsets[name], fontsize=8.5, color=INK, ha="center",
                    arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    ax.set_xlabel("Cost per 1,000 requests (USD, Groq paid prices)")
    ax.set_ylabel("Quality kept (% of always-strongest)")
    ax.set_title("Cost vs quality on the locked test set (149 prompts)", loc="left", fontsize=12)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.subplots_adjust(bottom=0.16)
    caption(fig, "Source: reports/benchmark_v2_test_curve.json. Balanced was pre-registered; "
                 "the rest of the curve was computed afterwards, for reporting only.")
    save(fig, "cost_vs_quality_v2.png")


def tier_mix() -> None:
    d = load(REPORTS / "benchmark_v2_test_curve.json")
    rows = [(f"Triage cheap (τ {d['presets']['cheap']['tau']})", d["presets"]["cheap"]),
            (f"Triage balanced (τ {d['presets']['balanced']['tau']})", d["presets"]["balanced"]),
            (f"Triage high (τ {d['presets']['high']['tau']})", d["presets"]["high"]),
            ("Perfect router (oracle)", d["references"]["oracle"]),
            ("Always medium", d["references"]["always_medium"])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [r[0] for r in rows][::-1]
    data = [r[1] for r in rows][::-1]
    for i, r in enumerate(data):
        left = 0.0
        for tier in ("low", "medium", "high"):
            share = 100 * r["mix"][tier]
            if share <= 0:
                continue
            ax.barh(i, share, left=left, color=TIER_COLORS[tier], edgecolor=SURFACE, linewidth=2, height=0.6)
            if share >= 6:
                ax.text(left + share / 2, i, f"{share:.0f}%", ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")
            left += share
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the 149 test prompts sent to each tier (%)")
    ax.set_title("Where each preset sends traffic", loc="left", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t]) for t in ("low", "medium", "high")]
    ax.legend(handles, ["low (ALLaM 2 7B)", "medium (gpt-oss-20b)", "high (gpt-oss-120b)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.27, bottom=0.32, right=0.97)
    caption(fig, "Source: reports/benchmark_v2_test_curve.json. No preset ever picks the low tier; "
                 "the oracle would send about half the prompts there.")
    save(fig, "tier_mix_presets.png")


def classifier_auc() -> None:
    entries = [("Task type only\n(baseline)", "baseline_task_only"),
               ("A: handcrafted\nfeatures, tuned", "classifier_a_logreg_cv"),
               ("B: embeddings,\nfixed C=1", "classifier_b"),
               ("B: embeddings,\ntuned", "classifier_b_logreg_cv"),
               ("C: both, tuned\nlogistic (chosen)", "classifier_c_logreg_cv"),
               ("C: both,\ngradient boosting", "classifier_c_gboost")]
    low, med = [], []
    for _, stem in entries:
        low.append(load(MODELS / f"{stem}_low_v1.json")["cv"]["roc_auc"])
        med.append(load(MODELS / f"{stem}_medium_v1.json")["cv"]["roc_auc"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    xs = range(len(entries))
    w = 0.36
    b1 = ax.bar([x - w / 2 - 0.01 for x in xs], low, w, color=AQUA, label="Will the cheap tier pass?")
    b2 = ax.bar([x + w / 2 + 0.01 for x in xs], med, w, color=BLUE, label="Will the medium tier pass?")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.axhline(0.5, color=INK_2, lw=1, ls="--", label="Chance (0.5)")
    ax.set_xticks(list(xs), [e[0] for e in entries], fontsize=8.5)
    ax.set_ylim(0.45, 0.72)
    ax.set_ylabel("ROC AUC (5-fold cross-validation)")
    ax.set_title("Classifier comparison on the training split (598 prompts)", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(bottom=0.24)
    caption(fig, "Source: models/*_v1.json (cv.roc_auc). Every model ranks prompts only modestly "
                 "better than chance.")
    save(fig, "classifier_auc.png")


def judge_agreement() -> None:
    d = load(REPORTS / "judge_calibration.json")
    g = d["groups"]
    rows = [("All 150 answers", d["all"]),
            ("GSM8K", g["dataset=gsm8k"]), ("MMLU", g["dataset=mmlu"]),
            ("Cheap-tier answers", g["tier=low"]), ("Medium-tier answers", g["tier=medium"]),
            ("Reference answer right", g["reference=reference right"]),
            ("Reference answer wrong", g["reference=reference WRONG"])]
    parts = [("right_judged_pass", "Right, judged pass", GREEN),
             ("wrong_judged_fail", "Wrong, judged fail", AQUA),
             ("right_judged_fail", "Right, judged fail (too strict)", ORANGE),
             ("wrong_judged_pass", "Wrong, judged pass (too lenient)", RED)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    labels = [r[0] for r in rows][::-1]
    for i, (_, r) in enumerate(rows[::-1]):
        c = r["confusion"]
        n = sum(c.values())
        left = 0.0
        for key, _, color in parts:
            share = 100 * c[key] / n
            if share > 0:
                ax.barh(i, share, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.62)
                if share >= 7:
                    ax.text(left + share / 2, i, str(c[key]), ha="center", va="center", color="white",
                            fontsize=8.5, fontweight="bold")
            left += share
        ax.text(101, i, f"{100 * r['agreement']:.0f}% agree (n={n})", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of answers (%); numbers in bars are counts")
    ax.set_title("LLM judge vs known truth (150 GSM8K/MMLU answers, pass at 4 of 5)", loc="left", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=p[2]) for p in parts]
    ax.legend(handles, [p[1] for p in parts], loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2,
              fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.22, bottom=0.3, right=0.97)
    caption(fig, "Source: reports/judge_calibration.json. Strict errors are all cheap answers; "
                 "lenient ones all come from a wrong reference answer.")
    save(fig, "judge_agreement.png")


def latency_per_tier() -> None:
    d = load(REPORTS / "latency_p95.json")["tiers"]
    names = {"low": "low\nALLaM 2 7B", "medium": "medium\ngpt-oss-20b", "high": "high\ngpt-oss-120b"}
    pcts = [("p50_ms", "p50", AQUA), ("p90_ms", "p90", BLUE), ("p95_ms", "p95", VIOLET), ("p99_ms", "p99", ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.19
    tiers = ["low", "medium", "high"]
    for j, (key, label, color) in enumerate(pcts):
        xs = [i + (j - 1.5) * (w + 0.01) for i in range(len(tiers))]
        vals = [d[t][key] / 1000 for t in tiers]
        bars = ax.bar(xs, vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(tiers)), [f"{names[t]}\n(n={d[t]['n']})" for t in tiers])
    ax.set_ylabel("Latency per answer (seconds)")
    ax.set_title("Answer latency by tier (Phase 2 calls, training split)", loc="left", fontsize=12)
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(bottom=0.24)
    caption(fig, "Source: reports/latency_p95.json (free-tier Groq calls from a laptop; cache hits and "
                 "errors excluded). These set the timeouts.")
    save(fig, "latency_per_tier.png")


def checker_calibration() -> None:
    d = load(REPORTS / "checker_calibration.json")["models"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    metrics = [("catch_rate", "Catch rate\n(bad answer flagged)"), ("false_alarm", "False-alarm rate\n(good answer flagged)")]
    w = 0.34
    for k, (model, color) in enumerate((("groq-small", BLUE), ("groq-large", ORANGE))):
        t = d[model]["trust_reweighted"]["medium"]
        sample = d[model]["by_tier (sample, unweighted)"]["medium"]
        vals = [100 * t["pool_weighted_catch_rate"], 100 * t["pool_weighted_false_alarm"]]
        ci = sample["catch_rate_ci"]   # every bad medium answer in the pool was sampled, so this CI applies
        xs = [i + (k - 0.5) * (w + 0.02) for i in range(len(metrics))]
        ax.bar(xs, vals, w, color=color, label=f"{model} as checker")
        ax.errorbar(xs[0], vals[0], yerr=[[vals[0] - 100 * ci[0]], [100 * ci[1] - vals[0]]], color=INK_2,
                    capsize=4, lw=1)
        for x, v in zip(xs, vals):
            ax.text(x, v / 2 if v > 10 else v + 1.5, f"{v:.1f}%", ha="center", va="center" if v > 10 else "bottom",
                    fontsize=9, color="white" if v > 10 else INK, fontweight="bold")
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics])
    ax.set_ylabel("% of medium-tier answers")
    ax.set_title("AI answer checker on 241 medium answers (fit split)", loc="left", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(bottom=0.22)
    caption(fig, "Source: reports/checker_calibration.json (pool-weighted; whisker: 95% CI). "
                 "Either checker misses about half of the bad answers.")
    save(fig, "checker_calibration.png")


if __name__ == "__main__":
    cost_vs_quality()
    tier_mix()
    classifier_auc()
    judge_agreement()
    latency_per_tier()
    checker_calibration()
