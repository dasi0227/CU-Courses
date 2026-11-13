from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ftec5660-hw4-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import comparison
import hw4


VERSIONS = ("B0", "V1", "V2", "V3", "V4")
CATEGORIES = ("fact", "table", "comparison", "unanswerable")
CATEGORY_LABELS = ("Fact", "Table", "Comparison", "Unanswerable")
INK = "#172B43"
BLUE = "#2865BD"
TEAL = "#148579"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.titleweight": "semibold",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CBD3DF",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
})

corpus = hw4.load_corpus(ROOT / "public_test/corpus")
questions = hw4.load_json(ROOT / "public_test/questions.json")
gold = hw4.load_json(ROOT / "public_test/ground_truth.json")
document_metadata = hw4.load_json(ROOT / "public_test/docs.json")
tables = hw4.load_json(ROOT / "public_test/tables.json")
metrics = {}
records = {}

for version in VERSIONS:
    with (ROOT / f"results_{version}.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(questions)
    assert {row["id"] for row in rows} == set(questions)
    assert all(row["question"] == questions[row["id"]] for row in rows)
    assert all((row["correct"] == "True") == hw4.answer_correct(row["answer"], gold[row["id"]]) for row in rows)
    records[version] = rows

    answerable = [row for row in rows if gold[row["id"]]["evidence"]]
    unanswerable = [row for row in rows if not gold[row["id"]]["evidence"]]
    correct = sum(row["correct"] == "True" for row in rows)
    hits = sum(row["retrieval_hit"] == "True" for row in answerable)
    kept_count = sum(bool(row["retrieved"]) for row in answerable)
    rejected_count = sum(not row["retrieved"] for row in unanswerable)
    accuracy = correct / len(rows)
    recall = hits / len(answerable)
    kept = kept_count / len(answerable)
    rejected = rejected_count / len(unanswerable)
    valid = sum(row["valid"] == "True" and bool(row["retrieved"]) for row in answerable) / len(answerable)
    parse_accuracy = sum(float(row["parse_score"]) for row in rows) / len(rows)

    comparison.VARIANT = version
    table_scores = [hw4.table_score(comparison.table_to_text(table["html"]), table) for table in tables]
    chunks = []
    passed = 0
    for name, html in corpus.items():
        result = comparison.chunk_document(name, html)
        passed += not hw4.document_problems(result, None, document_metadata[name], name)
        chunks.extend(result)

    points = {
        "1.1": hw4.POINTS["1.1"] * sum(table_scores) / len(table_scores),
        "1.2": hw4.POINTS["1.2"] * passed / len(corpus),
        "2.1": hw4.POINTS["2.1"] * valid,
        "2.2": hw4.POINTS["2.2"] * parse_accuracy,
        "2.3": hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO) / (hw4.RECALL_FULL - hw4.RECALL_ZERO)),
        "3": hw4.POINTS["3"] * max(0, kept + rejected - 1),
        "4": hw4.POINTS["4"] * accuracy,
    }
    categories = {}
    for category in CATEGORIES:
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {
            "correct": sum(row["correct"] == "True" for row in subset),
            "total": len(subset),
        }

    metrics[version] = {
        "correct": correct,
        "total": len(rows),
        "answer_accuracy": accuracy,
        "retrieval_hits": hits,
        "answerable_count": len(answerable),
        "recall_at_5": recall,
        "answerable_kept_count": kept_count,
        "answerable_kept": kept,
        "unanswerable_rejected_count": rejected_count,
        "unanswerable_count": len(unanswerable),
        "unanswerable_rejected": rejected,
        "query_parse_accuracy": parse_accuracy,
        "chunk_count": len(chunks),
        "points": points,
        "automatic_points": sum(points.values()),
        "categories": categories,
    }

v4_chunks = chunks
index = hw4.build_index(v4_chunks)
candidate_hits = 0
candidate_misses = []
for row in records["V4"]:
    if not gold[row["id"]]["evidence"]:
        continue
    parsed = json.loads(row["parsed"])
    companies = {hw4.company_key(name) for name in parsed["companies"]}
    tokens = hw4.re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", row["question"].lower())
    scores_a = index["bm25_a"].get_scores(tokens)
    scores_b = index["bm25_b"].get_scores(tokens)
    route_a = sorted(range(len(v4_chunks)), key=lambda i: scores_a[i], reverse=True)[:hw4.MAX_RETRIEVED]
    route_b = [i for i, chunk in enumerate(v4_chunks)
               if hw4.company_key(chunk.metadata["company"]) in companies
               and chunk.metadata["doc_type"] == parsed["doc_type"]]
    route_b = sorted(route_b, key=lambda i: scores_b[i], reverse=True)[:hw4.MAX_RETRIEVED]
    candidates = [v4_chunks[i] for i in dict.fromkeys(route_a + route_b)]
    if hw4.retrieval_hit(candidates, gold[row["id"]]):
        candidate_hits += 1
    else:
        candidate_misses.append(row["id"])

summary = {
    "variants": metrics,
    "v4_candidate_audit": {
        "complete_evidence_hits": candidate_hits,
        "answerable_count": metrics["V4"]["answerable_count"],
        "missing_question_ids": candidate_misses,
    },
    "v4_errors": [{"id": row["id"], "question": row["question"],
                   "answer": row["answer"], "expected": row["expected"],
                   "retrieved_sources": row["retrieved"]}
                  for row in records["V4"] if row["correct"] != "True"],
}
assert hw4.USAGE.llm_calls == 0
ASSETS.mkdir(exist_ok=True)
(ASSETS / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

x = list(range(len(VERSIONS)))
for filename, title, series in (
    ("retrieval_accuracy.png", "Retrieval and answer accuracy",
     (("answer_accuracy", "Answer accuracy: 60 questions", BLUE),
      ("recall_at_5", "Recall@5: 46 answerable questions", TEAL))),
    ("abstention.png", "Answerability decisions before generation",
     (("answerable_kept", "Answerable questions kept: 46", BLUE),
      ("unanswerable_rejected", "Unanswerable questions rejected: 14", TEAL))),
):
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.22)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    for offset, (key, label, color) in zip((-0.18, 0.18), series):
        values = [100 * metrics[version][key] for version in VERSIONS]
        bars = ax.bar([value + offset for value in x], values, width=0.33, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=4, fontsize=10)
    ax.set_xticks(x, VERSIONS)
    ax.set_ylim(0, 113)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylabel("Questions (%)")
    ax.set_title(title, loc="center", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=10)
    fig.savefig(ASSETS / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(10.5, 6))
fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.25)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
bottom = [0.0] * len(VERSIONS)
components = (
    ("1.1", "Table parsing", "#A9BACD"),
    ("1.2", "Chunking", "#60978E"),
    ("2.1", "Valid retrieval", "#A8C3E8"),
    ("2.2", "Query parsing", "#8779AC"),
    ("2.3", "Evidence recall", "#4B8CC5"),
    ("3", "Abstention", "#D6A34B"),
    ("4", "Answer accuracy", "#224E87"),
)
for key, label, color in components:
    values = [metrics[version]["points"][key] for version in VERSIONS]
    ax.bar(x, values, bottom=bottom, width=0.6, label=label, color=color)
    bottom = [current + value for current, value in zip(bottom, values)]
for position, total in zip(x, bottom):
    ax.text(position, total + 1, f"{total:.2f}", ha="center", va="bottom", fontsize=11, fontweight="semibold")
ax.set_xticks(x, VERSIONS)
ax.set_ylim(0, 80)
ax.set_yticks((0, 15, 30, 45, 60, 75))
ax.set_ylabel("Automatic points (maximum 75)")
ax.set_title("Automatic score by task", loc="center", pad=22)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=10)
fig.savefig(ASSETS / "score_breakdown.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

matrix = [[100 * metrics[version]["categories"][category]["correct"]
           / metrics[version]["categories"][category]["total"] for version in VERSIONS]
          for category in CATEGORIES]
fig, ax = plt.subplots(figsize=(10.5, 5.6), layout="constrained")
image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(x, VERSIONS)
ax.set_yticks(range(len(CATEGORIES)), CATEGORY_LABELS)
ax.set_title("Answer accuracy by question category", loc="center", pad=18)
ax.tick_params(length=0, pad=10)
for row_index, category in enumerate(CATEGORIES):
    for column_index, version in enumerate(VERSIONS):
        counts = metrics[version]["categories"][category]
        value = matrix[row_index][column_index]
        ax.text(column_index, row_index, f"{counts['correct']}/{counts['total']}\n{value:.1f}%",
                ha="center", va="center", fontsize=11,
                color="white" if value >= 60 else INK)
colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
colorbar.set_label("Correct answers (%)", labelpad=10)
colorbar.set_ticks((0, 20, 40, 60, 80, 100))
fig.savefig(ASSETS / "category_accuracy.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

for version in VERSIONS:
    item = metrics[version]
    print(f"{version}: accuracy={item['answer_accuracy']:.1%}, recall={item['recall_at_5']:.1%}, points={item['automatic_points']:.2f}/75")
print(f"V4 candidate evidence: {candidate_hits}/{metrics['V4']['answerable_count']}; missing: {candidate_misses}")
print(f"Figures and metrics saved to {ASSETS}")
