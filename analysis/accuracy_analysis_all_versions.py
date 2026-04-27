"""
accuracy_analysis_all_versions.py — Consolidated Accuracy Analysis
==================================================================
Accuracy analysis script comparing all three pipeline versions against
physician review as the gold standard.
University of Michigan, Michigan Medicine — April 2026

DESCRIPTION:
    Loads LLM results from all three pipeline versions (V1, V2, V3),
    merges with physician review labels, and computes accuracy metrics
    including sensitivity, specificity, PPV, NPV, F1, and accuracy.
    Also performs threshold tuning using the likelihood field and prints
    likelihood distributions for each version.

PIPELINE VERSIONS COMPARED:
    V1 — Baseline prompt only (no clinical guidance)
    V2 — Sepsis-3 definition + 5 few-shot examples (recommended)
    V3 — Sepsis-3 definition only (no examples)

INPUT FILES REQUIRED:
    LLM results (from running pipelines):
        - results/sepsis_analysis_1/llm_results.jsonl
        - results/sepsis_analysis_2/llm_results_(2).jsonl
        - results/sepsis_analysis_3/llm_results_v3.jsonl

    Physician review (place in data/ folder):
        - data/pos_sepsis3_physician_review_result.csv  (CSN, Sepsis_review_result)
        - data/neg_sepsis3_clinical_note.csv            (CSN, Note)
          Note: Negative cases treated as true negatives pending
          neg_sepsis3_physician_review_result.csv

OUTPUT:
    Printed comparison table, likelihood distributions, and threshold
    tuning results for all three versions. CSV results saved to:
        - results/sepsis_analysis_1/accuracy_results_v1.csv
        - results/sepsis_analysis_2/accuracy_results_v2.csv
        - results/sepsis_analysis_3/accuracy_results_v3.csv

USAGE:
    python analysis/accuracy_analysis_all_versions.py

REQUIREMENTS:
    pip install pandas scikit-learn
"""

import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# ── LOAD ALL 3 VERSIONS ───────────────────────────────────────────────────────
def load_results(filepath):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass
    df = pd.DataFrame({
        "CSN": [r["CSN"] for r in results],
        "llm_sepsis": [r["sepsis_agent"].get("sepsis_present", None) for r in results],
        "likelihood": [r["sepsis_agent"].get("likelihood", None) for r in results]
    })
    df = df.drop_duplicates(subset="CSN", keep="last")
    df = df.dropna(subset=["llm_sepsis"])
    df["llm_sepsis"] = df["llm_sepsis"].astype(bool)
    return df

v1 = load_results("results/sepsis_analysis_1/llm_results.jsonl")
v2 = load_results("results/sepsis_analysis_2/llm_results_(2).jsonl")
v3 = load_results("results/sepsis_analysis_3/llm_results_v3.jsonl")

print(f"V1: {len(v1)} cases | V2: {len(v2)} cases | V3: {len(v3)} cases")

# ── LOAD PHYSICIAN REVIEW ─────────────────────────────────────────────────────
pos_review = pd.read_csv("data/pos_sepsis3_physician_review_result.csv")
neg_notes = pd.read_csv("data/neg_sepsis3_clinical_note.csv")
neg_review = pd.DataFrame({"CSN": neg_notes["CSN"], "Sepsis_review_result": "Negative"})
physician_review = pd.concat([pos_review, neg_review], ignore_index=True)

# ── METRICS FUNCTION ──────────────────────────────────────────────────────────
def get_metrics(llm_df, physician_review):
    merged = llm_df.merge(physician_review, on="CSN", how="inner")
    merged["physician_sepsis"] = merged["Sepsis_review_result"].map({"Positive": True, "Negative": False})
    y_true = merged["physician_sepsis"]
    y_pred = merged["llm_sepsis"]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1   = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0
    acc  = (tp + tn) / (tp + tn + fp + fn)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Sensitivity": sens, "Specificity": spec,
            "PPV": ppv, "NPV": npv, "F1": f1, "Accuracy": acc,
            "merged": merged}

m1 = get_metrics(v1, physician_review)
m2 = get_metrics(v2, physician_review)
m3 = get_metrics(v3, physician_review)

# ── PRINT COMPARISON ──────────────────────────────────────────────────────────
print(f"\n{'Metric':<20} {'V1 (Baseline)':>15} {'V2 (Sepsis3+FS)':>16} {'V3 (Sepsis3)':>13}")
print("-" * 68)
for metric in ["Sensitivity", "Specificity", "PPV", "NPV", "F1", "Accuracy"]:
    print(f"{metric:<20} {m1[metric]:>15.3f} {m2[metric]:>16.3f} {m3[metric]:>13.3f}")
print(f"{'TP/TN/FP/FN':<20} {str(m1['TP'])+'/'+str(m1['TN'])+'/'+str(m1['FP'])+'/'+str(m1['FN']):>15} {str(m2['TP'])+'/'+str(m2['TN'])+'/'+str(m2['FP'])+'/'+str(m2['FN']):>16} {str(m3['TP'])+'/'+str(m3['TN'])+'/'+str(m3['FP'])+'/'+str(m3['FN']):>13}")

# ── LIKELIHOOD DISTRIBUTIONS ──────────────────────────────────────────────────
print(f"\n=== LIKELIHOOD DISTRIBUTIONS ===")
for name, df in [("V1", v1), ("V2", v2), ("V3", v3)]:
    print(f"\n{name}:")
    print(df["likelihood"].value_counts().to_string())

# ── THRESHOLD ANALYSIS FOR EACH VERSION ───────────────────────────────────────
thresholds = {
    "definite only":         ["definite"],
    "definite + probable":   ["definite", "probable"],
    "def + prob + possible": ["definite", "probable", "possible"],
    "any true (baseline)":   ["definite", "probable", "possible", "unlikely"]
}

for name, df, m in [("V1", v1, m1), ("V2", v2, m2), ("V3", v3, m3)]:
    print(f"\n=== THRESHOLD ANALYSIS: {name} ===")
    print(f"{'Threshold':<30} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6} {'F1':>6} {'Acc':>6} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print("-" * 90)
    merged = m["merged"]
    for tname, accepted in thresholds.items():
        merged["pred"] = merged.apply(
            lambda r: True if r["likelihood"] in accepted and r["llm_sepsis"] == True else False, axis=1
        )
        y_true = merged["physician_sepsis"]
        y_pred = merged["pred"]
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1   = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0
        acc  = (tp + tn) / (tp + tn + fp + fn)
        print(f"{tname:<30} {sens:>6.3f} {spec:>6.3f} {ppv:>6.3f} {npv:>6.3f} {f1:>6.3f} {acc:>6.3f} {tp:>4} {tn:>4} {fp:>4} {fn:>4}")

# ── SAVE ALL RESULTS ──────────────────────────────────────────────────────────
m1["merged"].to_csv("results/sepsis_analysis_1/accuracy_results_v1.csv", index=False)
m2["merged"].to_csv("results/sepsis_analysis_2/accuracy_results_v2.csv", index=False)
m3["merged"].to_csv("results/sepsis_analysis_3/accuracy_results_v3.csv", index=False)
print(f"\nResults saved to results/sepsis_analysis_1, results/sepsis_analysis_2, and results/sepsis_analysis_3 folders.")
print("Note: Negative cases treated as true negatives pending neg_sepsis3_physician_review_result.csv")
