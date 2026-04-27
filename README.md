# Validation of LLMs for the Accurate Identification of Clinical Sepsis Cases

Analysis of a three-agent GPT-5.2 pipeline developed to retrospectively identify sepsis cases from electronic health records (EHR) at University of Michigan, Michigan Medicine.

## Overview

This project implements and evaluates a three-agent LLM pipeline for automated retrospective sepsis identification from EHR data. The pipeline was evaluated against physician review as the gold standard across 187 hospitalization encounters. Three pipeline versions were tested through iterative prompt engineering to optimize performance.

## Repository Structure

```
Validation-of-LLMs-for-the-Accurate-Identification-of-Clinical-Sepsis-Cases/
│
├── pipelines/
│   ├── sepsis_pipeline_1.py        # V1 — baseline prompt, no clinical guidance
│   ├── sepsis_pipeline_2.py        # V2 — Sepsis-3 definition + 5 few-shot examples
│   └── sepsis_pipeline_3.py        # V3 — Sepsis-3 definition only
│
├── analysis/
│   └── accuracy_analysis_all_versions.py
│
├── requirements.txt
└── README.md
```

---

## Pipeline Architecture

Each patient encounter is processed through three sequential specialized agents:

**Agent 1 — Infection Agent**
Identifies infection presence, organism type (bacterial/viral/fungal/unknown), body system affected, onset time, and treatment agent.

**Agent 2 — Organ Dysfunction Agent**
Identifies affected organ systems (renal, circulatory, pulmonary, neurologic, hepatic, metabolic, hematologic) and earliest onset times.

**Agent 3 — Sepsis Agent**
Integrates outputs from both agents with full clinical data to determine sepsis presence, infectious source, onset time, and likelihood rating (definite/probable/possible/unlikely). This is the only agent that differs across the three pipeline versions.

All agents receive full untruncated clinical notes, laboratory values, vital signs, and medication records for the first 14 days of hospitalization.

---

## Model & Parameter Configuration

| Parameter | Value |
|---|---|
| Model | GPT-5.2 via UMGPT Toolkit (Azure OpenAI) |
| API Version | 2024-02-01 |
| Context Window | 400,000 tokens (no truncation applied) |
| max_completion_tokens | 1000 |
| Timeout | 600 seconds per API call |
| Parallel Workers | 4 (ThreadPoolExecutor) |
| Output Format | Structured JSON per agent |
| Checkpointing | Per-case JSONL append (resume on failure) |

---

## Pipeline Versions

| Version | File | Sepsis Agent Change | Key Result |
|---|---|---|---|
| V1 | sepsis_pipeline_1.py | Baseline prompt only | Sensitivity 0.909, PPV 0.513, F1 0.656 |
| V2 | sepsis_pipeline_2.py | Sepsis-3 definition + 5 few-shot examples | Sensitivity 0.727, PPV 0.780, F1 0.753 ✓ |
| V3 | sepsis_pipeline_3.py | Sepsis-3 definition only | Sensitivity 0.773, PPV 0.708, F1 0.739 |

**Recommended: V2** — best F1 (0.753), accuracy (0.888), PPV (0.780), and specificity (0.937).

---

## Results Summary

| Metric | V1 (Baseline) | V2 (Sepsis-3 + FS) | V3 (Sepsis-3) |
|---|---|---|---|
| Sensitivity | 0.909 | 0.727 | 0.773 |
| Specificity | 0.734 | 0.937 | 0.902 |
| PPV | 0.513 | **0.780** | 0.708 |
| NPV | 0.963 | 0.918 | 0.928 |
| F1 Score | 0.656 | **0.753** | 0.739 |
| Accuracy | 0.775 | **0.888** | 0.872 |
| TP/TN/FP/FN | 40/105/38/4 | 32/134/9/12 | 34/129/14/10 |

Evaluated against physician review as gold standard across 187 cases (96 Sepsis-3 positive, 91 negative). Negative cases treated as true negatives pending negative physician review file.

---

## Requirements

```bash
pip install openai pandas scikit-learn
```

---

## Data Requirements

Data files are not included in this repository (confidential patient data). Place all input files in a `data/` folder in the root of the repository before running.

| File | Columns | Description |
|---|---|---|
| pos_sepsis3_clinical_note.csv | CSN, Note | Clinical notes — Sepsis-3 positive cases (96) |
| neg_sepsis3_clinical_note.csv | CSN, Note | Clinical notes — Sepsis-3 negative cases (91) |
| pos_sepsis3_lab_medication.csv | CSN, Lab_data, Vital_sign, Medication | Lab/vitals/meds — positive cases |
| neg_sepsis3_lab_medication.csv | CSN, Lab_data, Vital_sign, Medication | Lab/vitals/meds — negative cases |
| pos_sepsis3_physician_review_result.csv | CSN, Sepsis_review_result | Physician review labels (Positive/Negative) |

---

## How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/alinafais/Validation-of-LLMs-for-the-Accurate-Identification-of-Clinical-Sepsis-Cases.git
cd Validation-of-LLMs-for-the-Accurate-Identification-of-Clinical-Sepsis-Cases
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add your data

Create a `data/` folder and place all five input CSV files inside it.

### Step 4 — Configure API credentials

Open each pipeline file and fill in your credentials:

```python
client = AzureOpenAI(
    api_key="YOUR_API_KEY",
    azure_endpoint="YOUR_ENDPOINT_URL",
    api_version="2024-02-01",
    organization="YOUR_ORG_CODE"
)
```

### Step 5 — Create output directories

```bash
mkdir -p results/sepsis_analysis_1
mkdir -p results/sepsis_analysis_2
mkdir -p results/sepsis_analysis_3
```

### Step 6 — Run the pipelines

```bash
# V1 - Baseline
nohup python pipelines/sepsis_pipeline_1.py > results/sepsis_analysis_1/pipeline_v1_log.txt 2>&1 &

# V2 - Sepsis-3 + Few-shot (recommended)
nohup python pipelines/sepsis_pipeline_2.py > results/sepsis_analysis_2/pipeline_v2_log.txt 2>&1 &

# V3 - Sepsis-3 only
nohup python pipelines/sepsis_pipeline_3.py > results/sepsis_analysis_3/pipeline_v3_log.txt 2>&1 &
```

Monitor progress:

```bash
tail -f results/sepsis_analysis_1/pipeline_v1_log.txt
```

Each pipeline run takes approximately 2-4 hours for 187 cases with 4 parallel workers. The pipeline uses checkpointing and if interrupted, simply rerun the same command and it will skip already-completed cases.

### Step 7 — Verify results

```bash
python -c "
import json
for v, path in [('V1', 'results/sepsis_analysis_1/llm_results.jsonl'),
                ('V2', 'results/sepsis_analysis_2/llm_results_(2).jsonl'),
                ('V3', 'results/sepsis_analysis_3/llm_results_v3.jsonl')]:
    results = [json.loads(l) for l in open(path)]
    errors = [r for r in results if 'error' in r.get('infection_agent', {})]
    unique = len(set(r['CSN'] for r in results))
    print(f'{v}: Total={len(results)}, Errors={len(errors)}, Unique={unique}')
"
```

Expected output:
V1: Total=187, Errors=0, Unique=187
V2: Total=187, Errors=0, Unique=187
V3: Total=187, Errors=0, Unique=187

### Step 8 — Run accuracy analysis

```bash
python analysis/accuracy_analysis_all_versions.py
```

This prints the comparison table across all three versions, threshold tuning analysis, likelihood distributions, and saves CSV accuracy results to each results folder.

---

## Output Format

Each pipeline produces a `.jsonl` file where each line is one patient case:

```json
{
  "CSN": 373117969,
  "infection_agent": {
    "infection_present": true,
    "infections": [
      {
        "organism_type": "bacterial",
        "body_system": "renal/urinary",
        "onset_time": "2024-12-15 17:00:00",
        "treated": true,
        "treatment_agent": "piperacillin-tazobactam"
      }
    ],
    "reasoning": "Clinical justification..."
  },
  "organ_dysfunction_agent": {
    "organ_dysfunction_present": true,
    "organ_systems": [
      {"system": "renal", "onset_time": "2024-12-15 17:52:00"}
    ],
    "reasoning": "Clinical justification..."
  },
  "sepsis_agent": {
    "sepsis_present": true,
    "infectious_source": "Complicated UTI with Klebsiella bacteremia",
    "sepsis_onset_time": "2024-12-15 17:52:00",
    "likelihood": "probable",
    "reasoning": "Clinical justification..."
  }
}
```

## Limitations

- Negative physician review file was unavailable so 91 Sepsis-3 negative cases treated as true negatives, potentially overestimating specificity.
- V2/V3 sensitivity tradeoff — model becomes more conservative, missing more true sepsis cases (FN increases from 4 to 12/10).
- Single site as results represent Michigan Medicine only. Multicenter generalizability across UCSD, UCLA, UCI is unknown.
- Sepsis onset time accuracy unvalidated and physician onset timestamps were not available for comparison.

## License

MIT License - see LICENSE file for details.
