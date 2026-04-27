"""
sepsis_pipeline_3.py — V3 Sepsis-3 Definition Only Pipeline
============================================================
Three-agent LLM pipeline for retrospective sepsis identification from EHR data.
University of Michigan, Michigan Medicine — April 2026

VERSION: V3 — Sepsis-3 Definition Only (No Few-Shot Examples)
    The Sepsis Agent prompt was updated with only the formal Sepsis-3 definition
    and important distinctions — no few-shot examples were included.
    This version was created for two purposes:
    1. To isolate the effect of the Sepsis-3 definition alone, separate from
       the few-shot examples used in V2, to determine which component drove
       the V2 improvement.
    2. To address timeout issues caused by V2's longer prompt on large
       clinical notes (mean: 189,742 chars, max: 662,694 chars).

RATIONALE:
    V3 produced intermediate results between V1 and V2 — better than V1 on
    specificity (0.902 vs 0.734), PPV (0.708 vs 0.513), F1 (0.739 vs 0.656),
    and accuracy (0.872 vs 0.775), but slightly worse than V2 on all metrics.
    This confirms that the few-shot examples in V2 added meaningful value
    beyond the Sepsis-3 definition alone. V3 also produced 81 "none" likelihood
    ratings, suggesting the definition alone left the model uncertain on many
    borderline cases.

PIPELINE ARCHITECTURE:
    Agent 1 — Infection Agent:
        Identifies infection presence, organism type, body system affected,
        onset time, and treatment agent. (Identical to V1 and V2)
    Agent 2 — Organ Dysfunction Agent:
        Identifies affected organ systems and earliest onset times.
        (Identical to V1 and V2)
    Agent 3 — Sepsis Agent (V3 — Sepsis-3 Definition Only):
        Applies the formal Sepsis-3 SOFA definition and important distinctions
        without any few-shot examples.

INPUT FILES (place in data/ folder):
    - pos_sepsis3_clinical_note.csv     (CSN, Note)
    - neg_sepsis3_clinical_note.csv     (CSN, Note)
    - pos_sepsis3_lab_medication.csv    (CSN, Lab_data, Vital_sign, Medication)
    - neg_sepsis3_lab_medication.csv    (CSN, Lab_data, Vital_sign, Medication)

OUTPUT:
    - results/sepsis_analysis_3/llm_results_v3.jsonl
      One JSON object per line, per patient case.
      Fields: CSN, infection_agent, organ_dysfunction_agent, sepsis_agent

USAGE:
    python sepsis_pipeline_3.py

REQUIREMENTS:
    pip install openai pandas scikit-learn

NOTES:
    - Checkpointing is enabled: if the pipeline is interrupted, rerun the same
      command and it will skip already-completed cases automatically.
    - 4 parallel workers are used to speed up processing.
    - Timeout is set to 600 seconds per API call to handle large clinical notes.
    - No truncation is applied — full notes are sent to the API.
      Mean note length: 189,742 chars. Max: 662,694 chars.
      GPT-5.2 context window: 400,000 tokens — sufficient for all cases.
"""

# V3 — Sepsis-3 definition only, no few-shot examples
import pandas as pd
import json
import time
import os
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── API Setup ──────────────────────────────────────────────────────────────────
MODEL = "gpt-5.2"
client = AzureOpenAI(
    api_key="",
    azure_endpoint="",
    api_version="",
    organization=""
)

# ── Loading data ───────────────────────────────────────────────────────────────
pos_notes = pd.read_csv("data/pos_sepsis3_clinical_note.csv")
neg_notes = pd.read_csv("data/neg_sepsis3_clinical_note.csv")
pos_labs = pd.read_csv("data/pos_sepsis3_lab_medication.csv")
neg_labs = pd.read_csv("data/neg_sepsis3_lab_medication.csv")

all_notes = pd.concat([pos_notes, neg_notes], ignore_index=True)
all_labs = pd.concat([pos_labs, neg_labs], ignore_index=True)

print(f"Total cases: {len(all_notes)}")

# ── Output file ────────────────────────────────────────────────────────────────
OUTPUT_FILE = "results/sepsis_analysis_3/llm_results_v3.jsonl"

completed_csns = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            try:
                completed_csns.add(json.loads(line)["CSN"])
            except:
                pass
print(f"Already completed: {len(completed_csns)} cases")

# ── Prompt templates ───────────────────────────────────────────────────────────
def build_infection_prompt(note, lab_data, vital_sign, medication):
    return f"""
ROLE
You are a clinical expert specializing in the identification of infections. You will be provided with clinical documentation, vital signs data, laboratory measurements, and imaging studies from a patient's entire hospital course. Your task is to review the entirety of the data to determine whether the patient had one or more active infections over the course of their hospitalization.

INSTRUCTIONS
1. Review all details provided in the CLINICAL DATA section below. Based on the data, determine whether one or more infections occurred during the hospitalization. For each infection, output the following information:
- Infectious organism type (e.g., bacterial, viral, or fungal)
- Body/organ system affected (e.g., circulatory, pulmonary, renal, hepatic, neurologic, etc.)
- Estimated earliest time of infection onset. If the patient was admitted with the infection already present, output the admission time as the estimated time of infection.
- Whether the patient was treated for the infection
- If treated, name the treatment agent

Output "No infection was present" if there was no infection present.

2. Provide a brief, 3-5 sentence summary of your clinical reasoning.

3. Your response MUST be output in the following JSON format and nothing else:
{{
  "infection_present": true or false,
  "infections": [
    {{
      "organism_type": "bacterial/viral/fungal/unknown",
      "body_system": "system name",
      "onset_time": "YYYY-MM-DD HH:MM:SS or admission time",
      "treated": true or false,
      "treatment_agent": "medication name or null"
    }}
  ],
  "reasoning": "3-5 sentence clinical justification."
}}

If no infection:
{{
  "infection_present": false,
  "infections": [],
  "reasoning": "3-5 sentence clinical justification."
}}

CLINICAL DATA

Clinical Notes:
{note}

Laboratory Values:
{lab_data}

Vital Signs:
{vital_sign}

Medications:
{medication}
"""

def build_organ_dysfunction_prompt(note, lab_data, vital_sign, medication):
    return f"""
ROLE
You are a clinical expert specializing in the identification of acute organ dysfunction to help with sepsis identification. You will be provided with clinical documentation, vital signs data, and laboratory measurements from a patient's entire hospital course.

INSTRUCTIONS
1. Review all details provided in the CLINICAL DATA section below. For each organ system that is affected, output:
- Organ system (e.g., circulatory, pulmonary, renal, hepatic, neurologic, etc.)
- Estimated earliest time of organ dysfunction onset.

Output "No organ dysfunction was present" if there was no organ dysfunction.

2. Provide a brief, 3-5 sentence summary of your clinical reasoning.

3. Your response MUST be output in the following JSON format and nothing else:
{{
  "organ_dysfunction_present": true or false,
  "organ_systems": [
    {{
      "system": "system name",
      "onset_time": "YYYY-MM-DD HH:MM:SS or admission time"
    }}
  ],
  "reasoning": "3-5 sentence clinical justification."
}}

If no organ dysfunction:
{{
  "organ_dysfunction_present": false,
  "organ_systems": [],
  "reasoning": "3-5 sentence clinical justification."
}}

CLINICAL DATA

Clinical Notes:
{note}

Laboratory Values:
{lab_data}

Vital Signs:
{vital_sign}

Medications:
{medication}
"""

def build_sepsis_prompt(note, lab_data, vital_sign, medication, infection_output, organ_output):
    return f"""
ROLE
You are a clinical expert specializing in the identification of sepsis from the medical record. You are working with two other AI agents — an Infection Agent and an Organ Dysfunction Agent — who have already reviewed the same clinical data.

SEPSIS DEFINITION
Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection, operationalized as a Sequential Organ Failure Assessment (SOFA) score increase of 2 or more points from baseline in the presence of suspected or confirmed infection. Organ dysfunction must be plausibly caused by the infection — not by an alternative non-infectious etiology such as heart failure, chronic kidney disease, medication side effects, or metabolic disorders.

IMPORTANT DISTINCTIONS
- Do NOT label as sepsis if organ dysfunction has a more likely non-infectious cause.
- Do NOT label as sepsis if there is infection but no associated organ dysfunction.
- Do NOT label as sepsis if organ dysfunction preceded the infection onset.
- Only label as DEFINITE if there is no plausible alternative cause for the organ dysfunction.
- Label as PROBABLE if infection is the most likely but not the only cause of organ dysfunction.
- Label as POSSIBLE if an alternative non-infectious cause is equally or more likely.
- Label as UNLIKELY if organ dysfunction is clearly due to a non-infectious cause.

INSTRUCTIONS
1. Review the INFECTION AGENT OUTPUT, ORGAN DYSFUNCTION AGENT OUTPUT, and CLINICAL DATA. Apply the Sepsis-3 definition strictly. Determine whether sepsis was present. Output:
- Suspected infectious source that caused sepsis
- Estimated time of sepsis onset (earliest time of organ dysfunction caused by infection)
- Likelihood rating: definite, probable, possible, or unlikely

2. Provide a brief, 3-5 sentence summary of your reasoning. Explicitly state whether organ dysfunction is plausibly caused by infection or has an alternative explanation.

3. Your response MUST be output in the following JSON format and nothing else:
{{
  "sepsis_present": true or false,
  "infectious_source": "description or null",
  "sepsis_onset_time": "YYYY-MM-DD HH:MM:SS or null",
  "likelihood": "definite/probable/possible/unlikely/none",
  "reasoning": "3-5 sentence clinical justification."
}}

INFECTION AGENT OUTPUT:
{infection_output}

ORGAN DYSFUNCTION AGENT OUTPUT:
{organ_output}

CLINICAL DATA

Clinical Notes:
{note}

Laboratory Values:
{lab_data}

Vital Signs:
{vital_sign}

Medications:
{medication}
"""

# ── Agent functions ────────────────────────────────────────────────────────────
def call_llm(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                timeout=600
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"  JSON parse error on attempt {attempt+1}, retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"  API error on attempt {attempt+1}: {e}")
            time.sleep(10)
    return {"error": "failed after 3 attempts"}


def process_case(row):
    csn = row["CSN"]
    note = str(row["Note"])

    lab_row = all_labs[all_labs["CSN"] == csn]
    if len(lab_row) == 0:
        lab_data = "No lab data available"
        vital_sign = "No vital sign data available"
        medication = "No medication data available"
    else:
        lab_data = str(lab_row.iloc[0]["Lab_data"])
        vital_sign = str(lab_row.iloc[0]["Vital_sign"])
        medication = str(lab_row.iloc[0]["Medication"])

    print(f"  Processing CSN {csn}...")

    infection_output = call_llm(build_infection_prompt(note, lab_data, vital_sign, medication))
    organ_output = call_llm(build_organ_dysfunction_prompt(note, lab_data, vital_sign, medication))
    sepsis_output = call_llm(build_sepsis_prompt(
        note, lab_data, vital_sign, medication,
        json.dumps(infection_output),
        json.dumps(organ_output)
    ))

    return {
        "CSN": csn,
        "infection_agent": infection_output,
        "organ_dysfunction_agent": organ_output,
        "sepsis_agent": sepsis_output
    }


# ── Run pipeline ───────────────────────────────────────────────────────────────
def run_pipeline():
    remaining = all_notes[~all_notes["CSN"].isin(completed_csns)]
    print(f"Cases to process: {len(remaining)}")

    with open(OUTPUT_FILE, "a") as out_file:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_case, row): row["CSN"]
                for _, row in remaining.iterrows()
            }
            for future in as_completed(futures):
                csn = futures[future]
                try:
                    result = future.result()
                    out_file.write(json.dumps(result) + "\n")
                    out_file.flush()
                    print(f"Saved CSN {csn}")
                except Exception as e:
                    print(f"Failed CSN {csn}: {e}")

    print(f"\nDone! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pipeline()
