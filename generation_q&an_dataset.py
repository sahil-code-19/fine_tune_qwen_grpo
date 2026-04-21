"""
Plumb's Veterinary Drug Handbook — Tiered QA Generator
=======================================================
Workflow:
  1. Feed raw drug text (from book extraction)
  2. Auto-classify drug into Tier 1–4
  3. Generate section-wise QA pairs scaled to that tier
  4. Augment every question with 5 paraphrase variants
  5. Export final JSONL ready for fine-tuning

Requirements:
    pip install anthropic tqdm
    export ANTHROPIC_API_KEY=your_key_here
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

# import anthropic
import dotenv
import ollama
from tqdm import tqdm

dotenv.load_dotenv()

print("ANTHROPIC_API_KEY========================>", os.getenv("ANTHROPIC_API_KEY"))

# client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "gemma3:12b"

# ─────────────────────────────────────────────────────────────
# TIER DEFINITIONS
# ─────────────────────────────────────────────────────────────


@dataclass
class TierConfig:
    tier: int
    name: str
    description: str
    # How many QA pairs to generate per section
    section_counts: dict
    # How many paraphrase variants per question
    paraphrase_count: int


TIERS = {
    1: TierConfig(
        tier=1,
        name="Simple",
        description="Single species, 1-2 indications, few interactions",
        section_counts={
            "prescriber_highlights": 2,
            "uses_indications": 2,
            "pharmacology": 1,
            "pharmacokinetics": 1,
            "contraindications": 2,
            "adverse_effects": 2,
            "drug_interactions": 2,
            "dosages": 3,
            "overdose_toxicity": 1,
            "storage_forms": 1,
            "client_information": 1,
            "refusals": 2,
        },
        paraphrase_count=4,
    ),
    2: TierConfig(
        tier=2,
        name="Standard",
        description="Multi-species, several indications, moderate interactions",
        section_counts={
            "prescriber_highlights": 3,
            "uses_indications": 3,
            "pharmacology": 2,
            "pharmacokinetics": 2,
            "contraindications": 3,
            "adverse_effects": 4,
            "drug_interactions": 4,
            "dosages": 5,
            "overdose_toxicity": 2,
            "reproductive_safety": 1,
            "laboratory_considerations": 1,
            "storage_forms": 2,
            "client_information": 1,
            "refusals": 3,
        },
        paraphrase_count=5,
    ),
    3: TierConfig(
        tier=3,
        name="Complex",
        description="Many species, many indications, many interactions",
        section_counts={
            "prescriber_highlights": 4,
            "uses_indications": 5,
            "pharmacology": 3,
            "pharmacokinetics": 3,
            "contraindications": 5,
            "adverse_effects": 6,
            "drug_interactions": 7,
            "dosages": 8,
            "overdose_toxicity": 3,
            "reproductive_safety": 2,
            "laboratory_considerations": 2,
            "storage_forms": 2,
            "client_information": 2,
            "refusals": 4,
        },
        paraphrase_count=5,
    ),
    4: TierConfig(
        tier=4,
        name="High-Risk",
        description="Controlled/chemo/narrow therapeutic index drugs",
        section_counts={
            "prescriber_highlights": 5,
            "uses_indications": 5,
            "pharmacology": 4,
            "pharmacokinetics": 4,
            "contraindications": 7,
            "adverse_effects": 8,
            "drug_interactions": 10,
            "dosages": 10,
            "overdose_toxicity": 5,
            "reproductive_safety": 3,
            "laboratory_considerations": 4,
            "storage_forms": 2,
            "client_information": 3,
            "refusals": 5,
        },
        paraphrase_count=6,
    ),
}

# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────


@dataclass
class QAPair:
    drug: str
    tier: int
    section: str
    question_type: str
    question: str
    thinking: str
    answer: str
    confidence: str  # high / medium / low
    species: Optional[str]
    safety_flag: bool
    refusal: bool
    paraphrases: list[str] = field(default_factory=list)

    def to_training_records(self) -> list[dict]:
        """
        Expands one QAPair into N training records:
        - 1 original question
        - N paraphrase variants
        All share the same answer and system prompt.
        """
        system = (
            "You are a veterinary drug reference assistant trained exclusively on "
            "Plumb's Veterinary Drug Handbook. "
            "Rules: "
            "(1) Only answer questions covered in your training data. "
            "(2) For dosage questions always state: species, route, dose range, frequency. "
            "(3) If unsure, say: 'I don't have reliable information on this. "
            "Please consult the full Plumb's handbook or a licensed veterinarian.' "
            "(4) Never guess dosages. "
            "(5) Flag dangerous drug combinations with WARNING:. "
            "(6) End every clinical answer with: "
            "Disclaimer: Always verify with a licensed veterinarian."
        )
        records = []
        for q in [self.question] + self.paraphrases:
            records.append(
                {
                    "system": system,
                    "instruction": q,
                    "input": "",
                    "output": self.answer,
                    "metadata": {
                        "drug": self.drug,
                        "tier": self.tier,
                        "section": self.section,
                        "question_type": self.question_type,
                        "confidence": self.confidence,
                        "species": self.species,
                        "safety_flag": self.safety_flag,
                        "refusal": self.refusal,
                        "is_paraphrase": q != self.question,
                    },
                }
            )
        return records


# ─────────────────────────────────────────────────────────────
# STEP 1 — TIER CLASSIFIER
# ─────────────────────────────────────────────────────────────

TIER_CLASSIFIER_PROMPT = """
You are classifying a veterinary drug for fine-tuning dataset generation.

Read the drug text below and classify it into exactly one tier:

TIER 1 — Simple:
- Covers 1-2 species
- 1-2 clinical indications
- Few drug interactions (≤3)
- No controlled substance / narrow therapeutic index

TIER 2 — Standard:
- Covers 2-3 species  
- 2-4 clinical indications
- Moderate interactions (4-6)
- No major safety concerns beyond standard precautions

TIER 3 — Complex:
- Covers 3+ species
- 4+ clinical indications OR widely used drug
- Many interactions (7+)
- Multiple dosing routes or complex dosing protocols

TIER 4 — High-Risk:
- Controlled substance OR chemotherapy agent OR narrow therapeutic index
- Requires close monitoring (TDM, frequent labs)
- Severe or life-threatening adverse effects possible
- Many serious drug interactions

Drug text:
{drug_text}

Respond with ONLY a JSON object, no other text:
{{
  "tier": <1, 2, 3, or 4>,
  "reason": "<one sentence explanation>",
  "species_covered": ["list", "of", "species"],
  "indication_count": <number>,
  "interaction_count": <number>,
  "is_controlled": <true/false>,
  "is_narrow_therapeutic_index": <true/false>
}}
"""


def classify_drug_tier(drug_name: str, drug_text: str) -> dict:
    """Calls LLM to classify the drug into a tier."""
    print(f"  [Tier classifier] Classifying {drug_name}...")
    # response = client.messages.create(
    #     model=MODEL,
    #     max_tokens=300,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": TIER_CLASSIFIER_PROMPT.format(drug_text=drug_text[:3000]),
    #         }
    #     ],
    # )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": TIER_CLASSIFIER_PROMPT.format(drug_text=drug_text[:3000]),
            }
        ],
    )
    raw = response.message.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    result = json.loads(raw)
    result["drug"] = drug_name
    return result


# ─────────────────────────────────────────────────────────────
# STEP 2 — QA PAIR GENERATOR
# ─────────────────────────────────────────────────────────────

QA_GENERATION_PROMPT = """
You are building a high-quality fine-tuning dataset from Plumb's Veterinary Drug Handbook.

## Context
Drug: {drug_name}
Tier: {tier} ({tier_name}) — {tier_description}
Section: {section}
QA pairs needed: {count}

## Drug Text
{drug_text}

---

## CRITICAL: REDUNDANCY & COMPREHENSIVENESS RULES

### ⛔ DO NOT GENERATE REDUNDANT QUESTIONS
Avoid asking the same core content twice, even with different wording. Examples of redundancy to PREVENT:
- Multiple "What is the primary indication?" questions (ask ONCE, then move to different angles)
- Repeating the same mechanism explanation (ask it once, then focus on practical implications)
- Duplicating concentration/safety warnings (emphasize once as CRITICAL, not repeatedly)

### ✅ DO ENSURE COMPREHENSIVE COVERAGE within this section
For each section assigned, generate DIVERSE question angles:
- Indications/Uses: Ask about primary use (1 pair), then secondary uses (1 pair), then practical scenarios (others)
- Dosages: Ask cattle dose separately from horse dose separately from sheep dose (NOT all in one Q)
- Drug Interactions: Ask about SPECIFIC drugs/classes if mentioned in text (not generic)
- Chemistry/Storage: Include formulation percentages, storage conditions, stability if in text
- Pharmacology: Ask mechanism (1 pair), then clinical implications (1 pair), then monitoring (1 pair)

---

## Question Type Definitions
Generate a MIX of these types:
- factual          → Direct recall: "What is the mechanism of action of X?"
- clinical_scenario → Reasoning required: "A dog with renal failure needs pain management — is X appropriate?"
- species_specific  → ONE species per question. Ask separately for dog, cat, horse, etc.
- monitoring       → "What adverse effects or lab values should be monitored?"
- comparison       → "Why choose X over Y for Z condition?"
- practical        → Administration route, storage, compounding, handling, product sourcing
- client_education → Plain language a pet owner would understand (no jargon)
- refusal          → Questions that CANNOT be answered from the text (wrong species, off-label not covered, out of scope)

---

## Answer Writing Rules

### For ALL answers:
- Write in complete, grammatically correct sentences.
- Be EXPLANATORY: don't just state a fact — briefly explain WHY it matters clinically.
  BAD:  "The dose is 5 mg/kg."
  GOOD: "The recommended dose is 5 mg/kg because lower doses may be subtherapeutic, while higher doses increase the risk of hepatotoxicity."
- Ground every claim STRICTLY in the drug text above. Never invent or extrapolate.
- Use plain, consistent English. Avoid Unicode special characters (use 'mcg' not 'μg', 'degrees F' not '°F').
- Double-check drug and product names for spelling accuracy (e.g., "Acetic acid" not "Acarbose")

### For dosage answers:
ALWAYS include ALL of: species | route | dose range | frequency | any relevant duration or titration notes.
If multiple species exist in text, create SEPARATE questions for each species rather than one combined question.

### For drug interaction answers:
If References mentions specific drug interactions, ask about EACH one separately (not nested).
Example from text: "Can interact with Aspirin, Azole antifungals, Iron, Phenobarbital, Quinidine"
→ Generate ~5 pairs, one asking about Aspirin, one about Azole antifungals, etc. (NOT one pair covering all)

### For chemistry/formulation answers:
If text mentions formulation details (concentrations, USP/NF specs, dilution instructions), include them.
Example: "Acetic Acid USP is 36-37%, Diluted NF is 5.7-6.3%, Vinegar is 3-5%"
→ Generate pairs asking about each formulation type

### For storage/regulatory answers:
If text mentions storage conditions, regulatory status, or product availability, generate practical Q&A.
Example: "Store in airtight containers" or "No veterinary-labeled products available"

### For safety-critical answers:
Begin the answer with: WARNING:

### For refusal questions:
The answer must express that information is unavailable 
and redirect appropriately. Vary the wording naturally 
based on WHY the refusal is happening.

Examples of valid refusal answers:

# Data gap — species not covered:
"Specific dosing data for this species is not available 
in this drug entry. Species-specific pharmacokinetics 
can differ significantly — please consult a veterinarian 
experienced with this species."

# Data gap — dose not listed:
"A specific dose for this indication is not provided 
in the available reference for this species. Please 
consult a licensed veterinarian for current dosing 
guidance."

# Off-label not covered:
"This use is not documented in the available drug 
reference for this species. Please consult a 
veterinarian before attempting this application."

Rules for refusal answers:
- Never invent or extrapolate clinical information
- Always explain WHY you cannot answer specifically
- Always redirect to appropriate resource
- Vary the wording — do not repeat identical phrases
- Keep it 1-3 sentences maximum
- Use "general" when the answer applies across multiple 
    species with no species-specific nuance. Use null only 
    for practical, storage, or compounding questions with 
    no clinical species context at all.

---

## JSON Field Constraints — follow these EXACTLY

- "species": must be ONE of these exact strings only:
    "dog" | "cat" | "horse" | "bird" | "rabbit" | "small_mammal" | "general" | null
  → If a question covers multiple species, SPLIT into separate QA objects (one per species).
  → Use null only for practical/storage/compounding questions with no species context.

- "question_type": must be ONE of:
    "factual" | "clinical_scenario" | "species_specific" | "monitoring" |
    "comparison" | "practical" | "client_education" | "refusal"

- "confidence": must be ONE of: "high" | "medium" | "low"
  → Use "high" when the drug text directly and completely supports the answer.
  → Use "medium" when the mechanism is plausible but clinical evidence is limited 
     or the source only partially covers the answer.
  → Use "low" when the source is incomplete, contradictory, or the answer 
     requires extrapolation beyond what the text provides.

- "safety_flag": boolean — true if the answer involves toxicity, contraindications, overdose, narrow therapeutic index, or any WARNING.

- "refusal": boolean — true only for refusal-type questions.

---

## CONFIDENCE DISTRIBUTION REQUIREMENT
You MUST follow this approximate distribution per batch:
- "high"   → 60% of pairs
- "medium" → 30% of pairs  
- "low"    → 10% of pairs

Before assigning "high" ask:
- Is this DIRECTLY stated in the drug text?
- Is there NO species extrapolation required?
- Are there NO missing data points?
If any answer is NO → use "medium" or "low"

---

## REFUSAL TYPE DEFINITION

Only ONE type of refusal in this prompt:

IN-DOMAIN REFUSAL (data gap):
The question IS about veterinary pharmacology 
but the drug text does not contain enough 
information to answer.

Examples:
- Dose not listed for this species in the text
- Drug interaction mentioned but not explained
- Off-label use not covered in this entry
- Exotic species where drug entry has no data

→ confidence: "low"
→ refusal: true
→ thinking: what source does NOT contain → 
            why extrapolation unsafe → 
            refusal is correct action

---

## THINKING FIELD RULES (CRITICAL)

- "thinking": internal reasoning trace before answering.
  Style depends on question type:

  NORMAL ANSWER (refusal=false):
  Show reasoning as flowing sentences — no headers or labels.
  Pattern: mechanism → evidence → caveat → confidence line
  End with one line: "Confidence: HIGH/MEDIUM/LOW — [derived reason]"
  
  Length by question type:
  - factual, client_education, practical → 2-3 sentences + confidence line
  - monitoring, species_specific         → 3-4 sentences + confidence line
  - clinical_scenario, safety_critical   → 4-6 sentences + confidence line
  - comparison → 4-5 sentences + confidence line
  - refusal    → 2-4 sentences + confidence line (already in refusal section)

  REFUSAL (refusal=true, in-domain but no data):
  Pattern: what source does NOT contain → why extrapolation unsafe → 
           explicit refusal statement
  End with: "Confidence: LOW — source does not contain this. 
             Refusal appropriate. Do not extrapolate."
  Length: 2-4 sentences + confidence line

  Examples:

  HIGH (factual, horse, HYPP):
  "Acetazolamide inhibits carbonic anhydrase promoting renal excretion 
  of potassium and bicarbonate. HYPP in horses causes hyperkalemia-induced 
  muscle depolarization — directly addressed by this mechanism. Equine-specific 
  clinical use is well established in Quarter Horse lineages. Note this manages 
  episodes only, not the underlying genetic defect — dietary potassium 
  restriction is also required.
  Confidence: HIGH — direct source coverage, established equine mechanism, 
  no conflicting evidence. Caveat retained: management only, not curative."

  MEDIUM (dosing frequency, horse):
  "Acetazolamide PK parameters in horses are documented — oral bioavailability, 
  clearance, and MRT values exist from published equine studies. However specific 
  dosing intervals for HYPP management are not standardized in the source. The PK 
  data supports frequent dosing but does not prescribe an interval.
  Confidence: MEDIUM — PK foundation solid, clinical dosing protocol 
  not directly stated. Answer the known part, do not specify interval."

  LOW (oral dose not listed):
  "The source lists acetazolamide for HYPP in horses but does not provide 
  a specific oral dose, route, or frequency. No dose range can be derived 
  from the available entry. Inferring from human or canine data would be 
  inappropriate — equine pharmacokinetics differ significantly and dosing 
  errors carry serious consequences.
  Confidence: LOW — requested dosing data absent from source. 
  Refusal appropriate. Do not estimate or extrapolate."


  Rules:
  - No headers, labels, or bullet points
  - No mention of any handbook or source by name
  - Confidence level derived at end — never declared upfront
  - All question types end with a confidence line
  - Never pad LOW or OOD with clinical factors not in source

---

## SPECIAL EMPHASIS FOR THIS SECTION: {section}

For "{section}" section specifically:
- Identify ALL factual claims, dosages, interactions, or safety notes in the drug text
- Generate questions that cover each of these comprehensively
- Avoid asking the same question twice
- Ensure each pair brings NEW clinical or practical information

--- 

## FEW-SHOT EXAMPLES
Study these carefully before generating.
Notice:
- Thinking DERIVES confidence at the end — never declares it upfront
- Thinking explicitly references what was found or not found in the provided drug text
- Confidence line always last — derived from reasoning above it
- Refusal thinking shows active restraint — names exactly what is missing


[
    {
        "question": "In horses, what condition is acetazolamide used to prevent or treat?",
        "thinking": "Acetazolamide inhibits carbonic anhydrase promoting renal excretion of potassium and bicarbonate. HYPP in horses causes hyperkalemia-induced muscle depolarization — directly addressed by this mechanism. Equine-specific clinical use for HYPP prevention and treatment is well established in Quarter Horse lineages carrying the Impressive bloodline mutation. Note this manages episodes only, not the underlying genetic defect — dietary potassium restriction is also required alongside drug therapy. Confidence: HIGH — indication well established, mechanism clear, equine-specific evidence present, no conflicting data found. Caveat retained: management only, not curative.",
        "answer": "Acetazolamide is used as a preventive and/or treatment for hyperkalemic periodic paralysis (HYPP) in horses. HYPP is a genetic muscle disorder that causes episodes of muscle weakness or paralysis, and acetazolamide can help manage these episodes.",
        "question_type": "factual",
        "confidence": "high",
        "species": "horse",
        "safety_flag": false,
        "refusal": false
    },
    {
        "question": "What is the primary reason Benoxinate-Fluorescein Ophthalmic is used in horses?",
        "thinking": "Benoxinate is a corneal anesthetic and fluorescein a diagnostic staining agent — their combined ophthalmic use is well characterized in equine practice. Equine diagnostic procedures requiring corneal anesthesia include tonometry, ultrasonography, and gonioscopy — all directly supported by this drug combination. Benoxinate efficacy is documented as comparable to proparacaine, a standard equine ophthalmic anesthetic — this comparison is a strong confidence signal confirming established clinical use. No species extrapolation required — equine-specific diagnostic use is directly established. Confidence: HIGH — dual mechanism established, equine diagnostic use directly confirmed, efficacy comparison to proparacaine present. No gaps or conflicting evidence found.",
        "answer": "Benoxinate-Fluorescein Ophthalmic is primarily used in horses for ophthalmic diagnostic procedures, such as tonometry, ocular ultrasonography, and gonioscopy. It provides corneal anesthesia, enabling these examinations to be performed with reduced discomfort to the horse. Its anesthetic effect is comparable to that achieved by topical proparacaine.",
        "question_type": "factual",
        "confidence": "high",
        "species": "horse",
        "safety_flag": false,
        "refusal": false
    },
    {
        "question": "If a veterinarian is considering acetazolamide for a dog with syringomyelia and increased CSF pressures, what is the documented evidence of its effectiveness?",
        "thinking": "Acetazolamide inhibits carbonic anhydrase reducing CSF production — the mechanism theoretically supports use in elevated CSF pressure conditions. However syringomyelia involves complex spinal fluid dynamics including obstruction and syrinx formation beyond simple CSF volume reduction, and the pathophysiology may not respond to pressure reduction alone. Clinical studies in dogs showed acetazolamide was ineffective in reducing clinical signs or ventricle to brain ratio in canine syringomyelia — the mechanistic rationale exists but clinical evidence does not support this specific indication. Confidence: MEDIUM — mechanism plausible, but clinical evidence for this indication in dogs is negative. Answer should present both the theoretical basis and the contradicting clinical findings.",
        "answer": "While acetazolamide has been used as an adjunctive treatment for increased CSF pressures associated with syringomyelia in dogs, studies have shown it was ineffective in reducing clinical signs or the ventricle:brain ratio. Therefore, its use in this scenario may not be beneficial.",
        "question_type": "factual",
        "confidence": "medium",
        "species": "dog",
        "safety_flag": false,
        "refusal": false
    },
    {
        "question": "What is a potential, rare, adverse reaction that may manifest when a patient sensitive to other local anesthetics receives benoxinate ophthalmic?",
        "thinking": "Benoxinate shares structural similarities with other local anesthetics making cross-sensitivity pharmacologically plausible. This reaction is recognized as rare and described as pseudoallergic mimicking an IgE-mediated event — mechanism is theoretically consistent with mast cell degranulation, though the specific pathway in veterinary patients is not well documented. The reaction is flagged as a recognized risk without robust frequency or outcome data specific to veterinary species. Safety flag is warranted independently of confidence level — anaphylactoid potential carries clinical significance regardless of how rarely it occurs. Confidence: MEDIUM — reaction recognized as rare, mechanistic basis theoretically consistent but veterinary-specific frequency and clinical evidence limited. Safety flag appropriate given anaphylactoid potential.",
        "answer": "WARNING: Although rare, benoxinate ophthalmic may cause a pseudoallergic (anaphylactoid) reaction mimicking an IgE-mediated vasodilatory event in patients sensitive to other local anesthetics. This reaction should be recognized and treated promptly.",
        "question_type": "monitoring",
        "confidence": "medium",
        "species": null,
        "safety_flag": true,
        "refusal": false
    },
    {
        "question": "What impact might acetazolamide have on a patient with a history of pulmonary obstruction?",
        "thinking": "Acetazolamide inhibits carbonic anhydrase causing metabolic acidosis through bicarbonate excretion. In patients with pulmonary obstruction, respiratory compensation for acidosis is already compromised — adding metabolic acidosis could worsen hypoxia and increase respiratory distress. This combination is established as a contraindication, however specific mechanistic detail and clinical outcome data for this scenario are incomplete — the risk is flagged without full pharmacological explanation available. The safety concern is grounded in physiological reasoning but the absence of complete mechanistic detail prevents a fully confident answer. Confidence: LOW — contraindication established but mechanistic detail and clinical evidence incomplete. Answer must flag contraindication clearly without overstating detail that is not available.",
        "answer": "WARNING: Acetazolamide is contraindicated in patients with significant pulmonary obstruction or respiratory compromise. The exact mechanism of harm in this context is not fully detailed in available references, but the risk of exacerbating respiratory compromise is sufficient to contraindicate its use. Please consult a licensed veterinarian before administering.",
        "question_type": "factual",
        "confidence": "low",
        "species": "general",
        "safety_flag": true,
        "refusal": false
    },
    {
        "question": "Can Benoxinate-Fluorescein Ophthalmic be administered to a small breed dog weighing 2 pounds for a minor ocular procedure?",
        "thinking": "Benoxinate-Fluorescein ophthalmic use in dogs is established for diagnostic procedures. However weight-specific dosing guidelines or safety thresholds for very small or underweight canine patients are not available — pharmacokinetics in a 2 pound dog may differ significantly from standard canine dosing assumptions. Systemic absorption from ophthalmic application could carry disproportionate risk at this body weight, and no data exists to answer this safely for this patient size. Confidence: LOW — canine ophthalmic use established but weight-specific safety data absent. Refusal appropriate. Do not extrapolate standard dosing to this patient size.",
        "answer": "I don't have reliable information on this specific weight-based dosing scenario for this drug. Please consult a licensed veterinarian experienced with very small breed patients for individualized dosing and safety guidance.",
        "question_type": "clinical_scenario",
        "confidence": "low",
        "species": "dog",
        "safety_flag": false,
        "refusal": true
    }
]

---

## Output Format
Respond with ONLY a valid JSON array. No preamble, no markdown fences, no trailing text.
Ensure all strings use ASCII characters only (no Unicode escapes, no special symbols).

[
  {{
    "question": "<the question>",
    "thinking": "<internal reasoning trace before answering>",
    "answer": "<full explanatory answer>",
    "question_type": "<one of the allowed types>",
    "confidence": "<high|medium|low>",
    "species": "<one allowed species string or null>",
    "safety_flag": <true|false>,
    "refusal": <true|false>
  }}
]
"""


def generate_qa_pairs(
    drug_name: str,
    drug_text: str,
    tier_config: TierConfig,
    section: str,
    count: int,
) -> list[dict]:
    """Generates QA pairs for one section of one drug."""
    if count == 0:
        return []

    # response = client.messages.create(
    #     model=MODEL,
    #     max_tokens=4000,
    #     messages=[{
    #         "role": "user",
    #         "content": QA_GENERATION_PROMPT.format(
    #             drug_name=drug_name,
    #             tier=tier_config.tier,
    #             tier_name=tier_config.name,
    #             tier_description=tier_config.description,
    #             drug_text=drug_text[:4000],
    #             section=section.replace("_", " ").title(),
    #             count=count,
    #         )
    #     }]
    # )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": QA_GENERATION_PROMPT.format(
                    drug_name=drug_name,
                    tier=tier_config.tier,
                    tier_name=tier_config.name,
                    tier_description=tier_config.description,
                    drug_text=drug_text,
                    section=section.replace("_", " ").title(),
                    count=count,
                ),
            }
        ],
    )

    raw = response.message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    pairs = json.loads(raw)
    return pairs[:count]  # Safety cap


# ─────────────────────────────────────────────────────────────
# STEP 3 — PARAPHRASE AUGMENTER
# ─────────────────────────────────────────────────────────────

PARAPHRASE_PROMPT = """
You are augmenting a veterinary fine-tuning dataset.

Original question about the drug "{drug_name}":
"{question}"

Generate exactly {count} alternative ways someone might ask this same question.
Vary the style realistically:
- Some formal (veterinarian style)
- Some casual (pet owner style)  
- Some very short ("acarbose with food?")
- Some with species embedded ("for my cat, when...")
- Some with slight misspellings or abbreviations
- Some phrased as a statement seeking confirmation

Critical rules:
- Do NOT change the meaning or introduce new clinical content
- Do NOT make the question unanswerable with the original answer
- Keep all questions relevant to the original intent

Respond with ONLY a JSON array of strings, no other text:
["variant 1", "variant 2", ...]
"""


def generate_paraphrases(
    drug_name: str,
    question: str,
    count: int,
) -> list[str]:
    """Generates paraphrase variants for a single question."""
    # response = client.messages.create(
    #     model=MODEL,
    #     max_tokens=800,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": PARAPHRASE_PROMPT.format(
    #                 drug_name=drug_name,
    #                 question=question,
    #                 count=count,
    #             ),
    #         }
    #     ],
    # )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PARAPHRASE_PROMPT.format(
                    drug_name=drug_name,
                    question=question,
                    count=count,
                ),
            }
        ],
    )

    # raw = response.content[0].text.strip()
    raw = response.message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    variants = json.loads(raw)
    return variants[:count]


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────


def process_drug(drug_name: str, drug_text: str) -> dict:
    """
    Full pipeline for one drug:
      classify → generate QA per section → paraphrase augment → return records
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {drug_name}")
    print(f"{'=' * 60}")

    # ── Step 1: Classify tier ──────────────────────────────
    classification = classify_drug_tier(drug_name, drug_text)
    tier_num = classification["tier"]
    tier_config = TIERS[tier_num]
    print(f"  → Tier {tier_num} ({tier_config.name}): {classification['reason']}")

    all_qa_pairs: list[QAPair] = []

    # ── Step 2: Generate QA pairs section by section ──────
    sections = tier_config.section_counts
    for section, count in sections.items():
        print(f"  [QA Gen] {section} → {count} pairs...")
        try:
            raw_pairs = generate_qa_pairs(
                drug_name=drug_name,
                drug_text=drug_text,
                tier_config=tier_config,
                section=section,
                count=count,
            )
            for p in raw_pairs:
                all_qa_pairs.append(
                    QAPair(
                        drug=drug_name,
                        tier=tier_num,
                        section=section,
                        question_type=p.get("question_type", "factual"),
                        question=p["question"],
                        answer=p["answer"],
                        confidence=p.get("confidence", "high"),
                        species=p.get("species"),
                        safety_flag=p.get("safety_flag", False),
                        refusal=p.get("refusal", False),
                        thinking=p.get("thinking", ""),
                    )
                )
            time.sleep(0.5)  # Rate limit courtesy
        except Exception as e:
            print(f"    ⚠ Error in section {section}: {e}")
            continue

    # ── Step 3: Paraphrase augmentation ──────────────────
    print(
        f"  [Paraphrase] Augmenting {len(all_qa_pairs)} pairs "
        f"with {tier_config.paraphrase_count} variants each..."
    )

    for qa in tqdm(all_qa_pairs, desc="  Augmenting", leave=False):
        if qa.refusal:
            # Refusal questions need fewer variants — 2 is enough
            n = min(2, tier_config.paraphrase_count)
        else:
            n = tier_config.paraphrase_count
        try:
            qa.paraphrases = generate_paraphrases(
                drug_name=drug_name,
                question=qa.question,
                count=n,
            )
            time.sleep(0.3)
        except Exception as e:
            print(f"    ⚠ Paraphrase error: {e}")
            qa.paraphrases = []

    # ── Step 4: Expand to training records ───────────────
    # training_records = []
    # for qa in all_qa_pairs:
    #     training_records.extend(qa.to_training_records())

    result = {
        "drug": drug_name,
        "tier": tier_num,
        "tier_name": tier_config.name,
        "classification": classification,
        "qa_pair_count": len(all_qa_pairs),
        # "training_record_count": len(training_records),
        "qa_pairs": [asdict(qa) for qa in all_qa_pairs],
        # "training_records": training_records,
    }

    print(
        f"  ✓ {len(all_qa_pairs)} QA pairs → "
        # f"{len(training_records)} training records (with paraphrases)"
    )
    return result


# ─────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────


def run_batch(
    drugs: list[dict],  # [{"name": "Acarbose", "text": "...full book text..."}]
    output_dir: str = "output",
) -> None:
    """
    Processes a list of drugs and writes:
      - output/<drug_name>.json       (full detail per drug)
      - output/dataset.jsonl          (flat training records, all drugs)
      - output/stats.json             (summary statistics)
    """
    os.makedirs(output_dir, exist_ok=True)
    all_training_records = []
    stats = {
        "total_drugs": len(drugs),
        "by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
        "total_qa_pairs": 0,
        "total_training_records": 0,
        "safety_flagged": 0,
        "refusals": 0,
        "errors": [],
    }

    for drug in tqdm(drugs, desc="Processing drugs"):
        try:
            output_file_name = f"{drug['name'].lower().replace(' ', '_')}-{MODEL.replace(':', '_')}.json"

            if os.path.exists(os.path.join(output_dir, output_file_name)):
                print(f"  → Skipping {drug['name']} (already processed)")
                continue

            result = process_drug(drug["name"], drug["text"])

            # Per-drug JSON
            drug_path = os.path.join(
                output_dir,
                output_file_name,
            )
            with open(drug_path, "w") as f:
                json.dump(result, f, indent=2)

            # Accumulate
            # all_training_records.extend(result["training_records"])
            stats["by_tier"][result["tier"]] += 1
            stats["total_qa_pairs"] += result["qa_pair_count"]
            stats["total_training_records"] += result["training_record_count"]
            stats["safety_flagged"] += sum(
                1 for qa in result["qa_pairs"] if qa["safety_flag"]
            )
            stats["refusals"] += sum(1 for qa in result["qa_pairs"] if qa["refusal"])

        except Exception as e:
            print(f"  ✗ Failed {drug['name']}: {e}")
            stats["errors"].append({"drug": drug["name"], "error": str(e)})

    # Write flat JSONL (one training record per line — standard fine-tune format)
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    with open(jsonl_path, "w") as f:
        for record in all_training_records:
            f.write(json.dumps(record) + "\n")

    # Write stats
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("BATCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Drugs processed    : {stats['total_drugs'] - len(stats['errors'])}")
    print(f"  Total QA pairs     : {stats['total_qa_pairs']}")
    print(
        f"  Training records   : {stats['total_training_records']} (incl. paraphrases)"
    )
    print(f"  Safety flagged     : {stats['safety_flagged']}")
    print(f"  Refusal examples   : {stats['refusals']}")
    print(f"  Errors             : {len(stats['errors'])}")
    print(f"\n  By tier:")
    for t, count in stats["by_tier"].items():
        print(f"    Tier {t} ({TIERS[t].name:<10}): {count} drugs")
    print(f"\n  Output: {jsonl_path}")


# ─────────────────────────────────────────────────────────────
# JSON LOADER  — reads plumbs_details.json
# ─────────────────────────────────────────────────────────────

PLUMBS_JSON = os.path.join(os.path.dirname(__file__), "plumbs_details.json")


def load_drugs_from_json(json_path: str) -> list[dict]:
    """
    Reads plumbs_details.json and returns a list of
    {"name": <title>, "text": <all sections concatenated>} dicts.
    Only entries with showMonograph=True are included.
    Extracts exactly 80 drugs (or fewer if fewer valid drugs exist).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Filter for showMonograph=True and valid titles FIRST
    valid_entries = []
    for entry in entries:
        if not entry.get("showMonograph", False):
            continue
        name = entry.get("title", "").strip()
        if not name:
            continue
        valid_entries.append(entry)

    # Then take the first 80 valid entries
    new_entries = valid_entries[:80]

    length = len(json.dumps(new_entries))
    print(f"Loaded {len(new_entries)} drugs from JSON (total {length} chars)")
    print(
        f"  (Filtered from {len(entries)} total entries, {len(valid_entries)} valid entries)"
    )

    drugs = []
    for entry in new_entries:
        name = entry.get("title", "").strip().replace("/", "-")

        # Build a readable text blob from all sections
        parts = [f"{name}"]
        drug_class = entry.get("drugClass", [])
        if drug_class:
            parts.append(f"Drug Class: {', '.join(drug_class)}")
        species = entry.get("species", [])
        if species:
            parts.append(f"Species: {', '.join(species)}")
        commercial = entry.get("commercialNames", [])
        if commercial:
            parts.append(f"Commercial Names: {', '.join(commercial)}")

        for section in entry.get("sections", []):
            title = section.get("title", "").strip()
            value = section.get("value", "").strip()
            if title and value:
                parts.append(f"\n{title}:\n{value}")

        print(f"  Built text for {name}.\n")
        drugs.append({"name": name, "text": "\n".join(parts)})

    return drugs


def find_drug(drugs: list[dict], name: str) -> dict | None:
    """Case-insensitive search for a drug by name (or alias prefix)."""
    name_lower = name.strip().lower()
    for d in drugs:
        if d["name"].lower() == name_lower:
            return d
    # Partial / prefix match as fallback
    for d in drugs:
        if d["name"].lower().startswith(name_lower):
            return d
    return None


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    os.makedirs("output", exist_ok=True)

    # Load all drugs from plumbs_details.json
    print(f"Loading drugs from {PLUMBS_JSON} …")
    all_drugs = load_drugs_from_json(PLUMBS_JSON)
    print(f"  Loaded {len(all_drugs)} monographs.\n")

    # ── Single-drug mode: python main.py "Acarbose" ────────
    if len(sys.argv) >= 2:
        drug_name_arg = " ".join(sys.argv[1:])
        drug = find_drug(all_drugs, drug_name_arg)

        if drug is None:
            print(f"❌ Drug '{drug_name_arg}' not found in plumbs_details.json.")
            print("   Available drugs (first 20):")
            for d in all_drugs[:20]:
                print(f"     • {d['name']}")
            sys.exit(1)

        assert drug is not None  # type narrowing for static checkers
        print(f"▶ Single-drug mode: {drug['name']}")
        result = process_drug(drug["name"], drug["text"])

        safe_name = drug["name"].lower().replace(" ", "_").replace("/", "_")
        full_path = os.path.join("output", f"{safe_name}_full.json")
        jsonl_path = os.path.join("output", f"{safe_name}_training.jsonl")

        with open(full_path, "w") as f:
            json.dump(result, f, indent=2)

        # with open(jsonl_path, "w") as f:
        #     for record in result["training_records"]:
        #         f.write(json.dumps(record) + "\n")

        print(f"\n✅ Done!")
        print(f"   Drug          : {result['drug']}")
        print(f"   Tier          : {result['tier']} ({result['tier_name']})")
        print(f"   QA pairs      : {result['qa_pair_count']}")
        # print(
        #     f"   Training recs : {result['training_record_count']} (with paraphrases)"
        # )
        print(f"   Full JSON     : {full_path}")
        # print(f"   Training JSONL: {jsonl_path}")

    # ── Batch mode: python main.py ─────────────────────────
    else:
        print("▶ Batch mode: processing all drugs from plumbs_details.json")
        run_batch(all_drugs, output_dir="output_v2")
