import os
import json
import re
import time
import ollama

DATASET_DIR = "./dataset"
MODEL = "gemma3:12b"


def generating_thinking(question, answer, confidence):
    prompt = f"""Generate SHORT structured thinking
        for this veterinary pharmacology example.

        Question: {question}
        Answer: {answer}
        Confidence: {confidence}

        Follow EXACTLY this structure:

        Domain check: [in scope or not?]
        Question understanding: [drug, species, aspect]
        Confidence check: [why this confidence level?]
        Confidence level: {confidence.upper()}
        Answer plan: [what to include]

        Return ONLY the thinking text.
        Keep it SHORT.
    """

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response["message"]["content"].strip()

    # Remove markdown fences if model adds them
    raw = re.sub(r"^```.*?\n|\n```$", "", raw, flags=re.DOTALL).strip()

    return raw  


for filename in os.listdir(DATASET_DIR):
    if not filename.endswith(".json"):
        continue

    if filename.startswith("stats") or filename.startswith("dataset"):
        continue

    file_path = os.path.join(DATASET_DIR, filename)

    print(f"Processing {filename}...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])

    for qa in qa_pairs:
        # Skip if already processed
        if "thinking" in qa:
            continue

        qa["thinking"] = generating_thinking(
            qa["question"],
            qa["answer"],
            qa["confidence"],
        )

        time.sleep(0.8)  # prevent overload

    # Save file after processing
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("Done.")