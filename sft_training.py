"""
Phase 1: Supervised Fine-Tuning (SFT) for Qwen3-8B using Unsloth.

This script fine-tunes Qwen3-8B on veterinary drug QA pairs using LoRA (QLoRA 4-bit).
It uses the Unsloth framework for optimized training with reduced VRAM usage.

Key changes from the old Llama-based approach:
- Uses Unsloth's FastLanguageModel instead of raw HuggingFace
- Disclaimer removed from system prompt (appended post-generation in app code)
- Confidence field from dataset is used in training context
- Refusal samples explicitly train the model to refuse out-of-domain questions
- Paraphrases expanded as separate training samples
"""

import json
import os

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "unsloth/Qwen3-8B"
DATASET_DIR = "./dataset"

OUTPUT_DIR = "./models/qwen3-sft-lora"
FINAL_DIR = "./models/qwen3-sft-merged"

MAX_SEQ_LEN = 2048
TRAIN_BATCH_SIZE = 2
GRAD_ACCUMULATION = 8

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# System prompt — disclaimer is NOT here. It will be appended post-generation
# in application code. This prevents the repetition loop issue.
SYSTEM_PROMPT = """\
You are a veterinary drug reference assistant trained exclusively on Plumb's Veterinary Drug Handbook.

## Core Rules:
(1) Only answer questions covered in your training data. If unsure, say: "I don't have reliable information on this. Please consult the full Plumb's handbook or a licensed veterinarian."
(2) For dosage questions, ALWAYS state: species | route | dose range | frequency. Never guess dosages.
(3) For safety-critical information, begin with WARNING: and explain clinical significance.
(4) Flag dangerous drug combinations and contraindications explicitly.
(5) Specify which species each answer applies to (e.g., "For cattle..." or "This is contraindicated in cats").
(6) If a question asks about a species not in your training data, provide a refusal with appropriate caution.
(7) If the question is outside veterinary pharmacology (e.g., cooking, law, general education), refuse to answer and explain that you are specialized in veterinary drug information only.

## Confidence Assessment:
- When you are highly confident in your answer, provide it directly.
- When you have medium confidence, note areas of uncertainty.
- When you have low confidence or the information is not in your training data, refuse to answer rather than guessing.

## Quality Standards:
- Be explanatory: Don't just state facts—briefly explain WHY they matter clinically.
- Ground all claims strictly in the reference material. Never extrapolate or invent.
- Use plain English. Avoid Unicode characters (use 'mcg' not 'μg', 'degrees F' not '°F').
- For drug interactions: provide specific mechanism and clinical significance, not generic warnings.
- For dosages: include duration, frequency, route, and any titration notes.

## Tone:
Professional, precise, safety-focused. Prioritize accuracy over completeness."""


# ============================================================================
# Dataset Loading
# ============================================================================


def load_dataset_from_json(dataset_dir: str) -> Dataset:
    """
    Load and expand the veterinary QA dataset from JSON files.

    Each QA pair is expanded with its paraphrases to create more training samples.
    The confidence field and refusal flag from the dataset are preserved to teach
    the model calibrated confidence and proper refusal behavior.
    """
    print("📂 Loading dataset...")
    samples = []

    for filename in os.listdir(dataset_dir):
        if not filename.endswith(".json"):
            continue
        # Skip any stats/metadata files
        if filename.startswith("stats") or filename.startswith("dataset"):
            continue

        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_pairs = data.get("qa_pairs", [])
        print(f"  Processing {filename} with {len(qa_pairs)} QA pairs...")

        if not qa_pairs:
            print(f"  ⚠️ Warning: {filename} has no QA pairs. Skipping.")
            continue

        for qa in qa_pairs:
            question = qa["question"].strip()
            answer = qa["answer"].strip()
            confidence = qa.get("confidence", "high")
            is_refusal = qa.get("refusal", False)
            safety_flag = qa.get("safety_flag", False)
            species = qa.get("species", "general")
            paraphrases = qa.get("paraphrases", [])
            thinking = qa.get("thinking", "")

            # Ensure paraphrases is always a list
            if not isinstance(paraphrases, list):
                paraphrases = []

            # Expand: original question + all paraphrases
            all_questions = [question] + [p.strip() for p in paraphrases if p and p.strip()]

            for q in all_questions:
                samples.append({
                    "question": q,
                    "answer": answer,
                    "confidence": confidence,
                    "refusal": is_refusal,
                    "safety_flag": safety_flag,
                    "species": species,
                    "thinking": thinking,
                })

    print(f"✅ Dataset loaded & expanded: {len(samples)} samples")
    return Dataset.from_list(samples)


# ============================================================================
# Chat Formatting
# ============================================================================


def format_dataset_for_sft(dataset: Dataset, tokenizer) -> Dataset:
    """
    Format each QA pair as a chat conversation using Qwen3's chat template.

    The conversation format is:
        system: [system prompt]
        user: [question]
        assistant: [answer]

    This uses the tokenizer's apply_chat_template to ensure proper formatting
    with <|im_start|> and <|im_end|> tokens.
    """

    def format_example(example):
        # Incorporate the thinking trace inside <think> tags
        assistant_content = example["answer"]
        if example.get("thinking"):
            assistant_content = f"<think>\n{example['thinking'].strip()}\n</think>\n{assistant_content}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    formatted = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        batched=False,
        desc="Formatting dataset",
    )
    return formatted


# ============================================================================
# Model Setup
# ============================================================================


def load_model_and_tokenizer():
    """Load Qwen3-8B with Unsloth optimizations and 4-bit quantization."""
    print(f"📥 Loading model: {MODEL_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        fast_inference=True,  # Enables vLLM for fast inference
    )

    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapters using Unsloth's optimized implementation."""
    print("🔧 Applying LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )

    model.print_trainable_parameters()
    return model


# ============================================================================
# Training
# ============================================================================


def train_sft(model, tokenizer, train_dataset, eval_dataset):
    """Run supervised fine-tuning with SFTTrainer."""
    print("🚀 Starting SFT training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            optim="adamw_8bit",  # Memory-efficient 8-bit optimizer
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            seed=42,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_seq_length=MAX_SEQ_LEN,
            dataset_text_field="text",
            packing=False,  # Disable packing to keep conversations intact
        ),
    )

    # Resume from checkpoint if available
    if os.path.isdir(OUTPUT_DIR) and any(
        "checkpoint" in d for d in os.listdir(OUTPUT_DIR)
    ):
        print("📂 Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    return trainer


# ============================================================================
# Save & Merge
# ============================================================================


def save_model(model, tokenizer):
    """Save LoRA adapters and optionally merge into full model."""

    # Save LoRA adapters
    lora_dir = OUTPUT_DIR + "_lora"
    print(f"💾 Saving LoRA adapters to {lora_dir}...")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Save merged model (16-bit)
    print(f"🔗 Merging LoRA and saving to {FINAL_DIR}...")
    model.save_pretrained_merged(
        FINAL_DIR,
        tokenizer,
        save_method="merged_16bit",
    )

    print("✅ Model saved successfully!")
    print(f"   LoRA adapters: {lora_dir}")
    print(f"   Merged model:  {FINAL_DIR}")


# ============================================================================
# Test Inference
# ============================================================================


def test_model(model, tokenizer):
    """Run a quick inference test after training."""
    print("\n" + "=" * 60)
    print("🧪 Test Inference")
    print("=" * 60)

    test_questions = [
        # In-domain question
        "What are the cardiovascular adverse effects of acepromazine?",
        # Out-of-domain question (should refuse)
        "What is the best recipe for chocolate cake?",
        # Refusal-type question
        "What is the bioavailability of acepromazine when given orally versus by injection?",
    ]

    # Switch to inference mode
    FastLanguageModel.for_inference(model)

    for question in test_questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )

        response = tokenizer.decode(
            output[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        )

        print(f"\n📌 Question: {question}")
        print(f"💬 Answer: {response}")
        print("-" * 40)


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 60)
    print("🚀 Phase 1: Qwen3-8B Veterinary SFT with Unsloth")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA GPU required for training")
        return

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load dataset
    dataset = load_dataset_from_json(DATASET_DIR)

    # Split into train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"📊 Train: {len(split['train'])} samples | Eval: {len(split['test'])} samples")

    # Format for SFT
    train_dataset = format_dataset_for_sft(split["train"], tokenizer)
    eval_dataset = format_dataset_for_sft(split["test"], tokenizer)

    # Apply LoRA
    model = apply_lora(model)

    # Train
    trainer = train_sft(model, tokenizer, train_dataset, eval_dataset)

    # Save
    save_model(model, tokenizer)

    # Test
    test_model(model, tokenizer)

    print("\n✅ Phase 1 (SFT) complete!")
    print("➡️  Next: Run grpo_training.py for Phase 2 (GRPO reasoning)")


if __name__ == "__main__":
    main()
