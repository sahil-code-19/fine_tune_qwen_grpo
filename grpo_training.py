"""
Phase 2: GRPO (Group Relative Policy Optimization) for Qwen3-8B using Unsloth.

This script applies GRPO reinforcement learning on top of the SFT-trained model
to reinforce reasoning behavior, calibrated confidence, and proper refusal.

Run this AFTER sft_training.py has completed successfully.

Key features:
- Loads from the SFT LoRA checkpoint
- Custom veterinary-domain reward functions
- Leverages Qwen3's native <think> reasoning tokens
- fast_inference=True leverages vLLM for high-throughput generation

Reward functions:
1. correctness_reward  - Key medical terms from ground truth appear in response
2. refusal_reward      - Proper refusal for out-of-domain, no unnecessary refusal
3. safety_reward       - Safety-flagged QAs include WARNING: prefix
4. confidence_reward   - Response calibration matches expected confidence level
5. format_reward       - Structured, professional responses without repetition
"""

import json
import os
import re

import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

# ============================================================================
# Configuration
# ============================================================================

# Load from SFT checkpoint (LoRA adapters)
SFT_LORA_DIR = "./models/qwen3-sft-lora"
# Fallback: load base model if no SFT checkpoint
BASE_MODEL_NAME = "unsloth/Qwen3-8B"

DATASET_DIR = "./dataset"
OUTPUT_DIR = "./models/qwen3-grpo"
FINAL_DIR = "./models/qwen3-grpo-merged"

MAX_SEQ_LEN = 2048

# GRPO-specific settings
NUM_GENERATIONS = 4        # Number of completions per prompt (reduce if low VRAM)
MAX_COMPLETION_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

# Training hyperparameters
LEARNING_RATE = 5e-6       # Lower LR for RL fine-tuning
MAX_STEPS = 500            # Increase for better results (300+ recommended)
LOGGING_STEPS = 10
SAVE_STEPS = 100

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

## Tone:
Professional, precise, safety-focused. Prioritize accuracy over completeness."""


# ============================================================================
# Dataset Loading for GRPO
# ============================================================================


def load_grpo_dataset(dataset_dir: str) -> Dataset:
    """
    Load dataset for GRPO training.

    For GRPO, we need:
    - prompt: The formatted conversation prompt (system + user message)
    - answer: The ground truth answer (used in reward functions)
    - Plus metadata fields used by reward functions (confidence, refusal, safety_flag)

    Unlike SFT, we do NOT expand paraphrases here — GRPO benefits from
    unique prompts rather than many variants of the same question.
    We use only the original questions.
    """
    print("📂 Loading dataset for GRPO...")
    samples = []

    for filename in os.listdir(dataset_dir):
        if not filename.endswith(".json"):
            continue
        if filename.startswith("stats") or filename.startswith("dataset"):
            continue

        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_pairs = data.get("qa_pairs", [])
        print(f"  Processing {filename} with {len(qa_pairs)} QA pairs...")

        for qa in qa_pairs:
            question = qa["question"].strip()
            answer = qa["answer"].strip()
            confidence = qa.get("confidence", "high")
            is_refusal = qa.get("refusal", False)
            safety_flag = qa.get("safety_flag", False)

            # Build the prompt as a conversation
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

            samples.append({
                "prompt": prompt,
                "answer": answer,
                "confidence": confidence,
                "refusal": is_refusal,
                "safety_flag": safety_flag,
            })

    print(f"✅ GRPO dataset loaded: {len(samples)} samples (original questions only)")
    return Dataset.from_list(samples)


# ============================================================================
# Reward Functions
# ============================================================================


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward based on whether key medical terms from the ground truth
    appear in the model's response.

    This uses a word-overlap approach rather than exact matching,
    since GRPO-generated responses may rephrase the answer.

    Score: 0.0 to 2.0
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, gt_answer in zip(responses, answer):
        # Extract key terms from ground truth (words > 4 chars, likely medical terms)
        gt_words = set(
            word.lower()
            for word in re.findall(r'\b\w+\b', gt_answer)
            if len(word) > 4
        )

        if not gt_words:
            rewards.append(1.0)  # Neutral if no key terms
            continue

        # Count how many key terms appear in the response
        response_lower = response.lower()
        matches = sum(1 for word in gt_words if word in response_lower)
        overlap_ratio = matches / len(gt_words)

        # Scale: 0.0 (no overlap) to 2.0 (full overlap)
        rewards.append(overlap_ratio * 2.0)

    return rewards


def refusal_reward_func(prompts, completions, answer, refusal, **kwargs) -> list[float]:
    """
    Reward the model for correctly refusing or not refusing.

    - If the ground truth is a refusal (refusal=True): reward if response
      contains refusal phrases, penalize if it gives a confident answer.
    - If the ground truth is NOT a refusal: penalize if the model unnecessarily refuses.

    Score: -1.0 to 1.5
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    refusal_phrases = [
        "i don't have reliable information",
        "i don't have reliable",
        "not in my training data",
        "please consult",
        "i cannot provide",
        "outside my area",
        "not covered in my training",
        "i'm not able to",
        "i am not able to",
        "specialized in veterinary",
    ]

    for response, is_refusal in zip(responses, refusal):
        response_lower = response.lower()
        has_refusal = any(phrase in response_lower for phrase in refusal_phrases)

        if is_refusal:
            # Should refuse — reward refusal, penalize confident answer
            rewards.append(1.5 if has_refusal else -1.0)
        else:
            # Should NOT refuse — penalize unnecessary refusal
            rewards.append(-0.5 if has_refusal else 0.5)

    return rewards


def safety_reward_func(prompts, completions, safety_flag, **kwargs) -> list[float]:
    """
    Reward for including WARNING: prefix in safety-critical answers.

    - If safety_flag=True: reward if response starts with or contains "WARNING:"
    - If safety_flag=False: neutral (no reward or penalty)

    Score: 0.0 to 1.0
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, is_safety in zip(responses, safety_flag):
        if is_safety:
            # Should include WARNING:
            if "WARNING:" in response or "WARNING" in response.upper()[:50]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            # Not safety-critical — neutral
            rewards.append(0.5)

    return rewards


def confidence_reward_func(prompts, completions, confidence, **kwargs) -> list[float]:
    """
    Reward based on whether the response's apparent confidence
    matches the expected confidence level from the dataset.

    - High confidence: reward direct, assertive answers
    - Medium confidence: reward hedging language ("may", "could", "in some cases")
    - Low confidence: reward explicit uncertainty or refusal

    Score: -0.5 to 1.0
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    hedging_words = ["may", "could", "might", "possibly", "in some cases",
                     "uncertain", "limited information", "not fully clear"]

    for response, conf_level in zip(responses, confidence):
        response_lower = response.lower()
        has_hedging = any(word in response_lower for word in hedging_words)
        is_short = len(response.split()) < 20

        if conf_level == "high":
            # High confidence: prefer direct, complete answers
            if not has_hedging and not is_short:
                rewards.append(1.0)
            elif has_hedging:
                rewards.append(0.3)  # Mild penalty for unnecessary hedging
            else:
                rewards.append(0.5)
        elif conf_level == "medium":
            # Medium confidence: prefer some hedging/qualification
            if has_hedging:
                rewards.append(1.0)
            else:
                rewards.append(0.3)
        else:  # low
            # Low confidence: prefer refusal or heavy qualification
            if is_short or has_hedging:
                rewards.append(1.0)
            else:
                rewards.append(-0.5)

    return rewards


def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward for well-structured, professional formatting.

    Penalizes:
    - Excessive repetition (signs of generation loop issues)
    - Extremely short responses (likely truncated)
    - Extremely long responses (likely hallucinating)

    Rewards:
    - Moderate length (50-300 words)
    - No repeated phrases

    Score: -1.0 to 1.0
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        word_count = len(response.split())

        # Penalty for repetition (check for repeated sentences)
        sentences = response.split(".")
        if len(sentences) > 3:
            unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
            repetition_ratio = len(unique_sentences) / len(
                [s for s in sentences if s.strip()]
            )
            if repetition_ratio < 0.5:
                # More than 50% repeated sentences — severe penalty
                rewards.append(-1.0)
                continue

        # Reward appropriate length
        if word_count < 10:
            rewards.append(-0.5)  # Too short
        elif word_count > 500:
            rewards.append(-0.3)  # Too long
        elif 30 <= word_count <= 300:
            rewards.append(1.0)   # Sweet spot
        else:
            rewards.append(0.5)   # Acceptable

    return rewards


# ============================================================================
# Model Setup
# ============================================================================


def load_model_for_grpo():
    """
    Load model for GRPO training.

    Priority: Load from SFT LoRA checkpoint if available,
    otherwise load base model.
    """
    if os.path.isdir(SFT_LORA_DIR):
        print(f"📥 Loading SFT checkpoint from: {SFT_LORA_DIR}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=SFT_LORA_DIR,
            max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=True,
            fast_inference=True,  # Enables vLLM for faster GRPO rollouts
        )
    else:
        print(f"⚠️ No SFT checkpoint found at {SFT_LORA_DIR}")
        print(f"📥 Loading base model: {BASE_MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=True,
            fast_inference=True,
        )

        # Apply LoRA since we're starting from base
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    return model, tokenizer


# ============================================================================
# Training
# ============================================================================


def train_grpo(model, tokenizer, dataset):
    """Run GRPO training with custom veterinary reward functions."""
    print("🚀 Starting GRPO training...")
    print(f"   Generations per prompt: {NUM_GENERATIONS}")
    print(f"   Max steps: {MAX_STEPS}")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        seed=42,
        report_to="none",

        # GRPO-specific: temperature for generation rollouts
        temperature=0.7,
        top_p=0.9,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            refusal_reward_func,
            safety_reward_func,
            confidence_reward_func,
            format_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Resume from checkpoint if available
    if os.path.isdir(OUTPUT_DIR) and any(
        "checkpoint" in d for d in os.listdir(OUTPUT_DIR)
    ):
        print("📂 Resuming from GRPO checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    return trainer


# ============================================================================
# Save
# ============================================================================


def save_model(model, tokenizer):
    """Save GRPO-trained model."""

    # Save LoRA adapters
    lora_dir = OUTPUT_DIR + "_lora"
    print(f"💾 Saving GRPO LoRA adapters to {lora_dir}...")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Save merged model
    print(f"🔗 Merging and saving to {FINAL_DIR}...")
    model.save_pretrained_merged(
        FINAL_DIR,
        tokenizer,
        save_method="merged_16bit",
    )

    print("✅ GRPO model saved successfully!")
    print(f"   LoRA adapters: {lora_dir}")
    print(f"   Merged model:  {FINAL_DIR}")


# ============================================================================
# Test Inference
# ============================================================================


def test_model(model, tokenizer):
    """Test the GRPO-trained model with various question types."""
    print("\n" + "=" * 60)
    print("🧪 GRPO Model Test Inference")
    print("=" * 60)

    test_questions = [
        # In-domain, high confidence
        "What are the main contraindications for acepromazine use?",
        # In-domain, should produce reasoning
        "Can I use acepromazine in a dog with severe cardiac disease and pre-existing hypotension?",
        # Should refuse — out of domain
        "How do I make pasta carbonara?",
        # Should refuse — not in training data
        "What is the LD50 of acepromazine in rabbits?",
        # Safety-flagged question
        "Is acepromazine safe to use as the only sedative for an aggressive dog?",
    ]

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
    print("🚀 Phase 2: GRPO Reasoning Reinforcement with Unsloth")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA GPU required for training")
        return

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Load model (from SFT checkpoint or base)
    model, tokenizer = load_model_for_grpo()

    # Load dataset
    dataset = load_grpo_dataset(DATASET_DIR)

    # Train with GRPO
    trainer = train_grpo(model, tokenizer, dataset)

    # Save
    save_model(model, tokenizer)

    # Test
    test_model(model, tokenizer)

    print("\n✅ Phase 2 (GRPO) complete!")
    print("🎉 Your Qwen3-8B veterinary reasoning model is ready!")


if __name__ == "__main__":
    main()
