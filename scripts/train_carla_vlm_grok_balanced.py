import os
import re
import gc
import time
import torch
from datasets import Dataset, Image, Value, Features
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info

# --- 1. Import CARLA Dataset Preprocessor ---
from carla_dataset_preprocessor import load_and_split_carla_dataset

# --- 2. Define System Message and Data Formatting ---
system_message = """You are a Vision Language Model specialized in steering vehicles.
Your task is to decide if the vehicle should turn left, turn right, or go straight based on the provided image and text description.
You should reply with only one of the following:
"Left", "Right", or "Straight".
"""

def format_data(sample):
    """
    Formats a single dataset sample into the chat conversation structure expected by Qwen2-VL.
    The label is extracted as a single string from the list.
    """
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"], # PIL Image object directly
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0] # Take the first element from the label list
                }
            ],
        },
    ]

# --- 3. Memory Management Helper Function ---
def clear_memory():
    """
    Clears CUDA memory and performs garbage collection to free up GPU resources.
    """
    for var in ['inputs', 'model', 'processor', 'trainer', 'peft_model']:
        if var in globals():
            del globals()[var]
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)
        gc.collect()
        time.sleep(2)
        print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("CUDA not available, skipping GPU memory clear.")

# --- Custom Data Collator ---
def collate_fn(examples):
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", use_fast=True)
    # Debug: Print the structure of the first example
    print("Debug: First example in collate_fn:", examples[0])
    
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example)[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_tokens = [151652, 151653, 151655]  # Qwen2-VL image token IDs
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch

# --- Main Script Execution ---
if __name__ == '__main__':
    # Define the path to your balanced CARLA dataset directory
    carla_dataset_path = "carla_dataset_640x480_07_3_bins_qwen_balanced"

    print(f"Starting CARLA VLM Fine-Tuning Script (Balanced Dataset)...")

    # Load and split your custom CARLA dataset
    try:
        carla_train_dataset, carla_eval_dataset, carla_test_dataset = load_and_split_carla_dataset(
            carla_dataset_path,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        print("\nCARLA Dataset loaded and split successfully.")
        print(f"Train samples: {len(carla_train_dataset)}")
        print(f"Eval samples: {len(carla_eval_dataset)}")
        print(f"Test samples: {len(carla_test_dataset)}")

        # --- Format Datasets as Conversation Lists ---
        print("\nFormatting datasets for Qwen2-VL chat template...")
        train_dataset = [format_data(sample) for sample in carla_train_dataset]
        eval_dataset = [format_data(sample) for sample in carla_eval_dataset]
        test_dataset = [format_data(sample) for sample in carla_test_dataset]
        print("Dataset formatting complete.")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Eval samples: {len(eval_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        # Debug: Verify dataset structure
        print("Debug: First train dataset example:", train_dataset[0])

    except Exception as e:
        print(f"Error loading CARLA dataset: {e}")
        exit()

    # --- Load Model and Processor ---
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    print(f"\nLoading model: {model_id} in bfloat16 precision...")
    clear_memory()

    # Load processor with use_fast=True to avoid slow processor warning
    processor = Qwen2VLProcessor.from_pretrained(model_id, use_fast=True)
    # Save processor to update deprecated video processor config
    processor.save_pretrained("qwen2-vl-processor-temp")
    print("Processor saved to update video processor config.")

    # Load model without quantization
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("Model and processor loaded in bfloat16 precision.")

    # --- Set Up QLoRA and SFTConfig ---
    print("\nSetting up QLoRA configuration...")
    model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print("LoRA adapters applied.")
    model.print_trainable_parameters()

    # --- Configure Training Arguments ---
    print("\nConfiguring SFTTrainer arguments...")
    training_args = SFTConfig(
        output_dir="qwen2-vl-2b-instruct-carla-sft-balanced",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=2048,
        dataset_text_field="",  # Disable default text field processing
        dataset_kwargs={"skip_prepare_dataset": True},  # Skip default dataset preprocessing
    )
    training_args.remove_unused_columns = False

    # --- Initialize and Train SFTTrainer ---
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()
    print("Training complete!")

    # --- Save the Fine-tuned Model and Processor ---
    output_model_dir = "qwen2-vl-2b-carla-fine-tuned-balanced"
    print(f"\nSaving fine-tuned model and processor to '{output_model_dir}'...")
    trainer.model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)
    print("Model and processor saved.")

    # --- Final Cleanup ---
    print("\nPerforming final memory cleanup...")
    clear_memory()
    print("Script finished.")
