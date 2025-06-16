# Evaluate Qwen VL 2b fine tuned (on carla figure of 8 datasets) model
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from carla_dataset_preprocessor import load_and_split_carla_dataset
from tqdm import tqdm
import gc

# System message (same as training)
system_message = """You are a Vision Language Model specialized in steering vehicles.
Your task is to decide if the vehicle should turn left, turn right, or go straight based on the provided image and text description.
You should reply with only one of the following:
"Left", "Right", or "Straight".
"""

def format_inference_sample(sample):
    """Formats a dataset sample for inference."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": sample["query"]}
        ]}
    ]

def run_inference(model, processor, dataset, device="cuda"):
    """Runs inference on a dataset and returns predictions and labels."""
    model.eval()
    predictions = []
    labels = []

    for sample in tqdm(dataset, desc="Running inference"):
        # Format sample
        conversation = format_inference_sample(sample)

        # Apply chat template
        text_input = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        # Process image
        image_inputs, _ = process_vision_info(conversation)

        # Prepare inputs
        inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)

        # Trim input tokens
        trimmed_ids = generated_ids[:, inputs.input_ids.shape[1]:]

        # Decode output
        output_text = processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # Validate prediction
        valid_outputs = ["Left", "Right", "Straight"]
        prediction = output_text if output_text in valid_outputs else "Invalid"

        predictions.append(prediction)
        labels.append(sample["label"][0])

        # Clean up
        del inputs, generated_ids, trimmed_ids
        torch.cuda.empty_cache()

    return predictions, labels

def compute_accuracy(predictions, labels):
    """Computes accuracy, ignoring invalid predictions."""
    correct = sum(p == l for p, l in zip(predictions, labels) if p != "Invalid")
    total = sum(1 for p in predictions if p != "Invalid")
    accuracy = (correct / total) * 100 if total > 0 else 0
    invalid_count = sum(1 for p in predictions if p == "Invalid")
    return accuracy, invalid_count

def clear_memory():
    """Clears GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define dataset and model paths
    datasets = [
        {
            "name": "Unbalanced",
            "path": "carla_dataset_640x480_07_3_bins_qwen",
            "model_dir": "qwen2-vl-2b-carla-fine-tuned"
        },
        {
            "name": "Balanced",
            "path": "carla_dataset_640x480_07_3_bins_qwen_balanced",
            "model_dir": "qwen2-vl-2b-carla-fine-tuned-balanced"
        }
    ]

    # Output file
    output_file = "inference_accuracy_20250616.txt"

    with open(output_file, "w") as f:
        for ds in datasets:
            print(f"\nProcessing {ds['name']} dataset...")
            f.write(f"{ds['name']} Dataset\n")

            # Load dataset (use train split)
            try:
                train_dataset, _, _ = load_and_split_carla_dataset(
                    ds["path"], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
                )
                print(f"Loaded {len(train_dataset)} training samples.")
                f.write(f"Training samples: {len(train_dataset)}\n")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                f.write(f"Error loading dataset: {e}\n")
                continue

            # Load model and processor
            try:
                processor = Qwen2VLProcessor.from_pretrained(ds["model_dir"], use_fast=True)
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    ds["model_dir"], torch_dtype=torch.bfloat16, device_map="auto"
                )
                model.to(device)
                print("Model and processor loaded.")
            except Exception as e:
                print(f"Error loading model/processor: {e}")
                f.write(f"Error loading model/processor: {e}\n")
                continue

            # Run inference
            predictions, labels = run_inference(model, processor, train_dataset, device)

            # Compute accuracy
            accuracy, invalid_count = compute_accuracy(predictions, labels)

            # Log results
            print(f"{ds['name']} Accuracy: {accuracy:.2f}%")
            print(f"Invalid predictions: {invalid_count}")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
            f.write(f"Invalid predictions: {invalid_count}\n")
            f.write("\n")

            # Clean up
            del model, processor
            clear_memory()

    print(f"\nResults saved to {output_file}")
