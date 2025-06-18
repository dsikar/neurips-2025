import torch
import numpy as np
import os
import time
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import argparse

# System message for Qwen2-VL
system_message = """You are a Vision Language Model specialized in steering vehicles.
Your task is to decide if the vehicle should turn left, turn right, or go straight based on the provided image and text description.
You should reply with only one of the following:
"Left", "Right", or "Straight".
"""

# Hardcoded token IDs for Qwen2-VL-2B-Instruct
QWEN_STEERING_TOKEN_MAP = {
    "Left": 5415,
    "Straight": 88854,
    "Right": 5979,
}

# Steering angle mapping for 3 bins
STEERING_ANGLES = {
    0: -0.0650,  # Left
    1: 0.0000,   # Straight
    2: 0.0650    # Right
}

def load_model(model_id, adapter_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = Qwen2VLProcessor.from_pretrained(model_id, use_fast=True)
        model.load_adapter(adapter_path)
        model.to(device)
        model.eval()
        print("Model and processor loaded successfully.")
        return model, processor, device
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        raise

def predict_image(model, processor, device, image_path):
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare conversation
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What steering direction should the vehicle take?"}
            ]}
        ]
        
        # Apply chat template and process image
        text_input = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversation)
        inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(device)
        
        # Get model outputs (logits)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits
        
        # Get last token logits and apply softmax
        last_logits = logits[0, -1, :]  # Shape: [vocab_size]
        probs = torch.softmax(last_logits, dim=-1)
        
        # Get probabilities for Left, Straight, Right
        prob_left = probs[QWEN_STEERING_TOKEN_MAP["Left"]].item()
        prob_straight = probs[QWEN_STEERING_TOKEN_MAP["Straight"]].item()
        prob_right = probs[QWEN_STEERING_TOKEN_MAP["Right"]].item()
        
        ordered_probabilities = [prob_left, prob_straight, prob_right]
        predicted_label = np.argmax(ordered_probabilities)  # Class index (0, 1, 2)
        predicted_angle = STEERING_ANGLES[predicted_label]  # Map to steering angle
        
        return ordered_probabilities, predicted_label, predicted_angle
    except Exception as e:
        print(f"Error processing image {os.path.basename(image_path)}: {e}")
        return None, None, None

def main(args):
    # Load model
    model, processor, device = load_model(args.model_id, args.adapter_path)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Monitoring {args.input_dir} for images...")
    
    while True:
        try:
            # Check for image files
            files = [f for f in os.listdir(args.input_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            if files:
                # Process first image found
                image_file = files[0]
                image_path = os.path.join(args.input_dir, image_file)
                print(f"Read image: {image_file}")
                
                # Generate prediction
                softmax_probs, predicted_label, predicted_angle = predict_image(model, processor, device, image_path)
                
                if softmax_probs is not None:
                    # Save output based on --softmax flag
                    output_file = os.path.join(args.output_dir, args.filename)
                    if args.softmax:
                        np.save(output_file, np.array(softmax_probs))
                        print(f"Predicted: (Softmax: {softmax_probs}, Class index: {predicted_label}), Saved to {output_file}")
                    else:
                        with open(output_file, 'w') as f:
                            f.write(f"{predicted_angle}\n")
                        print(f"Predicted: (Softmax: {softmax_probs}, Class index: {predicted_label}), Saved to {output_file}")
                    
                    # Delete image
                    os.remove(image_path)
                    print(f"Deleted image: {image_file}")
            
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL Steering Angle Inference Loop")
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2-VL-2B-Instruct',
                        help='HuggingFace model ID for Qwen2-VL')
    parser.add_argument('--adapter_path', type=str, 
                        default='/users/aczd097/archive/git/neurips-2025/scripts/qwen2-vl-2b-instruct-carla-sft-balanced/checkpoint-320',
                        help='Path to fine-tuned Qwen2-VL adapter')
    parser.add_argument('--input_dir', type=str, 
                        default='/users/aczd097/archive/git/neurips-2025/qwen/input_dir',
                        help='Directory to monitor for input images')
    parser.add_argument('--output_dir', type=str, default='/users/aczd097/archive/git/neurips-2025/qwen/output_dir',
                        help='Directory to save output files')
    parser.add_argument('--filename', type=str, default='prediction',
                        help='Output filename (without extension)')
    parser.add_argument('--softmax', action='store_true',
                        help='Save softmax probabilities to .npy file instead of single prediction')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
