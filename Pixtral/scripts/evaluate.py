"""
evaluate.py - Evaluate a fine-tuned Pixtral-12B model on a test set.

This script loads a LoRA-finetuned model, iterates through a test JSON file,
generates predictions, and computes performance metrics like accuracy,
a classification report, and a confusion matrix.

Usage:
    python evaluate.py
"""
import json
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm 

# Load the model
BASE_MODEL_ID = "mistral-community/pixtral-12b"
ADAPTER_PATH = "/home/s4nanodi/xpres/Crossmodal-Entailment/Pixtral-Fine-Tuning/output/fine-tuned-model/checkpoint-480" # Path to fine-tuned adapter checkpoint
TEST_JSON_PATH = "/home/s4nanodi/xpres/Crossmodal-Entailment/Pixtral-Fine-Tuning/data/test.json" # Path to test data file


def load_model_and_processor(base_model_id, adapter_path, device="cuda"):
    """Loads the 4-bit quantized base model and applies the LoRA adapter."""
    print("Loading base model in 4-bit with bfloat16...")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    processor = AutoProcessor.from_pretrained(base_model_id)
    return model.eval(), processor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model and processor
    model, processor = load_model_and_processor(BASE_MODEL_ID, ADAPTER_PATH, device)

    # Load the test data
    try:
        with open(TEST_JSON_PATH, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} samples from {TEST_JSON_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Test data file not found at: {TEST_JSON_PATH}")
        return

    all_predictions = []
    all_labels = []

    print("\nStarting evaluation...")

    for i, item in enumerate(tqdm(test_data, desc="Evaluating Samples")):
        try:
            # Parse the Data
            messages = item['messages']
            user_content = messages[0]['content']
            
            prompt_text = next(c['text'] for c in user_content if c['type'] == 'text')
            image_path = next(c['image_path'] for c in user_content if c['type'] == 'image')
            
            ground_truth_label = messages[1]['content'][0]['text'].strip().upper()
            image = Image.open(image_path).convert("RGB")

            # Format the prompt
            messages_for_template = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image"}
                    ]
                }
            ]
            text_prompt_with_placeholder = processor.apply_chat_template(
                messages_for_template, 
                add_generation_prompt=False,
                tokenize=False
            )
            inputs = processor(
                text=[text_prompt_with_placeholder.strip()],
                images=[image],
                return_tensors="pt"
            ).to(device)
            inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)

            # Inference
            generate_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            generated_text = processor.batch_decode(
                generate_ids[:, inputs["input_ids"].size(1):],
                skip_special_tokens=True
            )[0]
            
            prediction = generated_text.strip().upper()
            
            # Print the results for the first 5 samples
            if i < 5:
                print(f"\n--- Sample {i+1} ---")
                print(f"Ground Truth:     {ground_truth_label}")
                print(f"Raw Prediction:   '{generated_text}'")
                print(f"Cleaned Prediction: '{prediction}'")
            
            # Store Results
            all_predictions.append(prediction)
            all_labels.append(ground_truth_label)

        except (FileNotFoundError, KeyError, StopIteration, RuntimeError, ValueError) as e:
            print(f"\nSkipping a sample due to an error: {e}. Check data format or image path.")
            continue

    print("\nEvaluation complete.")
    
    if not all_labels:
        print("No samples were successfully evaluated. Exiting.")
        return

    # Unique labels from the ground truth for the report
    labels = sorted(list(set(all_labels)))

    # Accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print("\n" + "="*50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("="*50)

    print("\n Classification Report:")
    report = classification_report(all_labels, all_predictions, labels=labels, target_names=labels, zero_division=0)
    print(report)
    print("="*50)

    """# Confusion Matrix
    print("\n Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    # plt.savefig("confusion_matrix.png") 
    plt.show()"""


if __name__ == "__main__":
    main()