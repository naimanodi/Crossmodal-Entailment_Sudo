import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()


import os
from collections import Counter
# Disable HF Hub transfer and set visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

import json
#import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback  
)
from accelerate import Accelerator
from datasets import Dataset
import torch
from datasets import Features, Array3D, Value
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate
import numpy as np
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
from torchvision.transforms.functional import to_pil_image, resize

from transformers import BitsAndBytesConfig #add-ons
accelerator = Accelerator()

model_id = "mistral-community/pixtral-12b"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
with accelerator.main_process_first():
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True
        #device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
processor.image_processor.size = {"height": 336, "width": 336}

def get_image_token_count(image, dummy_text="describe this image"):
    """
    Compute the number of tokens generated for an image using the model's vision tower.
    Returns 0 if token computation fails.
    """
    try:
        inputs = processor(images=image, text=dummy_text, return_tensors="pt").to("cuda")
        inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype) #add-ones
        with torch.no_grad():
            output = model.vision_tower(pixel_values=inputs["pixel_values"])
        token_count = output.last_hidden_state.shape[1]
        if token_count == 0:
            raise ValueError("Image token count is zero.")
        return token_count
    except Exception as e:
        print(f"[ERROR] Failed to compute image tokens: {e}")
        return 0  


CHAT_TEMPLATE = """
{%- for message in messages %}
  {%- if message.role == "user" %}
      <s>[INST]
      {%- for item in message.content %}
          {%- if item.type == "text" %}
              {{ item.text }}
          {%- elif item.type == "image" %}
              \n[IMG]
          {%- endif %}
      {%- endfor %}
      [/INST]
  {%- elif message.role == "assistant" %}
      {%- for item in message.content %}
          {%- if item.type == "text" %}
              {{ item.text }}
          {%- endif %}
      {%- endfor %}
      </s>
  {%- endif %}
{%- endfor %}
""" 
processor.tokenizer.chat_template = CHAT_TEMPLATE
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.add_special_tokens({'additional_special_tokens': ['[IMG]']})


with open("/home/s4nanodi/xpres/Crossmodal-Entailment/Pixtral-Fine-Tuning/data/json_final/train_final.json", "r") as f: 
    train_raw_data = json.load(f)
    print(f"Loaded {len(train_raw_data)} samples from the training dataset.")

with open("/home/s4nanodi/xpres/Crossmodal-Entailment/Pixtral-Fine-Tuning/data/json_final/dev_final.json", "r") as f: 
    val_raw_data = json.load(f)
    print(f"Loaded {len(val_raw_data)} samples from the validation dataset.")


train_labels = [item['messages'][1]['content'][0]['text'] for item in train_raw_data]
print(f"Training data balance: {Counter(train_labels)}")

# Extract the first conversation
messages = train_raw_data[0]["messages"]

formatted_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

print(formatted_text)

def prepare_sample_batch(raw_data): 
    """
    Convert raw JSON dataset into the format expected by the DataCollator.
    """
    sample_batch = []

    for item in raw_data: 
        try:
            messages = item["messages"]

            question = next(c['text'] for c in messages[0]['content'] if c['type'] == 'text')
            image_path = next(c['image_path'] for c in messages[0]['content'] if c['type'] == 'image')
            answer = messages[1]["content"][0]["text"]
            
            sample_batch.append({
                "question": question,
                "answer": answer,
                "image_path": image_path
            })
        except Exception as e:
            print(f"Skipping sample due to error: {e}")
    return sample_batch


train_dataset_list = prepare_sample_batch(train_raw_data)
eval_dataset_list = prepare_sample_batch(val_raw_data)

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.inst_token_ids = self.processor.tokenizer(
            "[/INST]", add_special_tokens=False
        )["input_ids"]

    def find_subsequence(self, sequence, subsequence):
        seq_len = len(sequence)
        sub_len = len(subsequence)
        for i in range(seq_len - sub_len + 1):
            if torch.equal(sequence[i:i+sub_len], subsequence):
                return i
        return None

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = Image.open(example["image_path"]).convert("RGB")
            messages = [
                {"role": "user", "content": [{"type": "text", "text": example["question"]}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]}
            ]
            text = self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(image)

        if not images:
            return {}

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024 #2048
        )
        
        labels = batch["input_ids"].clone()
        for i, input_ids in enumerate(labels):
            start_idx = self.find_subsequence(input_ids, torch.tensor(self.inst_token_ids).to(input_ids.device))
            if start_idx is not None:
                label_start_index = start_idx + len(self.inst_token_ids)
                labels[i, :label_start_index] = -100
                eos_token_id = self.processor.tokenizer.eos_token_id
                eos_indices = torch.where(input_ids == eos_token_id)[0]
                if len(eos_indices) > 0:
                    labels[i, eos_indices[0] + 1:] = -100
            else:
                labels[i, :] = -100
        batch["labels"] = labels
        return batch

data_collator = MyDataCollator(processor)

# --- (LoraConfig with increased dropout) ---
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    use_rslora=True,
    target_modules="all-linear",
    lora_dropout=0.15,  # Increased dropout for better generalization
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- (Robust TrainingArguments for Generalization) ---
training_args = TrainingArguments(
    num_train_epochs=10,  # Set a high max; Early Stopping will find the true best epoch
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    
    learning_rate=2e-5,
    lr_scheduler_type="cosine",  # Use a scheduler for better stability
    warmup_ratio=0.1,
    weight_decay=0.02, # Increased weight decay
    
    output_dir="/home/s4nanodi/xpres/Crossmodal-Entailment/Pixtral-Fine-Tuning/output_final/fine-tuned-model",
    eval_strategy="steps",
    eval_steps=250,  # Evaluate every 250 steps
    save_strategy="steps",
    save_steps=250,  # Save checkpoint at the same interval
    load_best_model_at_end=True, # This is the key for Early Stopping
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    logging_steps=50,
    bf16=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
   #gradient_checkpointing_kwargs={'use_reentrant': True},
)

model = accelerator.prepare(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset_list,  
    eval_dataset=eval_dataset_list,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)   

trainer.train()