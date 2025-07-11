#This is my first line of the projectfrom transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import os

# Config
MODEL_NAME = "gpt2"  # or "gpt2-medium" if you have more GPU memory
DATA_FILE = "dataset.txt"
OUTPUT_DIR = "./gpt2-finetuned"

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Make sure the tokenizer uses padding (GPT2 normally doesn’t)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load dataset
def load_custom_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    return Dataset.from_dict({"text": lines})

dataset = load_custom_dataset(DATA_FILE)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator for dynamic batching
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="no",
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    prediction_loss_only=True,
    fp16=True,  # Enable if you have a GPU with mixed precision
    logging_dir='./logs'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Training complete! Model saved at {OUTPUT_DIR}")
