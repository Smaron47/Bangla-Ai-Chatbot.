# Install required libraries
!pip install datasets transformers kaggle gdown

# Import libraries
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import os

# Check if GPU is available and set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save path for models
SAVE_PATH = "/content/drive/My Drive/Chatbot_Models"
os.makedirs(SAVE_PATH, exist_ok=True)

# Helper function to standardize column names
def standardize_column(dataset, column_name):
    if column_name in dataset.column_names:
        return dataset.rename_column(column_name, "text")
    return dataset

# Load datasets with column standardization
dataset1 = load_dataset("iamshnoo/alpaca-cleaned-bengali")
dataset1 = standardize_column(dataset1['train'], "instruction")

dataset2 = load_dataset("sartajekram/BanglaRQA")
dataset2 = standardize_column(dataset2['train'], "question")

dataset3 = load_dataset("HydraIndicLM/bengali_alpaca_dolly_67k")
dataset3 = standardize_column(dataset3['train'], "prompt")

dataset4 = load_dataset("rishiraj/bengalichat")
dataset4 = standardize_column(dataset4['train'], "dialogue")

# Combine datasets
datasets = [dataset1, dataset2, dataset3, dataset4]
combined_dataset = concatenate_datasets(datasets)
print(f"Combined dataset size: {len(combined_dataset)}")

# Filter invalid rows and preprocess data
def preprocess_function(examples):
    valid_texts = [text for text in examples['text'] if isinstance(text, str) and text.strip()]
    if len(valid_texts) == 0:
        return {"input_ids": [], "attention_mask": []}
    return tokenizer(valid_texts, truncation=True, padding='max_length', max_length=512)

# Load tokenizer
model_name = "bert-base-multilingual-cased"  # Replace with other models for comparison
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize and filter rows
tokenized_dataset = combined_dataset.filter(lambda x: isinstance(x['text'], str) and x['text'].strip() != "")
tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True)

# Split dataset into training and testing
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load model and move to GPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Training arguments with GPU support
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,  # Adjust batch size based on GPU memory
    per_device_eval_batch_size=8,  # Adjust batch size based on GPU memory
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    push_to_hub=False,
    fp16=True,  # Use mixed precision for faster training on GPUs
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Save the model
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
