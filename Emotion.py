from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, pipeline
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# Load the dataset from a CSV file
dataset = load_dataset("csv", data_files="emotions.csv")


labels = list(set(dataset["train"]["label"])) # Store the data in a list Set() to make sure therer are no duplicates.

# Dictionary for index and reverse_index
index = {label: i for i, label in enumerate(labels)} 
reverse_index = {i: label for label, i in index.items()}
print("\nindex:")
print(index)

# Convert string labels to index
def encode_labels(example):
    example["label"] = index[example["label"]]
    return example
dataset = dataset.map(encode_labels)

# Load tokenizer to convert text to BERT input format
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize each text by adding padding and truncations
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)

# Split dataset: 80% for training, 20% for evaluation
dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load pre-trained DistilBERT model with correct label count
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(index),
    id2label=reverse_index,
    label2id=index
)

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Set training configuration
training_args = TrainingArguments(
    output_dir="./emotion_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=10
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("./emotion_model")
tokenizer.save_pretrained("./emotion_model")

# Load the model and tokenizer for inference
model = DistilBertForSequenceClassification.from_pretrained("./emotion_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./emotion_model")

# Create a classifier pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
