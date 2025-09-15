from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments, Trainer,DataCollatorWithPadding
import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# 1. Load dataset
dataset = load_dataset("dair-ai/emotion")

# 2. Inspect dataset
df = dataset["train"].to_pandas()
classes = dataset["train"].features["label"].names
df["label_name"] = df["label"].apply(lambda x: classes[x])

# Plot label distribution
df["label_name"].value_counts().plot.barh()
plt.title("Frequency of classes")
plt.show()

# 3. Tokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)  # No static padding

tokenized_data = dataset.map(tokenize_function, batched=True)

# 4. Model
num_labels = len(classes)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=num_labels
)

# 5. Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="distilbert-finetuned-emotion",
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # CPU-friendly
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_score",
    greater_is_better=True,
    disable_tqdm=False,
    seed=42
)

# 7. Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1_score": f1}

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 9. Train
trainer.train()

# 10. Evaluate / Predict
preds_outputs = trainer.predict(tokenized_data["test"])
print(preds_outputs.metrics)
