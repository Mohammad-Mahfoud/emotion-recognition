# emotion-recognition: 

DistilBERT Fine-tuned on Emotion Classification

A compact and reproducible project that fine-tunes DistilBERT on the dair-ai/emotion dataset to perform emotion classification on short text. The repository contains a simple pipeline that loads the dataset, tokenizes text, fine-tunes a Transformer-based classifier, and evaluates it using accuracy and weighted F1-score.

Key features:
###############
-Uses the datasets and transformers libraries for a minimal, end-to-end fine-tuning workflow.

-Dynamic padding via DataCollatorWithPadding for efficiency.

-Evaluation on validation and test splits with accuracy and weighted F1.

-Saves best model automatically using Trainer’s load_best_model_at_end.

-Small, CPU-friendly default batch sizes — easy to run on modest hardware.