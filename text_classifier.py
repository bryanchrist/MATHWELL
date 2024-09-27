# This script contains code to train a text classifier for solvability. You can easily modify it to create a classifer for accuracy or appropriateness. 
from datasets import load_dataset, DatasetDict
import evaluate
import pandas as pd
import numpy as np
from transformers import DataCollatorWithPaddingm, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
import logging
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Prepare data for solvability classifier
df = pd.read_csv('data/annotations.csv')
# Restrict df to the initial set of human annotation data generated from finetuned llama-2 to replicate experiments reported in paper
df = df[df['model'] == "finetuned_llama"]
df['solvability'] = df['solvability'].astype('int')
df = df.rename({'solvability': 'label', 'question': 'text'}, axis = 1)
df = df[['text', 'label']]
df = df[df['text'].notnull()]
# Find class weights for modified training objective based on inverse class balance
neg_weight = df['label'].mean()
pos_weight = 1 - neg_weight
df.to_csv('data/solvability.csv')

# Load the dataset
dataset = load_dataset('csv', data_files="data/solvability.csv")

# Do train/valid/test split
dataset_train_valid_test = dataset['train'].train_test_split(test_size = .2, seed = 42)
dataset_valid_test = dataset_train_valid_test['test'].train_test_split(test_size = .5, seed = 42)
# Create a DatasetDict to hold the splits
train_test_valid_dataset = DatasetDict({
    'train': dataset_train_valid_test['train'],
    'test': dataset_valid_test['train'],
    'valid': dataset_valid_test['test']
})

# Set up tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocess and collate data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train_test_valid_dataset = train_test_valid_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# This is a function to load and compute the metrics reported in our paper for the classifier
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    balanced_accuracy = evaluate.load('hyperml/balanced_accuracy')
    roc_auc_score= evaluate.load('roc_auc')
    f1 = evaluate.load('f1')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    predictions, labels = eval_pred
    
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert NumPy array to PyTorch tensor
    predictions_tensor = torch.from_numpy(predictions)
    
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(predictions_tensor, dim=1)
    
    # Convert probabilities back to NumPy array
    probabilities_np = probabilities.numpy()
    
    accuracy = accuracy.compute(predictions=predicted_labels, references=labels)
    balanced_accuracy = balanced_accuracy.compute(predictions=predicted_labels, references=labels)
    roc_auc = roc_auc_score.compute(references = labels, prediction_scores = probabilities_np[:, 1])
    f1 = f1.compute(predictions=predicted_labels, references=labels)
    precision = precision.compute(predictions=predicted_labels, references=labels)
    recall = recall.compute(predictions=predicted_labels, references=labels)
    return {'accuracy':  accuracy, 'balanced_accuracy' : balanced_accuracy, 'f1':f1, 'auc': roc_auc, 'precision': precision, 'recall': recall}

# Create labels for classifier
id2label = {0: "NOT SOLVABLE", 1: "SOLVABLE"}
label2id = {"NOT SOLVABLE": 0, "SOLVABLE": 1}

# Load distilbert 
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# Create tensor for class weights
class_weights = torch.tensor([neg_weight, pos_weight], device = model.device)

# Create class for weighted trainer
class WeightedBertTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # Assuming labels are passed in the inputs
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(logits, dim=1)

        # Assuming binary classification, so consider only the positive class (index 1)
        positive_class_probabilities = probabilities[:, 1]
        # Use weighted binary cross-entropy loss
        if self.class_weights is not None:
            # Ensure class_weights tensor matches the size of the labels tensor
            loss_fct = nn.CrossEntropyLoss(weight = self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        else:
            # If class weights are not provided, use regular binary cross-entropy loss
            loss = F.binary_cross_entropy(positive_class_probabilities, labels.float())
                
        if return_outputs:
            return loss, outputs
        return loss
        
# Set up training arguments
training_args = TrainingArguments(
    output_dir="solvability_classifier",
    learning_rate=.0001,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    weight_decay=.9,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_strategy = 'epoch',
    logging_steps = 1)

# Set up trainer
trainer = WeightedBertTrainer(class_weights = class_weights,
     model=model,
     args=training_args,
     train_dataset=tokenized_train_test_valid_dataset["train"],
     eval_dataset=tokenized_train_test_valid_dataset["valid"],
     tokenizer=tokenizer,
     data_collator=data_collator,
     compute_metrics=compute_metrics,
 )

# Train model
trainer.train()

# Evaluate performance on test data
trainer.evaluate(eval_dataset = tokenized_train_test_valid_dataset["test"])
