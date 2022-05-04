import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, roc_auc_score, hamming_loss
from scipy.special import expit as sigmoid

device = "cuda:0" if torch.cuda.is_available() else "cpu"
labels = ["label_identity_attack", "label_insult", "label_obscene",
          "label_severe_toxicity", "label_threat", "label_toxicity"]

dataset = load_dataset('json', data_files={'train': 'train.jsonl', 'test': 'test.jsonl'})

tokenizer_name = "TurkuNLP/bert-base-finnish-cased-v1"  # or "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset = dataset.map(lambda it: {'labels': torch.FloatTensor([it[label] for label in labels])})
dataset = dataset.remove_columns(labels).remove_columns(["id", "lang"])
tokenized_data = dataset.map(lambda items: tokenizer(items["text"],
                                                     truncation=True,
                                                     max_length=256,
                                                     padding="max_length"), batched=True)
tokenized_data = tokenized_data.remove_columns(["text"])

model = AutoModelForSequenceClassification.from_pretrained(
    tokenizer_name,
    num_labels=len(labels),
    problem_type="multi_label_classification",
    id2label={int(i): v for i, v in enumerate(labels)},
    label2id={str(v): i for i, v in enumerate(labels)},
)

# Calculate weights
df_pandas = pd.DataFrame(dataset['train'])
class_counts = [v for v in df_pandas[labels].sum()]
class_weights = np.ones_like(class_counts)

neg_counts = [len(dataset['train']) - count for count in class_counts]
for i, (pos, neg) in enumerate(zip(class_counts, neg_counts)):
    class_weights[i] = (neg / pos) * 0.5
class_weights = torch.tensor(class_weights).to(device)


def compute_metrics(pred):
    y_act = pred.label_ids
    y_pred = sigmoid(pred.predictions)
    y_pred = (y_pred > 0.5).astype(float)
    metrics = classification_report(y_act, y_pred, target_names=labels, output_dict=True)
    metrics["avg_f1"] = metrics["weighted avg"]["f1-score"]
    metrics['roc_auc'] = roc_auc_score(y_act, y_pred, labels=labels, average="macro")
    metrics['hamming_l0ss'] = hamming_loss(y_act, y_pred)
    return metrics


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        loss = loss_func(outputs['logits'], inputs["labels"])
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="steps",
    warmup_steps=2000,
    eval_steps=2500,
    save_steps=2500,
    save_total_limit=5,
    metric_for_best_model="eval_avg_f1"
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"].select(range(20_000)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()