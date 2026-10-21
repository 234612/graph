import evaluate
from datasets import load_from_disk
from transformers import Trainer, AutoModelForTokenClassification, TrainingArguments, AutoTokenizer, \
    DataCollatorForTokenClassification, EvalPrediction

from configuration import *


def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for i, label in enumerate(LABELS)}

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id)

    train_dataset = load_from_disk(PROCESSED_DATA_DIR / 'train')
    eval_dataset = load_from_disk(PROCESSED_DATA_DIR / 'valid')

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 最终交付模型由 trainer.save_model() 单独落到 CHECKPOINT_DIR / "best_model"
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),

        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=int(EPOCHS * len(train_dataset) // BATCH_SIZE * WARMUP_RATIO),
        weight_decay=WEIGHT_DECAY,
        save_steps=SAVE_STEPS,
        save_strategy='steps',
        save_total_limit=None,
        fp16=True,
        logging_strategy='steps',
        logging_steps=LOGS_STEPS,
        eval_strategy='steps',
        eval_steps=EVAL_STEPS,
        metric_for_best_model='eval_overall_f1',
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    def compute_metrics_fn(eval: EvalPrediction):
        seqeval = evaluate.load('seqeval')
        preds = eval.predictions.argmax(axis=-1)
        labels = eval.label_ids
        unpad_labels, unpad_preds = [], []
        for pred, label in zip(preds, labels):
            m = label != -100
            unpad_labels.append([id2label[i] for i in label[m]])
            unpad_preds.append([id2label[i] for i in pred[m]])
        return seqeval.compute(references=unpad_labels, predictions=unpad_preds)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()

    trainer.save_model(str(CHECKPOINT_DIR / 'best_model'))


if __name__ == '__main__':
    train()