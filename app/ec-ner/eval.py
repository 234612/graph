import evaluate
from datasets import load_from_disk
from transformers import Trainer, AutoModelForTokenClassification, AutoTokenizer, \
    DataCollatorForTokenClassification, EvalPrediction

from configuration import *


def eval():
    model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / "best_model"))
    eval_dataset = load_from_disk(PROCESSED_DATA_DIR / 'test')
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR / 'best_model'))
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, return_tensors='pt')

    def compute_metrics_fn(eval: EvalPrediction):
        seqeval = evaluate.load('seqeval')
        preds = eval.predictions.argmax(axis=-1)
        labels = eval.label_ids
        unpad_labels, unpad_preds = [], []
        for pred, label in zip(preds, labels):
            m = label != -100
            unpad_labels.append([model.config.id2label[i] for i in label[m]])
            unpad_preds.append([model.config.id2label[i] for i in pred[m]])
        return seqeval.compute(references=unpad_labels, predictions=unpad_preds)

    trainer = Trainer(model=model, eval_dataset=eval_dataset, data_collator=collator,
                      compute_metrics=compute_metrics_fn)
    result = trainer.evaluate()
    print(result)


if __name__ == '__main__':
    eval()