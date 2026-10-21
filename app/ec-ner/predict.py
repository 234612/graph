import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from configuration import *


class Predictor:

    def __init__(self, model, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    def predict(self, inputs: str | list[str]):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        token_list = [list(x) for x in inputs]
        inputs_tensor = self.tokenizer(token_list, is_split_into_words=True, padding=True,
                                       truncation=True, return_tensors="pt")
        inputs_tensor = {k: v.to(self.device) for k, v in inputs_tensor.items()}
        with torch.no_grad():
            outputs = self.model(**inputs_tensor)
        preds = outputs.logits.argmax(dim=-1).tolist()

        final_predictions = []
        for pred, tokens in zip(preds, token_list):
            # 去掉开头 [CLS]
            pred = pred[1:len(tokens) + 1]
            final_predictions.append([self.model.config.id2label[i] for i in pred])
        return final_predictions[0] if is_str else final_predictions

    def extract(self, inputs: str | list[str]):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        entities_list = [self._extract_entities(list(x), labels)
                         for x, labels in zip(inputs, self.predict(inputs))]
        return entities_list[0] if is_str else entities_list

    def _extract_entities(self, tokens, labels):
        entities, current = [], ""
        for token, label in zip(tokens, labels):
            if label == 'B':
                if current:
                    entities.append(current)
                current = token
            elif label == 'I':
                if current:
                    current += token
            else:
                if current:
                    entities.append(current)
                current = ""
        if current:
            entities.append(current)
        return entities


def predict():
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR / "best_model"))
    model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / "best_model"))
    predictor = Predictor(model, tokenizer)
    text = "Bose SoundSport Free 真无线蓝牙耳机 运动耳机 博士防掉落耳塞 黑色"
    print(predictor.extract(text))


if __name__ == '__main__':
    predict()