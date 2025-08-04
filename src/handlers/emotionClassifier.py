import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

class EmotionClassifier:
    def __init__(self, model_path="models/emotionClassifier"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.labels: list = ["sadness", "grief", "fear", "remorse", "disappointment", "nervousness", "embarrassment"]

    def predict(self, text, threshold=0.9):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

        for label, prob in zip(self.labels, probs):
            if prob >= threshold:
                return label
        return None

# emotionClassifier = EmotionClassifier("models/emotionClassifier")
# print(emotionClassifier.predict("I feel overwhelmed and not sure if I can keep up this term"))
# print(emotionClassifier.predict("When is the tuition payment deadline"))