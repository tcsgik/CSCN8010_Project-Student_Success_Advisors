import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

class EmotionClassifier:
    """
    Classifies emotional tone in a given text using a fine-tuned BERT model.

    This class loads a pre-trained emotion classification model (fine-tuned on a subset of 
    negative emotions) and predicts the most probable emotion label from a fixed list, 
    if the confidence exceeds a specified threshold.

    Methods:
        predict(text, threshold=0.9):
            Returns the predicted emotion label if the model's confidence exceeds the threshold;
            otherwise, returns None.
    """
    def __init__(self, model_path="models/emotionClassifier"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.labels: list = ["sadness", "grief", "fear", "remorse", "disappointment", "nervousness", "embarrassment"]

    def predict(self, text, threshold=0.85):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

        for label, prob in zip(self.labels, probs):
            if prob >= threshold:
                return label
        return None