import torch
from transformers import BertTokenizer, BertForSequenceClassification

class IntentClassifier():
    """
    Classifies the intent behind a student's query using a fine-tuned BERT model.

    This class loads a pre-trained intent classification model and predicts the most 
    likely intent category from a predefined list of student-related intents such as 
    course information or enrollment.

    Methods:
        predict(text):
            Returns the predicted intent label for the given text input.
    """
    def __init__(self, model_path="models/intentClassifier"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.labels = [
            "Course Information",
            "Enrollment / Course Registration",
            "Withdrawal or Drop Course",
            "Access Issues (portal/login)",
            "Technical Support",
            "Tuition/Fees Inquiry",
            "Scholarship/Financial Aid",
            "Mental Health Concerns",
            "Stress or Burnout",
            "Bullying or Harassment",
            "Administrative Support",
            "Campus Facilities",
            "Housing/Accommodation",
            "Extracurricular Activities",
            "General Complaint"
        ]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_class_id = outputs.logits.argmax().item()
        return self.labels[predicted_class_id]
