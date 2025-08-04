import torch
from transformers import BertTokenizer, BertForSequenceClassification

class IntentClassifier():
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


# intentClassifier = IntentClassifier()
# print(intentClassifier.predict("Hi, I'm trying to figure out how to pay my tuition fees."))
# print(intentClassifier.predict("When is the tuition payment deadline"))
# print(intentClassifier.predict("Are there any upcoming student events"))