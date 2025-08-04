import csv
from datetime import datetime
import os

class InteractionLogger:
    def __init__(self, log_file='logs/log.csv'):
        self.log_file = log_file
        # Create the file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Datetime', 'Student', 'Question', 'Intent', 'Emotion'])

    def log(self, student, question, intent, emotion):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([now, student, question, intent, emotion])
