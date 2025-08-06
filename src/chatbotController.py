from src.handlers.answerGenerator import AnswerGenerator
from src.handlers.emotionClassifier import EmotionClassifier
from src.handlers.intentClassifier import IntentClassifier
from src.handlers.interactionLogger import InteractionLogger
from src.handlers.searchEngine import FaissSearchEngine
import concurrent.futures

class ChatbotController:
    """
    ChatbotController orchestrates the main components of the student-facing chatbot system.
    
    It integrates intent classification, emotion detection, vector-based knowledge retrieval,
    answer generation, and interaction logging. The controller processes incoming student queries
    by identifying their intent and emotional state in parallel. If distress is detected, it
    responds with a referral to a human advisor. Otherwise, it retrieves relevant context from the
    knowledge base and generates a user-friendly answer using a language model.
    """
    def __init__(self):
        self.intentClassifier = IntentClassifier()
        self.emotionClassifier = EmotionClassifier()
        self.answer_generator = AnswerGenerator()
        self.vector_search = FaissSearchEngine()
        self.logger = InteractionLogger()

    def get_knowledge_base(self, query):
        kbResults = self.vector_search.search(query, top_k=10)
        # Combine top-k chunks into a single context string
        context = "\n\n".join([f"{chunk['content']}" for chunk, _ in kbResults])
        if len(context) > 10000:
            context = context[:10000]
        return context

    def get_answer(self, query):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            emotionPrediction = executor.submit(self.emotionClassifier.predict, query)
            intentPrediction = executor.submit(self.intentClassifier.predict, query)

            emotion = emotionPrediction.result()
            intent = intentPrediction.result()

        # escalate to human agent
        if (emotion is not None and emotion in ["anger", "sadness", "fear", "disgust"]):
            # log
            self.logger.log('student_123',query,intent,emotion,"")
            return "I'm really sorry you're feeling this way. You don’t have to go through it alone. Please speak with a Student Success Advisor who can support you. You can book an appointment at <a href='https://collegeportal.edu/ssa-booking'>https://collegeportal.edu/ssa-booking</a> or call us directly at 555-123-4567."
        # get context from knowledge base
        context = self.get_knowledge_base(query)
        # Generate answer
        answer = self.answer_generator.generate_answer_with_ollama(context, query)
        # log
        self.logger.log('student_123',query,intent,emotion, answer)
        return answer
    
chatbot = ChatbotController()

def answer(question):
    print (f"{question}\n")
    answer = chatbot.get_answer(question)
    print(f"Answer:\n{answer}\n")

questions = [
    "Hi, I'm trying to figure out how to pay my tuition fees.",
    # "Thanks. Do I need to pay the full amount at once?",
    # "How do I make a payment?",
    # "What happens if I miss a payment?",
    # "Can I have an extension/ instalment plan of my payment due",
    # "Why has my payment not been posted to the Student Portal/ Can you confirm my payment",
    # "How do I pay for a course on Held Enrolment",
    # "How can apply for scholarship",
    # "any financial assistance available in Ontario",
    # "When is the last day to drop a course without penalty",
    # "Are there any upcoming student events",
    # "Where can I get help with my resume",
    # "is there a place I can do yoga",
    # "How to View my Timetable",
    # "I can't see my Timetable",
    # "How do I withdraw from my program",
    # "How do I change my block or add/drop a course",
#     "I feel overwhelmed and not sure if I can keep up this term"
]

for question in questions:
    answer(question)