from openai import OpenAI
from dotenv import load_dotenv
import os
import requests

class AnswerGenerator:
    """
    Generates natural language answers to student queries using large language models (LLMs).

    This class supports integration with both OpenAI (e.g., GPT-4) and local LLMs (e.g., LLaMA via Ollama)
    to provide student support responses based on retrieved knowledge base context.

    The answers are generated in the role of a student success advisor, with a consistent prompt 
    guiding the model to use provided context and gracefully handle unknowns.

    Methods:
        generate_answer_with_openai(context, question, model):
            Uses OpenAI's ChatCompletion API to generate a response based on the given context and question.
        
        generate_answer_with_ollama(context, query, model):
            Sends a prompt to a locally hosted Ollama model and returns the generated response.
    """
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def generate_answer_with_openai(self, context: str, question: str, model: str = "gpt-4") -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful student success adivor. Use the provided context to answer the student's question. If the answer is not in the context, say 'Sorry, I'm not sure about that.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def generate_answer_with_ollama(self, context, query, model="llama3"):
        prompt = (
            "You are a helpful student success advisor. "
            "Use the provided context to answer the student's question. "
            "If the answer is not in the context, say 'Sorry, I'm not sure about that.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False  # Disable streaming for simplicity
            }
        )
        return response.json()['response'].strip()
