import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

texts = [
    "How do I train a machine learning model?",
    "What are the steps for teaching an AI system?",
    "What is photosynthesis in plants?",
]

for text in texts:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )

    embedding = result.embeddings[0].values

    print("\nText:", text)
    print("Vector dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])