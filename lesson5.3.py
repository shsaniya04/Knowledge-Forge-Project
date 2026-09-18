import os

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

embedding_model = "gemini-embedding-001"


def get_embedding(text):
    result = client.models.embed_content(
        model=embedding_model,
        contents=text,
    )

    return np.array(result.embeddings[0].values)


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


texts = [
    "How do I train a machine learning model?",
    "What are the steps for teaching an AI system?",
    "What is photosynthesis in plants?",
]

embeddings = [get_embedding(text) for text in texts]

print("Similarity between Text 1 and Text 2:",
      cosine_similarity(embeddings[0], embeddings[1]))

print("Similarity between Text 1 and Text 3:",
      cosine_similarity(embeddings[0], embeddings[2]))