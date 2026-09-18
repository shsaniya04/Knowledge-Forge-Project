import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

document_context = """
KnowledgeForge is a fictional software project.
It was created to study LLM and RAG engineering.
The project currently supports document ingestion and semantic retrieval.
"""

user_question = "What features does KnowledgeForge currently support?"

prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Context:
{document_context}

Question:
{user_question}
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are a document-grounded assistant. "
            "Do not invent facts that are absent from the provided context."
        ),
        temperature=0.2,
        max_output_tokens=200,
    ),
)

print(response.text)