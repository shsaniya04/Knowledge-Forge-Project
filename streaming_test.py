import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

client = genai.Client(api_key=api_key)

stream = client.models.generate_content_stream(
    model="gemini-3.6-flash",
    contents= "Explain what machine learning for beginner."
)

for chunk in stream: 
    print(chunk.text, end = "", flush = True)