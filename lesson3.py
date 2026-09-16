import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents= "Write a creative two-line description of a rainy evening.",
    config = types.GenerateContentConfig(
        temperature = 0.5,
        max_output_tokens = 1000, 
        top_p = 0.9, 
        top_k = 5
    )
)

print("Gemini: ", response.text)