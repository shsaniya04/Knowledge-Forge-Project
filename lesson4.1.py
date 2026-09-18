import os 

from dotenv import load_dotenv
from google import genai 
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key: 
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model = "gemini-3.6-flash", 
    contents = "Explain overfitting in machine learning.", 
    config = types.GenerateContentConfig(
        system_instruction = (
            "You are a ML Tutor."
            "Explain concepts to a beginner using simple language."
            "Include one practical example."
        ), 
        temperature = 0.2, 
        max_output_tokens = 5000
    )
)

print(response.text)