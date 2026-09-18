import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=(
        "Extract the following information about machine learning: "
        "definition, two applications, and one limitation. "
        "Return only a JSON object."
    ),
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    ),
)

print(response.text)

# Optional: Parse the returned JSON
data = json.loads(response.text)
print(data)