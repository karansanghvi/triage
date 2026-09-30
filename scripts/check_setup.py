import os
from dotenv import load_dotenv

# Reads the .env file in the project root and loads its values 
load_dotenv()

KEYS = [
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_BASE_URL",
]

print("Checking environment setup...\n")
for name in KEYS:
    value = os.getenv(name)
    status = "set" if value else "missing"
    print(f" {name}: {status}")