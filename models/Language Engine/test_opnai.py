from openai import OpenAI
import os


dotenv_path = "../.env"

from dotenv import load_dotenv

load_dotenv(dotenv_path = dotenv_path)
print(os.getenv("OPENAI_API_KEY"))

client = OpenAI( api_key="-"   )

print(client.models.list())

