import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a professional RAG assistant.

Rules:
- Answer only using the provided context.
- If the context does not contain the answer, say:
  "I don't have enough information in the uploaded documents."
- Be clear and concise.
- Mention the source when possible.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You answer questions using retrieved document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content