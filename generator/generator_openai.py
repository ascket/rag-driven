from openai import OpenAI
from data_models import OpenAIModels
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_AI_PAY_KEY"),
    organization=os.getenv("ORG_PAY_ID")
)
model = OpenAIModels.gpt_4o_mini

query = "define a rag store"


def call_llm_with_full_text(itext: str = query):
    text_input = '\n'.join(itext)
    prompt = f"Please elaborate on the following content: \n{text_input}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert Natural Language Processing exercise expert."},
            {"role": "assistant", "content": "1. You can explain read the input and answer in detail"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    print(call_llm_with_full_text())
