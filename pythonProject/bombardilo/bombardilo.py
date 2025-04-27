import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#1st model
async def async_call_b_0(pathi: str, temperature: float = 0.0, model: str = "gpt-4o-2024-11-20") -> str:
    """Send a prompt to an LLM."""
    path = "/home/nazarii-yukhnovskyi/Documents/Personal/hackathons/med.AI/docs/Sample Patient Card.doc"

    res = ""
    with open(path, "rb") as f:
        res = f.read()
    prompt = """
    I can't read these bytes well. what is written here?""" + str(res)
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content" : prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


#2nd mode
# l

async def main():
    res = await async_call_b_0("")
    print(res)

asyncio.run(main())
