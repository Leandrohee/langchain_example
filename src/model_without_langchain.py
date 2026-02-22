from openai import OpenAI
from dotenv import load_dotenv
import os
import time


"""
This is a basic model on how to call the openAi client

It has to have a:
    1. prompt
    2. client
    3. answer 
"""

load_dotenv()


def modelWithoutLangChain():
    start_time = time.time()

    api_key = os.getenv("OPENAI_API_KEY")
    n_of_days = 10
    n_of_kids = 2
    place_of_the_trip = "New york"
    family_interest = "food"

    prompt = f"""
        Create a {n_of_days} days road map trip for a family with {n_of_kids} kids in
        {place_of_the_trip}. This family enjoys {family_interest}.
    """

    client = OpenAI(api_key=api_key)

    answer = client.chat.completions.create(
        # model= "gpt-5-nano",
        # model= "gpt-4.1-nano",
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a road map trip assistance"},
            {"role": "user", "content": prompt},
        ],
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    only_text = answer.choices[0].message.content
    # tokens_used = getattr(answer.usage, 'total_tokens', None)
    tokens_used = answer.usage and answer.usage.total_tokens

    print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds\n")
    # print(answer)
    print(only_text)
    print(f"Tokens used: {tokens_used}")
