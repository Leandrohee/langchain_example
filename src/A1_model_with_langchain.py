from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import time
from pydantic import SecretStr


"""
This is a basic example on how to use a simple langchain model

It has to have a:
    1. prompt
    2. model 
"""

load_dotenv()


def modelWithLangchain():
    start_time = time.time()

    api_key = os.getenv("OPENAI_API_KEY")
    n_of_days = 5
    n_of_kids = 0
    place_of_the_trip = "Chicago"
    family_interest = "sports"

    prompt = f"""
        Create a {n_of_days} days road map trip for a family with {n_of_kids} kids in
        {place_of_the_trip}. This family enjoys {family_interest}.
    """

    model = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.5,  # This is how creative is the model (0,5 ~ 0,7 is good)
        api_key=SecretStr(api_key) if api_key else None,
    )

    answer = model.invoke(prompt)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds\n")

    only_text = answer.content
    tokens_used = answer.response_metadata["token_usage"]["total_tokens"]

    print(only_text)
    print(f"Tokens used: {tokens_used}")
