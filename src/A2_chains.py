from dotenv import load_dotenv
import os, time
from pydantic import Field, BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug

set_debug(True)

# VARIABLE
start_time = time.time()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# CLASSES
class Destiny(BaseModel):
    city: str = Field("The city recommended to visit")
    motive: str = Field("The motive for which is interesting to visit this city")


class Restaurant(BaseModel):
    city: str = Field("The city recommended to visit")
    restaurants: str = Field("The restaurants in this city")


# PARSERS
cityParser = JsonOutputParser(pydantic_object=Destiny)
restaurantParser = JsonOutputParser(pydantic_object=Restaurant)

# PROMPTS
promptCity = PromptTemplate(
    template="""
    Sugest a city knowing my interest for {interest}
    {exit_format}
    """,
    input_variables=["interest"],
    partial_variables={"exit_format": cityParser.get_format_instructions()},
)

promptRestaurant = PromptTemplate(
    template="""
    Sugest cools restaurants in this city: {city}
    {exit_format}
    """,
    input_variables=["city"],
    partial_variables={"exit_format": restaurantParser.get_format_instructions()},
)

promptCultural = PromptTemplate(
    template="""
    Sugest nice activities and cultural places in this city: {city}
    """,
    input_variables=["city"],
)

# MODEL
model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.5,  # This is how creative is the model (0,5 ~ 0,7 is good)
    api_key=SecretStr(api_key) if api_key else None,
)

# CHAINS
city_chain = promptCity | model | cityParser
restaurant_chain = promptRestaurant | model | restaurantParser
cultural_chain = promptCultural | model | StrOutputParser()
main_chain = city_chain | restaurant_chain | cultural_chain

# ANSWER
answer = main_chain.invoke({"interest": "beaches"})

# PRINTS
end_time = time.time()
elapsed_time = end_time - start_time
print(answer)
print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds\n")
