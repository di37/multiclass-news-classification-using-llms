import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from dotenv import load_dotenv

_ = load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SYSTEM_PROMPT = """You are an AI assistant trained to classify news into one of the following categories: sport, business, politics, tech, or entertainment. You will be provided with a line or paragraph of news in the form of string text, and your task is to analyze the content and classify it into the most appropriate category accordingly.

# Classify the text as:
# - sport: If the text is related to sports, athletes, sporting events, or sports-related topics.
# - business: If the text discusses business, finance, economy, companies, or related subjects.
# - politics: If the text is about politics, government, policies, elections, or political issues.
# - tech: If the text focuses on technology, gadgets, software, startups, or tech-related matters.
# - entertainment: If the text is related to movies, TV shows, music, celebrities, or entertainment industry topics."""

SYSTEM_PROMPT = """You are an AI model trained to classify news articles. Follow this structured approach to ensure accurate classification:

1. Read and analyze the content of the input text thoroughly.
2. Identify key terms and themes that signify one of the categories:
   - Sport: sports, athletes, teams, games, matches, competitions, scores.
   - Tech: technology, computers, software, internet, mobile apps, gadgets, innovations.
   - Business: companies, markets, finance, economics, trade, investing, industries, commerce.
   - Entertainment: movies, TV, music, celebrities, pop culture, arts.
   - Politics: government, elections, policies, legislation, political parties, world leaders, geopolitics.
3. Use only the choices above and not your internal knowledge.
4. Produce the classification result in the format: "Output: <category>."

Focus on the main theme and ignore minor references to other topics. Output only one category label per text."""

with open("./data/news_category_classification/user_prompt_template2.txt", "r") as f:
    USER_PROMPT = f.read()
