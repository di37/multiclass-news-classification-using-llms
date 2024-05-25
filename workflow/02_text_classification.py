import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import os
import pandas as pd
from utils import classify_news_categories
import logging
import glob

# Assuming the classify_news_categories function is already defined elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting classification...")
models = ['llama3', 'llama3:70b', 'mixtral', 'gpt-3.5-turbo']
for model_name in models:
    for file in glob.glob("data/news_category_classification/test-examples_*.csv"):
        try:
            df = pd.read_csv(file)
            logging.info(f"Classifying {file} using model: {model_name}")
            df[f'predicted_category_{model_name}'] = df['text'].apply(
                lambda x: classify_news_categories(text=x, model_name=model_name)
            )
            df.to_csv(file, index=False)
            logging.info(f"Done classifying {file}.")
        except Exception as e:
            logging.error(f"Failed to classify {file} with model {model_name}. Error: {e}")
logging.info("Classification completed.")