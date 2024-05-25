import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from utils import OPENAI_API_KEY, SYSTEM_PROMPT, USER_PROMPT

import logging
import pandas as pd
from langchain_community.chat_models import ChatOllama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def split_csv(input_file, rows_per_file, output_prefix):
    # Read the CSV file
    try:
        data = pd.read_csv(input_file)
        logging.info(f"Successfully read {input_file}")
    except Exception as e:
        logging.error(f"Error reading {input_file}: {e}")
        return

    # Determine number of splits needed
    total_rows = data.shape[0]
    total_splits = (total_rows + rows_per_file - 1) // rows_per_file
    logging.info(
        f"Total rows in file: {total_rows}. Rows per split: {rows_per_file}. Total splits needed: {total_splits}"
    )

    # Split the file and write to new CSV files
    for i in range(total_splits):
        start_row = i * rows_per_file
        end_row = min(start_row + rows_per_file, total_rows)
        output_file = f"{output_prefix}_{i+1}.csv"

        # Save the slice of the dataframe to a new CSV file
        data[start_row:end_row].to_csv(output_file, index=False)
        logging.info(f"Written {output_file} with rows {start_row+1} to {end_row}")


def classify_news_categories(text: str, model_name: str):
    logging.info(f"Starting classification for news text {text[:20]}...")
    try:
        # model = ChatGroq(temperature=0.1, api_key=GROQ_API_KEY, model_name=model_name)
        if model_name == "gpt-3.5-turbo":
            model = ChatOpenAI(model=model_name, temperature=0.1, api_key=OPENAI_API_KEY)
        else:
            model = ChatOllama(model=model_name, temperature=0.1, max_tokens=20)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        )
        chain = prompt | model | StrOutputParser()
        result = chain.invoke(
            {"text_to_be_classified": text}
        )  
        # Convert result to lowercase
        result = result.casefold()
        logging.info(
            f"Classification for news text {text[:20]} completed successfully. Result: {result}"
        )
        return result

    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return "error"

def combine_dataframes(dataframes, axis=0, join='outer'):
    """
    Combines multiple DataFrames into a single DataFrame.

    Args:
        dataframes (list): A list of DataFrames to be combined.
        axis (int, optional): The axis to concatenate along. 0 for rows, 1 for columns. Default is 0.
        join (str, optional): How to handle indexes on other axis. 'outer' for union, 'inner' for intersection. Default is 'outer'.

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    combined_df = pd.concat(dataframes, axis=axis, join=join)
    return combined_df


def compute_multiclass_metrics(y_true, y_pred, average='macro'):
    """
    Computes various multi-class classification metrics.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        average (str): Averaging method for precision, recall, and F1-score.
                       Can be 'micro', 'macro', 'weighted', or None.
    
    Returns:
        dict: Dictionary containing the computed metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # 'confusion_matrix': cm
    }
    
    return metrics

def extract_word(text):
    # Replace "\n" or "\n\n" with a space
    if len(text.split()) > 2:
        text = re.sub(r'\\n\\n?', ' ', text).replace("\"", "").replace(".", "")
        
        # Extract the word after ": "
        match = re.search(r':\s*(\w+)', text)
        if match:
            return match.group(1)
        
        # Extract the first occurrence of a word in the specified categories
        categories = ['politics', 'sport', 'tech', 'business', 'entertainment']
        for category in categories:
            match = re.search(fr'{category}:\s*(\w+)', text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Convert the sentence to lowercase and split it into words
        words = text.lower().split()
        
        # Iterate over each word in the sentence
        for word in words:
            # Check if the word belongs to any of the categories
            if word in categories:
                return word
    
    return text.strip()

def get_most_common_word(row):
    # Convert the row to a list
    word_list = row.tolist()
    
    # Convert the list to a string
    text = ' '.join(str(word) for word in word_list)
    
    # Split the text into individual words
    words = text.split()
    
    # Count the frequency of each word
    word_counts = Counter(words)
    
    # Get the most common word
    most_common_word = max(word_counts, key=word_counts.get)
    
    return most_common_word

def plot_confusion_matrix(true_category, predicted_category, category_name):
    cm = confusion_matrix(true_category, predicted_category)
    labels = sorted(set(true_category))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {category_name}')
    plt.show()