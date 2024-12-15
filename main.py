import pandas as pd
import re

FILENAME = 'all-the-news-2-1.csv'
OUTPUT_FILE = 'head.csv'
ROWSTOREAD = 10100
MAX_WORD_COUNT = 1000

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Remove trailing content with excessive line breaks or concatenated content
def clean_article_text(text):
    # Remove excessive line breaks or long sections of whitespace
    text = re.sub(r'\n{2,}', '\n', text)  # Replace multiple newlines with one
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with one
    return text.strip()

# Load the dataset
df = pd.read_csv(FILENAME, nrows=ROWSTOREAD)

# Keep only rows with non-empty values in 'title' and 'article'
df_cleaned = df[['title', 'article']].dropna()

# Remove rows where 'article' or 'title' are mostly whitespace or too short
df_cleaned = df_cleaned[
    df_cleaned['title'].str.strip().astype(bool) & 
    df_cleaned['article'].str.strip().astype(bool)
]

# Remove rows where articles are too long
df_cleaned = df_cleaned[df_cleaned['article'].str.split().str.len() < MAX_WORD_COUNT]

df_cleaned['article'] = df_cleaned['article'].apply(clean_article_text)

# Save the cleaned data to a new CSV file
df_cleaned.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')


""" Current prompt:
i'd like you to read each of the following titles and tell me if they are related or not related to working class issues. it shouldn't just be about things that happen to involve workers or come against workers, it needs to specifically be about labor-related issues. I'm looking for titles that are related to labor unions, collective action of working class people or the poor, and state suppression of working class people, not just titles that affect workers in general or are generally about states. i want you to take an extermely strict interpretation of this, it needs to be titles that meet these specific criteria and aren't borderline. provide me the index of each title that meets those criteria:

Given the following news article:
---
[Article Text]
---
Label whether this article is:
1. Related or Not Related to the following issues:
    - issue
    - issues
2. If related, whether it is Positive or Negative toward workers.
Output format: {"relevance": "Related/Not Related", "sentiment": "Positive/Negative"}


# correct answers for 0-99: 6, 25, 40, 45, 89
conda activate pytorch_env
"""