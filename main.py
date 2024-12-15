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

Given the following news article:
---
[Article Text]
---
Label whether each article is:
1. Related or Not Related to the one or more of the following issues, using a strict interpretation:
- collective bargaining
- unionization efforts
- wage disputes
- minimum wage policies
- right-to-work laws
- fair labor standards
- workers' rights
- pension reforms
- contract negotiations
- walkouts
- picket lines
- labor strikes
- work stoppages
- demonstrations by workers
- labor unrest
- organized protests
- wage theft
- unsafe working conditions
- sweatshops
- overtime abuse
- forced labor
- labor trafficking
- underpaid workers
- worker misclassification
- anti-union campaigns
- union busting
- corporate lobbying against labor
- deregulation of labor protections
- monopolistic practices impacting workers
- austerity measures targeting the poor
- privatization of public resources
- minimum wage freezes
- wealth inequality
- economic justice
- poverty rates
- access to healthcare for workers
- rising cost of living and wages
- housing crises affecting the working class
- taxation policies impacting the poor
- police crackdowns on protests
- surveillance of union activities
- anti-protest laws
- arrests of union leaders
- military action against strikers
- gig economy labor rights
- platform worker protests
- remote work policies for low-income workers
- AI and automation replacing jobs
- climate change affecting labor

Output format: {"relevance": "Related/Not Related"}


# correct answers for 0-99: 6, 25, 40, 45, 89
conda activate pytorch_env
"""