import pandas as pd

FILENAME = 'all-the-news-2-1.csv'
ROWSTOREAD = 100

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv(FILENAME, nrows=ROWSTOREAD)

df[['title', 'article']].to_csv('head.csv', index=False, encoding='utf-8')


""" Current prompt:
i'd like you to read each of the following titles and tell me if they are related or not related to working class issues. it shouldn't just be about things that happen to involve workers or come against workers, it needs to specifically be about labor-related issues. I'm looking for titles that are related to labor unions, collective action of working class people or the poor, and state suppression of working class people, not just titles that affect workers in general or are generally about states. i want you to take an extermely strict interpretation of this, it needs to be titles that meet these specific criteria and aren't borderline. provide me the index of each title that meets those criteria:

Given the following news article:
---
[Article Text]
---
Label whether this article is:
1. Related or Not Related to working-class issues.
2. If related, whether it is Positive or Negative toward workers.
Output format: {"relevance": "Related/Not Related", "sentiment": "Positive/Negative"}


# correct answers for 0-99: 6, 25, 40, 45, 89
conda activate pytorch_env
"""