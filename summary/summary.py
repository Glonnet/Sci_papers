import pandas as pd
from transformers import pipeline

data = pd.read_excel('/home/alex/code/Glonnet/Sci_papers/raw_data/papers.xlsx')
X = data['full-text']
y = data['abstract']


# Load the BART summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example: A long scientific passage
scientific_text = X[0]

# Generate a summary
summary = summarizer(scientific_text, max_length=100, min_length=30, do_sample=False)

print("Summary:", summary[0]['summary_text'])
