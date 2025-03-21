import numpy as np
import pandas as pd
import os
path = os.getcwd() + '/Summarization/papers.xlsx'
df = pd.read_excel(path)

df_summary = df[df["paper_id"] == "1c3021528dea5b342a90fa28d3a4477315602eb9"]

# extract full paper text as as string
full_text = "".join(df_summary["full-text"].values)

from transformers import BartTokenizer, TFBartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large")

def summarize_text(text, model, tokenizer, max_chunk_size=1024):
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=max_chunk_size, return_tensors="tf", truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=100,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
            )
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    return " ".join(summaries)

for bloc in range(1, full_text.count('\n\n') + 1, 2):
    summary = summarize_text(full_text.split('\n\n')[bloc], model, tokenizer)
    print(full_text.split('\n\n')[bloc - 1] + ": " + summary)
