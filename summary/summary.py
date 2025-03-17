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

#print("Summary:", summary[0]['summary_text'])



tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text, model, tokenizer, max_chunk_size=1024):
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=max_chunk_size, return_tensors="pt", truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=200,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    return " ".join(summaries)
summarize_text(scientific_text, model, tokenizer)
