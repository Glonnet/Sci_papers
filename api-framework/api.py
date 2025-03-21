from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd

'''
# Initialize FastAPI app
app = FastAPI()

# Define request model
class TextRequest(BaseModel):
    text: str

# Load your fine-tuned model & tokenizer
# Option 1 - The model is saved locally on your computer
# MODEL_PATH = "/home/alex/code/Glonnet/Sci_papers/summary/trained_model"  # Path to your saved model
# tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
# model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

# Option 2 - The model is saved inside a bucket on the cloud
MODEL_PATH = "gs://your-bucket-name/trained_model"
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)


@app.post("/summarize/")
async def summarize_text(request: TextRequest):
    text = request.text
    if not text:
        return {"error": "No text provided"}

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"summary": summary}
'''

'''
# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Glonnet/contents")
model = AutoModelForSeq2SeqLM.from_pretrained("Glonnet/contents")
# Function to summarize text
def sum(text):
    # Tokenize the input text
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=1024, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
df_new = pd.read_excel('raw_data/papers_test.xlsx')
full_text =df_new["full-text"][0]
full_text
def proper(sentenses):
    words=sentenses.split(". ")
    new=". ".join([word.capitalize() for word in words])
    return new
for bloc in range(1, full_text.count('\n\n') + 1, 2):
    summary = sum(full_text.split('\n\n')[bloc]).split('[')[1].split("'")[1]
    summary_f = summary[:summary.rindex('.') + 1]
    print(full_text.split('\n\n')[bloc - 1] + "\n\n" + proper(summary_f) + '\n\n')
'''
