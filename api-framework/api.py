from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

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
