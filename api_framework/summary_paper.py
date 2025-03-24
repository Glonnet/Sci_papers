
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os





# Load the trained model and tokenizer
#tokenizer = AutoTokenizer.from_pretrained("Glonnet/contents")
# model = AutoModelForSeq2SeqLM.from_pretrained("Glonnet/contents")

local_model_path = "./models"  # Replace with your desired local path
# tokenizer.save_pretrained(local_model_path)
# model.save_pretrained(local_model_path)

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)

# Function to summarize text
def print_sum(text):


    def sum(text=text):
    # Tokenize the input text
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")

    # Generate the summary
        summary_ids = model.generate(inputs["input_ids"], max_length=1024, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


    def proper(sentenses):
        words  = sentenses.split(". ")
        new = ". ".join([word.capitalize() for word in words])
        return new
    sum_ = []
    for bloc in range(1, text.count('\n\n') + 1, 2):
        summary = sum(text.split('\n\n')[bloc]).split('[')[1].split("'")[1]
        summary_f = summary[:summary.rindex('.') + 1]
        sum_.append(text.split('\n\n')[bloc - 1] + "\n\n" + proper(summary_f) + '\n\n')
    return "\n\n" + "".join(sum_)
