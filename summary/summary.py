import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict


data = pd.read_excel('/home/alex/code/Glonnet/Sci_papers/raw_data/papers.xlsx')
X = data['full-text']
y = data['abstract']
# 1st article for testing purposes
scientific_text = X[0]

# Step 1: Split Data (80% Train, 10% Validation, 10% Test)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Step 2: Create model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")


def chunk_text(text, tokenizer, max_length=512):
    """Splits text into chunks without breaking sentences, ensuring token limits."""
    sentences = text.split('. ')  # Split by sentence
    chunks, current_chunk = [], []

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)

        # If adding this sentence exceeds max_length, save current chunk and start new one
        if len(current_chunk) + len(tokenized_sentence) > max_length:
            chunks.append(tokenizer.decode(current_chunk))  # Decode back to text
            current_chunk = tokenized_sentence  # Start new chunk
        else:
            current_chunk.extend(tokenized_sentence)

    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))  # Add last chunk

    return chunks

def summarize_text(text, model, tokenizer, max_length=20, min_length=5):
    """Summarizes text by processing it in chunks."""
    chunks = chunk_text(text, tokenizer)

    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=100, return_tensors="pt", truncation=True)

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

# Step 3: Create a Tokenizer
def preprocess_function(feature, target):

    # Tokenize input (scientific paper) and target (summary)
    source_ids = tokenizer(feature, truncation=True, padding="max_length", max_length=512) #might have to play with ml values
    target_ids = tokenizer(target, truncation=True, padding="max_length", max_length=128)

    # Replace pad token id with -100 for loss computation
    labels = target_ids["input_ids"]
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels]

    return {
        "input_ids": source_ids["input_ids"],
        "attention_mask": source_ids["attention_mask"],
        "labels": labels
    }

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "validation": Dataset.from_pandas(val_data),
    "test": Dataset.from_pandas(test_data),
})

# Step 4: Tokenize the Dataset
tokenized_dataset = dataset.map(lambda batch: preprocess_function(batch["full-text"], batch["abstract"]), batched=True)


# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="/home/alex/code/Glonnet/Sci_papers/summary/training.py",  # Replace with your output directory
    per_device_train_batch_size=8,
    num_train_epochs=2,  # Adjust number of epochs as needed
    remove_unused_columns=False
)


# Step 6: Create Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['val']
)

trainer.train()

# Step 7: Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation results
print(eval_results)

# Save the model and tokenizer after training
# model.save_pretrained("/home/alex/code/Glonnet/Sci_papers/summary/model.py")
# tokenizer.save_pretrained("/home/alex/code/Glonnet/Sci_papers/summary/model.py")

'''



# output = summarize_text(scientific_text, model, tokenizer)
# print(output)
'''
