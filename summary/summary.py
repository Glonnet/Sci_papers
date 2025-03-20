import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import wandb
import evaluate
#import os
#os.environ["WANDB_DISABLED"] = "true"
wandb.init(mode="offline")

'''
To run this code, you might need to:
pip install openpyxl
pip install datasets
pip install wandb
pip install evaluate
pip install rouge_score
pip install 'accelerate>=0.26.0'
'''


# Step 1: Get the data
data = pd.read_excel('/home/alex/code/Glonnet/Sci_papers/raw_data/papers.xlsx')
X = data['full-text']
y = data['abstract']


# Step 2: Create model, tokenizer + chunk and summarizing functions
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
        inputs = tokenizer(chunk, max_length=100, return_tensors="tf", truncation=True)

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
def preprocess_function(batch):
  source = [str(i) for i in batch[['full-text']].values]
  target = [str(i) for i in batch[['abstract']].values]

  source_ids = tokenizer(source, max_length=128, padding='max_length', truncation=True)
  target_ids = tokenizer(target, max_length=128, padding='max_length', truncation=True)
  labels = target_ids["input_ids"]
  labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
  return {"input_ids": source_ids["input_ids"], "attention_mask": source_ids["attention_mask"], "labels": labels}

'''
'input_ids' represent the tokenized form of your input text. Each token (which could be a word or part of a word) is converted into a unique integer ID based on the model's vocabulary.

'attention_mask' is a tensor that indicates which tokens should be attended to and which should be ignored (usually padding tokens). It’s a binary mask where typically:
'''

# Step 4: Convert DataFrame appropriately
tokenized_dataset = preprocess_function(data)
token_df = pd.DataFrame(tokenized_dataset, columns=['input_ids', 'attention_mask', 'labels'], index=data.index)
df_source = pd.concat([data, token_df], axis=1, join='inner').reset_index(drop=True)

# Step 5: Split Data (80% Train, 10% Test)
train_data, test_data = train_test_split(df_source, test_size=0.2, random_state=42)

# Step 6: Reshape DataFrames


train_ds = Dataset.from_pandas(train_data).remove_columns(['__index_level_0__'])
test_ds = Dataset.from_pandas(test_data).remove_columns(['__index_level_0__'])

# Step 7: Implement ROUGE-score function
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return result

# Step 8: Define training arguments
training_args = TrainingArguments(
    output_dir="/home/alex/code/Glonnet/Sci_papers/summary/training.py",  # Replace with your output directory
    run_name="scientific_paper_summarization",
    per_device_train_batch_size=8,
    num_train_epochs=2,  # Adjust number of epochs as needed
    learning_rate=3e-5,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    remove_unused_columns=False
)
'''
Best score obtained so far with current metrics:
training_loss=2.445428466796875
'eval_loss': 2.862306594848633
'''

# Step 9: Create Trainer Object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

# Step 10: Train the Model
trainer.train()

# Step 11: Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation results
print(eval_results)

# Save the model and tokenizer after training
# model.save_pretrained("/home/alex/code/Glonnet/Sci_papers/summary/model.py")
# tokenizer.save_pretrained("/home/alex/code/Glonnet/Sci_papers/summary/model.py")
