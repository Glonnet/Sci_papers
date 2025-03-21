import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Summarization.summary_paper import print_sum
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict():
    """
    return a summary from an input text file provided by the user
    """

    df_new = pd.read_excel('raw_data/papers_test.xlsx')

    full_text =df_new["full-text"][0]

    y_pred = print_sum(full_text)
    return {"Summary": y_pred}

@app.get("/")
def root():
    return {'Summarize': "Let's summarize!"}
