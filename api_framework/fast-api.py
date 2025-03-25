import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_framework.summary_paper import print_sum
from fastapi import FastAPI, UploadFile, File
from typing import Annotated
from io import BytesIO

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict_test")
def predict():
    """
    return a summary from an input text file provided by the user
    """

    df_new = pd.read_excel('raw_data/papers_test.xlsx')

    full_text =df_new["full-text"][0]

    y_pred = print_sum(full_text)
    return {"Summary": y_pred}

@app.post("/predict")
async def create_file(
    myfile: Annotated[UploadFile, File()]):

    contents = await myfile.read()
    df = pd.read_pickle(BytesIO(contents))
    full_txt = df['full-text'].values[0]

    y_pred = print_sum(full_txt)
    return {"Summary": y_pred}

# class Item(BaseModel):
#     name: str
#     description: str | None = None
#     price: float
#     tax: float | None = None


# @app.post("/items/")
# async def create_item(item: Item):
#     return item

@app.get("/")
def root():
    return {'Summarize': "Let's summarize!"}
