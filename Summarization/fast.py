import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Summarization.summary_paper import sum, proper
from multiprocessing import Pool, cpu_count
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# def process_row(row):
#     """
#         Process a single row of the dataset.
#     """
#     full_text = row["full-text"]
#     summary = print_sum(full_text)
#     return summary

def sections (text):
    list_section = []
    for i in range(0, text.count('\n\n'), 2):
        dict_ = {}
        dict_[text.split('\n\n')[i]] = text.split('\n\n')[i+1]
        list_section.append(dict_)
    return list_section


def summarization(section):
    summary = sum("".join(section.values()))
    sum1 = summary[summary.find('[') + 1:]
    sum2 = sum1[sum1.find("'") + 1:]
    sum3 = sum2[sum2.find('].') + 1:]
    summary_f = sum3[:sum3.rindex('.') + 1]
    summary_final = proper(summary_f) + '\n\n'
    name = "".join(section.keys())
    return name + "\n\n" + summary_final

@app.get("/predict")
def predict():
    """
    Return summaries for all rows in the dataset using multiprocessing.
    """
    # Load the dataset
    df_new = pd.read_excel('raw_data/papers_test.xlsx')
    text = df_new['full-text'][0]
    list_section = sections(text)
    # Use multiprocessing to process each row in parallel
    with Pool(processes=cpu_count()) as pool:
        summaries = "".join(pool.map(summarization, list_section))

    # Return the summaries as a JSON response
    print(list_section)
    return {"Summaries": summaries}
@app.get("/")
def root():
    return {'Summarize': "Let's summarize!"}
