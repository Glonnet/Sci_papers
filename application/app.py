import streamlit as st
import requests
import os
import pandas as pd

url = 'http://0.0.0.0:8000'
#url = 'https://api-summary-780137948407.europe-west1.run.app'

st.title("Scientific paper summarizer")
st.header("Please upload an Excel file: ")
file_buffer = st.file_uploader('')
df = pd.read_excel('raw_data/papers_test.xlsx')

# Banner that rolls down with document options
with st.expander("📂 Click to View and Select Documents"):
    # List of available documents

    documents = [f"{title}" for title in df['title']]

    # Multi-select allows the user to choose one document
    selected_doc = st.selectbox("Choose a document to summarize:", documents)
    #df.loc[selected_doc.split(':')[1]]
    doc = str(selected_doc.strip())
    selection = df[df['title'] == doc]
    st.subheader(df['title'][df['title'] == doc].values[0])
    st.text("\n\n" + str(df['full-text'][df['title'] == doc].values[0]))


    selection.to_pickle("article.pkl")



if selection is not None:
    if st.button("Summarize"):
        #res = requests.post(url + "/predict", files={'myfile': selection}).json()
        #res = requests.post(url + "/predict", json={'myfile': str(selection)}).json()
        with open('article.pkl', 'rb') as f:
            response = requests.post(
                url + "/predict",
                files={'myfile': f}
            ).json()

        st.write(response)








# # Use the correct API URL (local or cloud deployment)
# API_URL = os.getenv("https://api-summary-780137948407.europe-west1.run.app/")

# st.title("Scientific Paper Summarizer")

# # Input text area for the user
# text = st.text_area("Enter the scientific paper content:")

# if st.button("Summarize"):
#     if text:
#         # Send input to your FastAPI model
#         response = requests.post(API_URL, json={"text": text})
#         if response.status_code == 200:
#             summary = response.json().get("summary", "No summary available")
#             st.subheader("Summary:")
#             st.write(summary)
#         else:
#             st.error("Error: Failed to fetch summary from API.")
#     else:
#         st.warning("Please enter text to summarize.")
