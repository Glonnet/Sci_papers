import streamlit as st
import requests
import os

# Use the correct API URL (local or cloud deployment)
API_URL = os.getenv("https://api-summary-780137948407.europe-west1.run.app/")

st.title("Scientific Paper Summarizer")

# Input text area for the user
text = st.text_area("Enter the scientific paper content:")

if st.button("Summarize"):
    if text:
        # Send input to your FastAPI model
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            summary = response.json().get("summary", "No summary available")
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.error("Error: Failed to fetch summary from API.")
    else:
        st.warning("Please enter text to summarize.")
