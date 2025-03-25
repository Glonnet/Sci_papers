## Objective : Summarization of scientific papers

## Context

literature review is one of the first tasks carried out before conducting any research project. It enables researchers summaryzing previous research outcomes on a particular topic, which allow them to identify gaps in the current literature and bring a relationale for conducting a new research.

## Database

We will be using the Cord-19 dataset, which was created in 2020 during the pandemic by a collaboration between the white house and a coalition of research groups.

This database is freely available from kaggle:
[https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge]

The database includes several relevant features such as: **'title** of the scientific paper, **author names**, **abstract** and **full text content**.
each article content is provided as a JSON file, which include list of dictionaries whose keys and values are related to a specific information in the article. To adress this objective we will focus on the body text. Each paragraph content is formated as a text string and there is a possibility to match each text string with it's related section in the document.
Using python JSON library we will extract all the text strings from the first sentence of the introduction until the last sentence of the conclusion. We will then pull them togother to get a unique string.
