# Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Read Excel data file with 1000 articles
df = pd.read_excel('/home/caba/code/Glonnet/Sci_papers/raw_data/papers.xlsx')

# Display the first 5 rows of the dataframe

df.head()
