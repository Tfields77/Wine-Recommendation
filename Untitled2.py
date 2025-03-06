#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('/Users/boogie/Downloads/archive/winemag-data-130k-v2.csv')


# In[4]:


# Fill missing numeric columns with 0 and text (object) columns with "Unknown"
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

object_cols = df.select_dtypes(include=['object']).columns
df[object_cols] = df[object_cols].fillna("Unknown")


# In[5]:


df.head


# In[6]:


df['region_1'] =df['region_1'].fillna('Unknown')
#fill missing value with 'unknown' in region column


# In[7]:


df = df.dropna(subset=['price'])
#drop rows where the price is missing


# In[8]:


df.isna().sum()
#check for missing values


# In[9]:


df.loc[:, 'designation'] = df['designation'].fillna('unknown')
df.loc[:, 'region_2'] = df['region_2'].fillna('unknown')
df.loc[:, 'taster_name'] = df['taster_name'].fillna('unknown')
df.loc[:, 'taster_twitter_handle'] = df['taster_twitter_handle'].fillna('unknown')


# In[10]:


df = df.drop(columns=['Unnamed: 0']) 
#drop the unnamed column


# In[11]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
#preprocess the data using onehotencoder


# In[12]:


variety_encoded = encoder.fit_transform(df[['variety']])
region_encoded = encoder.fit_transform(df[['region_1']])
#one hot encode region and variety


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[14]:


tfidf = TfidfVectorizer(stop_words='english')
description_tfidf = tfidf.fit_transform(df['description'])
#tf-idf for description
                        


# In[15]:


import numpy as np


# In[16]:


import pandas as pd


# In[17]:


from sklearn.metrics.pairwise import cosine_similarity


# In[18]:


from scipy.sparse import hstack 


# In[19]:


import streamlit as st


# In[20]:


import streamlit as st
import gc
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, vstack

# CHANGED: now we show output with st.write, not print
st.title("Chunk-Based Similarity Example")

chunk_size = 200
total_rows = 129971  # total rows in dataset
current_chunk = 0

df = pd.read_csv('/Users/boogie/Downloads/archive/winemag-data-130k-v2.csv')
df = df.sample(frac=0.3, random_state=42)
total_rows = df.shape[0]

st.write("Total rows after sampling:", total_rows)

X_list = []  # to store feature matrices
meta_list = []  # to store metadata

# sample fraction for fitting
sample_frac = 0.05  
sample_variety = df[['variety']].sample(frac=sample_frac, random_state=42)
sample_region = df[['region_1']].sample(frac=sample_frac, random_state=42)
sample_description = df['description'].sample(frac=sample_frac, random_state=42)

pre_encoder_variety = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
pre_encoder_variety.fit(sample_variety)

pre_encoder_region = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
pre_encoder_region.fit(sample_region)

pre_tfidf = TfidfVectorizer(stop_words='english', max_features=500)
pre_tfidf.fit(sample_description)

def process_and_calculate_similarity(start_index):
    # load chunk rows
    chunk = df.iloc[start_index:start_index + chunk_size]

    # Transform
    variety_encoded = pre_encoder_variety.transform(chunk[['variety']])
    region_encoded = pre_encoder_region.transform(chunk[['region_1']])
    description_tfidf = pre_tfidf.transform(chunk['description'])
    
    X = hstack([variety_encoded, region_encoded, description_tfidf])
    similarity_matrix = cosine_similarity(X)
    
    X_list.append(X)
    meta_list.append(
        chunk[['title', 'description', 'taster_twitter_handle', 'variety', 'price', 'region_1']]
    )
    
    # Return the similarity matrix and chunk
    return similarity_matrix, chunk  

# Loop through dataset in chunks
recommendations = []
current_chunk = 0

while current_chunk * chunk_size < total_rows:
    similarity_matrix, chunk = process_and_calculate_similarity(current_chunk * chunk_size)
    
    # For each wine in this chunk, find its top similar wines
    for wine_index in range(chunk.shape[0]):
        similar_wines = similarity_matrix[wine_index]
        similar_wine_indices = similar_wines.argsort()[-6:-1][::-1]
        recommended_wines = chunk.iloc[similar_wine_indices]
        recommendations.append(recommended_wines[['title', 'variety', 'price', 'region_1']])
    
    current_chunk += 1
    del similarity_matrix, chunk
    gc.collect()

final_recommendations = pd.concat(recommendations, ignore_index=True)

st.write("Final Recommendations:")
st.dataframe(final_recommendations)

X_full = vstack(X_list)
meta_full = pd.concat(meta_list, ignore_index=True)

st.write("X_full shape:", X_full.shape)
st.write("meta_full shape:", meta_full.shape)


# In[21]:


import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Adjust pandas display options for better readability (optional)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
pd.set_option('display.max_colwidth', 120)

def ask_branch_questions():
    # REPLACED input() with Streamlit widgets
    branch = st.selectbox(
        "Select wine type:",
        ["red", "white", "sparkling"]
    )
    
    preference_string = ""

    if branch == "red":
        flavor_answer = st.radio(
            "For red wines, what flavor profile do you prefer?",
            ["dry tannic", "fruity juicy", "spicy robust", "earthy nuanced"]
        )
        preference_string += flavor_answer + " "

        body_answer = st.radio(
            "Which body do you enjoy most?",
            ["light-bodied", "medium-bodied", "full-bodied"]
        )
        preference_string += body_answer + " "

    elif branch == "white":
        taste_answer = st.radio(
            "For white wines, what taste profile appeals to you?",
            ["crisp dry", "lightly sweet", "rich oaky"]
        )
        preference_string += taste_answer + " "
        
        acid_answer = st.radio(
            "What acidity level do you enjoy?",
            ["high acidity", "moderate acidity", "low acidity"]
        )
        preference_string += acid_answer + " "

    elif branch == "sparkling":
        sweet_answer = st.radio(
            "For sparkling wines, what sweetness do you prefer?",
            ["brut dry", "extra brut", "demi-sec off-dry sweet"]
        )
        preference_string += sweet_answer + " "
        
        occasion_answer = st.radio(
            "For which occasion are you selecting sparkling wine?",
            ["casual everyday", "celebratory", "special gourmet pairing"]
        )
        preference_string += occasion_answer + " "

    else:
        st.write("Invalid branch selected. Defaulting to red wine preferences.")
        branch = "red"
        preference_string = "dry tannic full-bodied "

    return branch, preference_string.strip()


# -----------------------------
# Minimal usage example
# -----------------------------

# 1. Ask for User Preferences
selected_branch, user_query_text = ask_branch_questions()

st.write("Selected branch:", selected_branch)
st.write("User preference summary:", user_query_text)

# 2. Compute Similarity Scores
# (Assumes you already loaded and fitted pre_tfidf, X_full, meta_full somewhere)
# e.g., pre_tfidf = ...
#       X_full = ...
#       meta_full = ...
# This is just a placeholder to show how you'd display the results in Streamlit.

# We'll pretend the user has clicked a button to see recommendations:
if st.button("Show Recommendations"):
    user_query_vector = pre_tfidf.transform([user_query_text])

    tfidf_feature_count = len(pre_tfidf.get_feature_names_out())
    X_tfidf_full = X_full[:, -tfidf_feature_count:]

    similarity_scores = cosine_similarity(user_query_vector, X_tfidf_full).flatten()
    meta_full['query_similarity'] = similarity_scores

    nlp_recommendations = meta_full.sort_values(by='query_similarity', ascending=False)

    # 3. Display Top 5, Omitting 'taster_name'
    all_cols = list(nlp_recommendations.columns)
    if 'taster_name' in all_cols:
        all_cols.remove('taster_name')

    st.write("Top 5 Wine Recommendations Based on Your Preferences (Excluding 'taster_name'):")
    top5 = nlp_recommendations.head(5)[all_cols]
    st.dataframe(top5)


# In[22]:





# In[ ]:




