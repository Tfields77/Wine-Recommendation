{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6cb6a79a-1394-4620-9603-5767283d8259",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "32b95b01-298e-40d7-b0ea-18e32ef8a242",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6e18ca4d-57da-4caf-9ce2-a6fbc352eb29",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('/Users/boogie/Downloads/archive/winemag-data-130k-v2.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0105005b-1c7a-4053-abeb-1d728ce47d69",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill missing numeric columns with 0 and text (object) columns with \"Unknown\"\n",
    "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
    "df[numeric_cols] = df[numeric_cols].fillna(0)\n",
    "\n",
    "object_cols = df.select_dtypes(include=['object']).columns\n",
    "df[object_cols] = df[object_cols].fillna(\"Unknown\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "27fe6967-1bf2-45dd-9451-7d24ac44cd17",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method NDFrame.head of         Unnamed: 0   country  \\\n",
       "0                0     Italy   \n",
       "1                1  Portugal   \n",
       "2                2        US   \n",
       "3                3        US   \n",
       "4                4        US   \n",
       "...            ...       ...   \n",
       "129966      129966   Germany   \n",
       "129967      129967        US   \n",
       "129968      129968    France   \n",
       "129969      129969    France   \n",
       "129970      129970    France   \n",
       "\n",
       "                                              description  \\\n",
       "0       Aromas include tropical fruit, broom, brimston...   \n",
       "1       This is ripe and fruity, a wine that is smooth...   \n",
       "2       Tart and snappy, the flavors of lime flesh and...   \n",
       "3       Pineapple rind, lemon pith and orange blossom ...   \n",
       "4       Much like the regular bottling from 2012, this...   \n",
       "...                                                   ...   \n",
       "129966  Notes of honeysuckle and cantaloupe sweeten th...   \n",
       "129967  Citation is given as much as a decade of bottl...   \n",
       "129968  Well-drained gravel soil gives this wine its c...   \n",
       "129969  A dry style of Pinot Gris, this is crisp with ...   \n",
       "129970  Big, rich and off-dry, this is powered by inte...   \n",
       "\n",
       "                                   designation  points  price  \\\n",
       "0                                 Vulkà Bianco      87    0.0   \n",
       "1                                     Avidagos      87   15.0   \n",
       "2                                      Unknown      87   14.0   \n",
       "3                         Reserve Late Harvest      87   13.0   \n",
       "4           Vintner's Reserve Wild Child Block      87   65.0   \n",
       "...                                        ...     ...    ...   \n",
       "129966  Brauneberger Juffer-Sonnenuhr Spätlese      90   28.0   \n",
       "129967                                 Unknown      90   75.0   \n",
       "129968                                   Kritt      90   30.0   \n",
       "129969                                 Unknown      90   32.0   \n",
       "129970           Lieu-dit Harth Cuvée Caroline      90   21.0   \n",
       "\n",
       "                 province             region_1           region_2  \\\n",
       "0       Sicily & Sardinia                 Etna            Unknown   \n",
       "1                   Douro              Unknown            Unknown   \n",
       "2                  Oregon    Willamette Valley  Willamette Valley   \n",
       "3                Michigan  Lake Michigan Shore            Unknown   \n",
       "4                  Oregon    Willamette Valley  Willamette Valley   \n",
       "...                   ...                  ...                ...   \n",
       "129966              Mosel              Unknown            Unknown   \n",
       "129967             Oregon               Oregon       Oregon Other   \n",
       "129968             Alsace               Alsace            Unknown   \n",
       "129969             Alsace               Alsace            Unknown   \n",
       "129970             Alsace               Alsace            Unknown   \n",
       "\n",
       "               taster_name taster_twitter_handle  \\\n",
       "0            Kerin O’Keefe          @kerinokeefe   \n",
       "1               Roger Voss            @vossroger   \n",
       "2             Paul Gregutt           @paulgwine    \n",
       "3       Alexander Peartree               Unknown   \n",
       "4             Paul Gregutt           @paulgwine    \n",
       "...                    ...                   ...   \n",
       "129966  Anna Lee C. Iijima               Unknown   \n",
       "129967        Paul Gregutt           @paulgwine    \n",
       "129968          Roger Voss            @vossroger   \n",
       "129969          Roger Voss            @vossroger   \n",
       "129970          Roger Voss            @vossroger   \n",
       "\n",
       "                                                    title         variety  \\\n",
       "0                       Nicosia 2013 Vulkà Bianco  (Etna)     White Blend   \n",
       "1           Quinta dos Avidagos 2011 Avidagos Red (Douro)  Portuguese Red   \n",
       "2           Rainstorm 2013 Pinot Gris (Willamette Valley)      Pinot Gris   \n",
       "3       St. Julian 2013 Reserve Late Harvest Riesling ...        Riesling   \n",
       "4       Sweet Cheeks 2012 Vintner's Reserve Wild Child...      Pinot Noir   \n",
       "...                                                   ...             ...   \n",
       "129966  Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...        Riesling   \n",
       "129967                  Citation 2004 Pinot Noir (Oregon)      Pinot Noir   \n",
       "129968  Domaine Gresser 2013 Kritt Gewurztraminer (Als...  Gewürztraminer   \n",
       "129969      Domaine Marcel Deiss 2012 Pinot Gris (Alsace)      Pinot Gris   \n",
       "129970  Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...  Gewürztraminer   \n",
       "\n",
       "                                          winery  \n",
       "0                                        Nicosia  \n",
       "1                            Quinta dos Avidagos  \n",
       "2                                      Rainstorm  \n",
       "3                                     St. Julian  \n",
       "4                                   Sweet Cheeks  \n",
       "...                                          ...  \n",
       "129966  Dr. H. Thanisch (Erben Müller-Burggraef)  \n",
       "129967                                  Citation  \n",
       "129968                           Domaine Gresser  \n",
       "129969                      Domaine Marcel Deiss  \n",
       "129970                          Domaine Schoffit  \n",
       "\n",
       "[129971 rows x 14 columns]>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "86e8aa43-ce1a-4a57-8418-f4e66797c53d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['region_1'] =df['region_1'].fillna('Unknown')\n",
    "#fill missing value with 'unknown' in region column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "11601071-c82d-4a3c-945a-0badd74ddf29",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropna(subset=['price'])\n",
    "#drop rows where the price is missing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2f0286e0-5b77-41c4-8260-6980faf49220",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Unnamed: 0               0\n",
       "country                  0\n",
       "description              0\n",
       "designation              0\n",
       "points                   0\n",
       "price                    0\n",
       "province                 0\n",
       "region_1                 0\n",
       "region_2                 0\n",
       "taster_name              0\n",
       "taster_twitter_handle    0\n",
       "title                    0\n",
       "variety                  0\n",
       "winery                   0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()\n",
    "#check for missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8abbc559-bac0-4195-ac87-86ec4da3b8ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[:, 'designation'] = df['designation'].fillna('unknown')\n",
    "df.loc[:, 'region_2'] = df['region_2'].fillna('unknown')\n",
    "df.loc[:, 'taster_name'] = df['taster_name'].fillna('unknown')\n",
    "df.loc[:, 'taster_twitter_handle'] = df['taster_twitter_handle'].fillna('unknown')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "50d42622-da2a-4108-9757-0b6bb6999061",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(columns=['Unnamed: 0']) \n",
    "#drop the unnamed column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "bc480adf-0f85-477f-ab2a-52cf31441e2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import OneHotEncoder\n",
    "encoder = OneHotEncoder(sparse_output=False)\n",
    "#preprocess the data using onehotencoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7d414f15-ea55-4a81-8960-29ba9e6cac1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "variety_encoded = encoder.fit_transform(df[['variety']])\n",
    "region_encoded = encoder.fit_transform(df[['region_1']])\n",
    "#one hot encode region and variety"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "1a9420aa-798f-4402-8631-e8c3d70ffba5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "09551944-c4fa-48e1-b016-2f30686b1f54",
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf = TfidfVectorizer(stop_words='english')\n",
    "description_tfidf = tfidf.fit_transform(df['description'])\n",
    "#tf-idf for description\n",
    "                        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e46e1ad7-0919-4ea9-baa7-1dbbaaf1febe",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3311ae75-360a-4378-bcc5-b29df98a5347",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d706df5a-6fbc-4eaf-8939-2035653bde5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics.pairwise import cosine_similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "4e30155b-9087-4a0f-ae6d-4dbc53fc81ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.sparse import hstack "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "4bd6289a-f3f3-4405-b54e-cfefb0ceed7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "c16eba45-8ed9-4a3b-b960-33f247623c50",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-06 17:21:42.820 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import gc\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from scipy.sparse import hstack, vstack\n",
    "\n",
    "# CHANGED: now we show output with st.write, not print\n",
    "st.title(\"Chunk-Based Similarity Example\")\n",
    "\n",
    "chunk_size = 200\n",
    "total_rows = 129971  # total rows in dataset\n",
    "current_chunk = 0\n",
    "\n",
    "df = pd.read_csv('/Users/boogie/Downloads/archive/winemag-data-130k-v2.csv')\n",
    "df = df.sample(frac=0.3, random_state=42)\n",
    "total_rows = df.shape[0]\n",
    "\n",
    "st.write(\"Total rows after sampling:\", total_rows)\n",
    "\n",
    "X_list = []  # to store feature matrices\n",
    "meta_list = []  # to store metadata\n",
    "\n",
    "# sample fraction for fitting\n",
    "sample_frac = 0.05  \n",
    "sample_variety = df[['variety']].sample(frac=sample_frac, random_state=42)\n",
    "sample_region = df[['region_1']].sample(frac=sample_frac, random_state=42)\n",
    "sample_description = df['description'].sample(frac=sample_frac, random_state=42)\n",
    "\n",
    "pre_encoder_variety = OneHotEncoder(sparse_output=True, handle_unknown='ignore')\n",
    "pre_encoder_variety.fit(sample_variety)\n",
    "\n",
    "pre_encoder_region = OneHotEncoder(sparse_output=True, handle_unknown='ignore')\n",
    "pre_encoder_region.fit(sample_region)\n",
    "\n",
    "pre_tfidf = TfidfVectorizer(stop_words='english', max_features=500)\n",
    "pre_tfidf.fit(sample_description)\n",
    "\n",
    "def process_and_calculate_similarity(start_index):\n",
    "    # load chunk rows\n",
    "    chunk = df.iloc[start_index:start_index + chunk_size]\n",
    "\n",
    "    # Transform\n",
    "    variety_encoded = pre_encoder_variety.transform(chunk[['variety']])\n",
    "    region_encoded = pre_encoder_region.transform(chunk[['region_1']])\n",
    "    description_tfidf = pre_tfidf.transform(chunk['description'])\n",
    "    \n",
    "    X = hstack([variety_encoded, region_encoded, description_tfidf])\n",
    "    similarity_matrix = cosine_similarity(X)\n",
    "    \n",
    "    X_list.append(X)\n",
    "    meta_list.append(\n",
    "        chunk[['title', 'description', 'taster_twitter_handle', 'variety', 'price', 'region_1']]\n",
    "    )\n",
    "    \n",
    "    # Return the similarity matrix and chunk\n",
    "    return similarity_matrix, chunk  \n",
    "\n",
    "# Loop through dataset in chunks\n",
    "recommendations = []\n",
    "current_chunk = 0\n",
    "\n",
    "while current_chunk * chunk_size < total_rows:\n",
    "    similarity_matrix, chunk = process_and_calculate_similarity(current_chunk * chunk_size)\n",
    "    \n",
    "    # For each wine in this chunk, find its top similar wines\n",
    "    for wine_index in range(chunk.shape[0]):\n",
    "        similar_wines = similarity_matrix[wine_index]\n",
    "        similar_wine_indices = similar_wines.argsort()[-6:-1][::-1]\n",
    "        recommended_wines = chunk.iloc[similar_wine_indices]\n",
    "        recommendations.append(recommended_wines[['title', 'variety', 'price', 'region_1']])\n",
    "    \n",
    "    current_chunk += 1\n",
    "    del similarity_matrix, chunk\n",
    "    gc.collect()\n",
    "\n",
    "final_recommendations = pd.concat(recommendations, ignore_index=True)\n",
    "\n",
    "st.write(\"Final Recommendations:\")\n",
    "st.dataframe(final_recommendations)\n",
    "\n",
    "X_full = vstack(X_list)\n",
    "meta_full = pd.concat(meta_list, ignore_index=True)\n",
    "\n",
    "st.write(\"X_full shape:\", X_full.shape)\n",
    "st.write(\"meta_full shape:\", meta_full.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "af84429d-7a15-4b88-a1a5-b0a613dee546",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "# Adjust pandas display options for better readability (optional)\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.width', 160)\n",
    "pd.set_option('display.max_colwidth', 120)\n",
    "\n",
    "def ask_branch_questions():\n",
    "    # REPLACED input() with Streamlit widgets\n",
    "    branch = st.selectbox(\n",
    "        \"Select wine type:\",\n",
    "        [\"red\", \"white\", \"sparkling\"]\n",
    "    )\n",
    "    \n",
    "    preference_string = \"\"\n",
    "\n",
    "    if branch == \"red\":\n",
    "        flavor_answer = st.radio(\n",
    "            \"For red wines, what flavor profile do you prefer?\",\n",
    "            [\"dry tannic\", \"fruity juicy\", \"spicy robust\", \"earthy nuanced\"]\n",
    "        )\n",
    "        preference_string += flavor_answer + \" \"\n",
    "\n",
    "        body_answer = st.radio(\n",
    "            \"Which body do you enjoy most?\",\n",
    "            [\"light-bodied\", \"medium-bodied\", \"full-bodied\"]\n",
    "        )\n",
    "        preference_string += body_answer + \" \"\n",
    "\n",
    "    elif branch == \"white\":\n",
    "        taste_answer = st.radio(\n",
    "            \"For white wines, what taste profile appeals to you?\",\n",
    "            [\"crisp dry\", \"lightly sweet\", \"rich oaky\"]\n",
    "        )\n",
    "        preference_string += taste_answer + \" \"\n",
    "        \n",
    "        acid_answer = st.radio(\n",
    "            \"What acidity level do you enjoy?\",\n",
    "            [\"high acidity\", \"moderate acidity\", \"low acidity\"]\n",
    "        )\n",
    "        preference_string += acid_answer + \" \"\n",
    "\n",
    "    elif branch == \"sparkling\":\n",
    "        sweet_answer = st.radio(\n",
    "            \"For sparkling wines, what sweetness do you prefer?\",\n",
    "            [\"brut dry\", \"extra brut\", \"demi-sec off-dry sweet\"]\n",
    "        )\n",
    "        preference_string += sweet_answer + \" \"\n",
    "        \n",
    "        occasion_answer = st.radio(\n",
    "            \"For which occasion are you selecting sparkling wine?\",\n",
    "            [\"casual everyday\", \"celebratory\", \"special gourmet pairing\"]\n",
    "        )\n",
    "        preference_string += occasion_answer + \" \"\n",
    "\n",
    "    else:\n",
    "        st.write(\"Invalid branch selected. Defaulting to red wine preferences.\")\n",
    "        branch = \"red\"\n",
    "        preference_string = \"dry tannic full-bodied \"\n",
    "\n",
    "    return branch, preference_string.strip()\n",
    "\n",
    "\n",
    "# -----------------------------\n",
    "# Minimal usage example\n",
    "# -----------------------------\n",
    "\n",
    "# 1. Ask for User Preferences\n",
    "selected_branch, user_query_text = ask_branch_questions()\n",
    "\n",
    "st.write(\"Selected branch:\", selected_branch)\n",
    "st.write(\"User preference summary:\", user_query_text)\n",
    "\n",
    "# 2. Compute Similarity Scores\n",
    "# (Assumes you already loaded and fitted pre_tfidf, X_full, meta_full somewhere)\n",
    "# e.g., pre_tfidf = ...\n",
    "#       X_full = ...\n",
    "#       meta_full = ...\n",
    "# This is just a placeholder to show how you'd display the results in Streamlit.\n",
    "\n",
    "# We'll pretend the user has clicked a button to see recommendations:\n",
    "if st.button(\"Show Recommendations\"):\n",
    "    user_query_vector = pre_tfidf.transform([user_query_text])\n",
    "\n",
    "    tfidf_feature_count = len(pre_tfidf.get_feature_names_out())\n",
    "    X_tfidf_full = X_full[:, -tfidf_feature_count:]\n",
    "\n",
    "    similarity_scores = cosine_similarity(user_query_vector, X_tfidf_full).flatten()\n",
    "    meta_full['query_similarity'] = similarity_scores\n",
    "\n",
    "    nlp_recommendations = meta_full.sort_values(by='query_similarity', ascending=False)\n",
    "\n",
    "    # 3. Display Top 5, Omitting 'taster_name'\n",
    "    all_cols = list(nlp_recommendations.columns)\n",
    "    if 'taster_name' in all_cols:\n",
    "        all_cols.remove('taster_name')\n",
    "\n",
    "    st.write(\"Top 5 Wine Recommendations Based on Your Preferences (Excluding 'taster_name'):\")\n",
    "    top5 = nlp_recommendations.head(5)[all_cols]\n",
    "    st.dataframe(top5)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "006da411-36e5-49be-b731-3e45f09fe265",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import gc\n",
    "\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from scipy.sparse import hstack, vstack\n",
    "\n",
    "# ------------------------------------------------------\n",
    "# Streamlit Title\n",
    "# ------------------------------------------------------\n",
    "st.title(\"Wine Recommendations by Boogie\")\n",
    "\n",
    "# ------------------------------------------------------\n",
    "# Config\n",
    "# ------------------------------------------------------\n",
    "chunk_size = 200\n",
    "csv_path = \"/Users/boogie/Downloads/archive/winemag-data-130k-v2.csv\"  # <-- adjust if needed\n",
    "\n",
    "# ------------------------------------------------------\n",
    "# Load and Sample Data\n",
    "# ------------------------------------------------------\n",
    "df = pd.read_csv(csv_path)\n",
    "df = df.sample(frac=0.3, random_state=42)  # 30% sample\n",
    "total_rows = df.shape[0]\n",
    "st.write(\"Total rows after sampling:\", total_rows)\n",
    "\n",
    "# ------------------------------------------------------\n",
    "# Fit Encoders & TF-IDF on a 5% Sample\n",
    "# ------------------------------------------------------\n",
    "sample_frac = 0.05\n",
    "sample_variety = df[['variety']].sample(frac=sample_frac, random_state=42)\n",
    "sample_region = df[['region_1']].sample(frac=sample_frac, random_state=42)\n",
    "sample_description = df['description'].sample(frac=sample_frac, random_state=42)\n",
    "\n",
    "pre_encoder_variety = OneHotEncoder(sparse_output=True, handle_unknown='ignore')\n",
    "pre_encoder_variety.fit(sample_variety)\n",
    "\n",
    "pre_encoder_region = OneHotEncoder(sparse_output=True, handle_unknown='ignore')\n",
    "pre_encoder_region.fit(sample_region)\n",
    "\n",
    "pre_tfidf = TfidfVectorizer(stop_words='english', max_features=500)\n",
    "pre_tfidf.fit(sample_description)\n",
    "\n",
    "# ------------------------------------------------------\n",
    "# Process Data in Chunks, Compute Similarity\n",
    "# ------------------------------------------------------\n",
    "X_list = []\n",
    "meta_list = []\n",
    "\n",
    "def process_and_calculate_similarity(start_index):\n",
    "    chunk = df.iloc[start_index : start_index + chunk_size]\n",
    "    variety_encoded = pre_encoder_variety.transform(chunk[['variety']])\n",
    "    region_encoded  = pre_encoder_region.transform(chunk[['region_1']])\n",
    "    desc_tfidf      = pre_tfidf.transform(chunk['description'])\n",
    "\n",
    "    X = hstack([variety_encoded, region_encoded, desc_tfidf])\n",
    "    similarity_matrix = cosine_similarity(X)\n",
    "\n",
    "    X_list.append(X)\n",
    "    meta_list.append(\n",
    "        chunk[['title', 'description', 'taster_twitter_handle', 'variety', 'price', 'region_1']]\n",
    "    )\n",
    "    return similarity_matrix, chunk\n",
    "\n",
    "recommendations = []\n",
    "current_chunk = 0\n",
    "\n",
    "while current_chunk * chunk_size < total_rows:\n",
    "    similarity_matrix, chunk = process_and_calculate_similarity(current_chunk * chunk_size)\n",
    "\n",
    "    # For each wine in this chunk, find top-5 similar wines\n",
    "    for wine_index in range(chunk.shape[0]):\n",
    "        similar_wines = similarity_matrix[wine_index]\n",
    "        similar_wine_indices = similar_wines.argsort()[-6:-1][::-1]\n",
    "        recommended_wines = chunk.iloc[similar_wine_indices]\n",
    "        recommendations.append(\n",
    "            recommended_wines[['title', 'variety', 'price', 'region_1']]\n",
    "        )\n",
    "\n",
    "    current_chunk += 1\n",
    "    del similarity_matrix, chunk\n",
    "    gc.collect()\n",
    "\n",
    "final_recommendations = pd.concat(recommendations, ignore_index=True)\n",
    "\n",
    "# ------------------------------------------------------\n",
    "# Display Results\n",
    "# ------------------------------------------------------\n",
    "st.write(\"Final Recommendations:\")\n",
    "st.dataframe(final_recommendations)\n",
    "\n",
    "X_full = vstack(X_list)\n",
    "meta_full = pd.concat(meta_list, ignore_index=True)\n",
    "\n",
    "st.write(\"X_full shape:\", X_full.shape)\n",
    "st.write(\"meta_full shape:\", meta_full.shape)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
