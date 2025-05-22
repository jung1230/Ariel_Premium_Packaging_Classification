from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import random
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

# Reading a CSV file into a DataFrame, selecting specific columns
fields = ['item_description']
df = pd.read_csv('order_details_202505141508.csv', usecols=fields)

# use for loop to print first 10 rows of the DataFrame
for i in range(20):
    print(df.iloc[i].values[0])

import re

print(df.shape)
print("\n\nAfter removing duplicates")
df['item_description'] = df['item_description'].str.lower() # there are lots of words with the same meaning but different cases
print(df.shape)

def clean_data(text):
    # apply number masking to reduce sensitivity to specific quantity values
    text = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', text)#\b means word boundary. \d+ means Match 1 or more digits
    # remove non-informative puncutuation
    text = re.sub(r'[-:*"\'.,]', '', text)
    # remove non-informative words
    text = re.sub(r'\b(please|kindly)\b', '', text) 
    # remove repeated spaces
    text = re.sub(r'\s+', ' ', text) # \s+ = one or more whitespace character
    # remove "special package"
    text = re.sub(r'\b(special package|special packaging)\b', '', text)
    # remove space at the beginning and end of the string
    text = text.strip()
    # remove "special package"
    text = re.sub(r'\b(special package|special packaging)\b', '', text)

    return text



df['item_description'] = df['item_description'].apply(clean_data)



# remove duplicates
df.drop_duplicates(inplace=True)
print(df.shape)

for i in range(20):
    print(df.iloc[i].values[0])

# turn it into a list
item_list = df['item_description'].tolist()
print(len(item_list))

grid = {
    "n_neighbors": [25, 50, 75, 100, 150, 200],
    "min_dist": [0.0, 0.1, 0.2, 0.3],
    "n_components": [2, 5, 10, 15, 20, 25, 30],
    "Umetric": ['euclidean', 'manhattan', 'cosine'],
    "min_samples": [5, 10, 15, 20, 25, 30],
    "min_cluster_size": [5, 10, 15, 20, 25, 30],
    "cluster_selection_epsilon": [0.0, 0.1, 0.2, 0.3],
    "Hmetric": ['euclidean', 'manhattan'],
}

# test only
# grid = {
#     "n_neighbors": [25, 50],
#     "min_dist": [0.3],
#     "n_components": [2, 5],
#     "Umetric": ['euclidean'],
#     "min_samples": [10],
#     "min_cluster_size": [5, 10],
#     "cluster_selection_epsilon": [0.1],
#     "Hmetric": ['euclidean'],
# }

random_seeds = [33, 34, 35]  
models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']

results = []

for model_name in models:
    model = SentenceTransformer(model_name)
    orig_emb = model.encode(item_list, show_progress_bar=True)

    for params in ParameterGrid(grid):
        silhouettes, dbis, chis = [], [], []

        for seed in random_seeds:
            random.seed(seed)
            np.random.seed(seed)

            # UMAP on the original embeddings
            reducer = umap.UMAP(
                n_neighbors=params['n_neighbors'],
                min_dist=params['min_dist'],
                n_components=params['n_components'],
                metric=params['Umetric'],
                random_state=seed
            )
            emb2d = reducer.fit_transform(orig_emb)

            # HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_samples=params['min_samples'],
                min_cluster_size=params['min_cluster_size'],
                cluster_selection_epsilon=params['cluster_selection_epsilon'],
                metric=params['Hmetric'],
                prediction_data=True
            )
            clusterer.fit(emb2d)
            labels = clusterer.labels_

            # skip trivial clusterings
            if len(set(labels)) < 2 or (labels == -1).all():
                # Skip or assign a sentinel score
                continue

            silhouettes.append(silhouette_score(emb2d, labels))
            dbis.append(davies_bouldin_score(emb2d, labels))
            chis.append(calinski_harabasz_score(emb2d, labels))

        if silhouettes:  # at least one valid run
            results.append({
                **params,
                "model": model_name,
                "silhouette": np.mean(silhouettes),
                "dbi": np.mean(dbis),
                "chi": np.mean(chis),
            })

# Turn into DataFrame for easy sorting/filtering
df = pd.DataFrame(results)

# Find best by silhouette
best = df.sort_values("silhouette", ascending=False).iloc[0]
print("Best configuration (by silhouette):")
print(best)

df.to_csv("umap_hdbscan_hyperparam_results.csv", index=False)