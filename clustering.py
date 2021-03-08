import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import args as args_dependency
import dataset as ds
import string
from typing import List
import numpy as np
import json
import hashlib
import pickle
import os

from features.FeatureFunction import FeatureFunction
from features.AvgSentenceLen import AvgSentenceLen


# If we come up with feature extractors we should add them to this list
CUSTOM_FEATURE_EXTRACTORS: List[FeatureFunction] = [AvgSentenceLen()]
SVD_COMPONENTS = 1000  # Increase this for a more accurate dim-reduction, decrease it for a smaller one.


def extract_custom_features(contexts: List[str]):
    custom_features = np.zeros((len(contexts), len(CUSTOM_FEATURE_EXTRACTORS)))
    for i in range(len(contexts)):
        for j in range(len(CUSTOM_FEATURE_EXTRACTORS)):
            custom_features[i, j] = CUSTOM_FEATURE_EXTRACTORS[j].evaluate(contexts[i])
    return custom_features


#Text pre-processing
def text_process(text):
    """removes punctuation, stopwords, and returns a list of the remaining words, or tokens"""
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    return ' '.join([stemmer.lemmatize(word) for word in nopunc])

def get_hash_str(text):
    md_object = hashlib.md5(text.encode())
    return md_object.hexdigest()

def normalize_matrix_so_cols_have_zero_mean_unit_variance(mtx: np.ndarray) -> np.ndarray:
    mtx -= np.mean(mtx, axis=0).reshape(1, -1)
    mtx /= np.std(mtx, axis=0).reshape(1, -1)
    return mtx

def main():
    args = args_dependency.get_train_test_args()

    #custom_features = extract_custom_features(X_train)

    MAX_DF = args['max_df']
    MIN_DF = args['min_df']
    MAX_FEATURES = args['max_features']
    N_CLUSTER = args['num_clusters']
    sub_label = f"feature{MAX_FEATURES}_mindf{MIN_DF}_maxdf{MAX_DF}"
    label = f"n{N_CLUSTER}_{sub_label}"
    X_transformed_pick = f"save/train_text_{sub_label}.pickle"
    X_train_pick = "save/all_train_text_id.pickle"
    cached_processed = 'save/all_train_text_processed' 

    # Load training text
    if os.path.exists(X_train_pick):
        text_to_id_dict = pickle.load(open(X_train_pick, 'rb'))
        X_train = list(text_to_id_dict.keys())
    else:
        # read data
        data = ['datasets/indomain_train/squad',
               'datasets/indomain_train/nat_questions',
               'datasets/indomain_train/newsqa',
               'datasets/oodomain_train/duorc',
               'datasets/oodomain_train/race',
               'datasets/oodomain_train/relation_extraction']

        all_data = {}
        for i in data:
            data_dict = ds.read_squad(i, 'save')
            all_data = ds.merge(data_dict, all_data)

        text_to_id_dict = dict(zip(all_data['context'], all_data['topic_id']))
        X_train = list(text_to_id_dict.keys())
        pickle.dump(text_to_id_dict, open(X_train_pick, 'wb'))

    if os.path.exists(X_transformed_pick):
        X_transformed = pickle.load(open(X_transformed_pick, 'rb'))
    else:
        #Vectorisation : -
        #https://scikit-learn.org/stable/modules/decomposition.html#lsa
        #tfidfconvert = TfidfVectorizer(analyzer=text_process).fit(X_train)
        if os.path.exists(cached_processed):
            X_train_processed = pickle.load(open(cached_processed, 'rb'))
        else:
            nltk.download('stopwords')
            nltk.download('wordnet')

            # get custom features before modifying contexts
            X_train_processed = [text_process(item) for item in X_train]
            pickle.dump(X_train_processed, open(cached_processed, 'wb'))

        #tfidfconvert = TfidfVectorizer(max_features=MAX_FEATURES, sublinear_tf=True, max_df=MAX_DF, min_df=MIN_DF).fit(X_train_processed)
        tfidfconvert = TfidfVectorizer(max_features=MAX_FEATURES, max_df=MAX_DF, min_df=MIN_DF).fit(X_train_processed)

        X_transformed=tfidfconvert.transform(X_train_processed)
        pickle.dump(tfidfconvert, open(f"save/tfidf_{sub_label}.pickle", "wb"))
        pickle.dump(X_transformed, open(f"save/train_text_{sub_label}.pickle", "wb"))

    #svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=args["seed"])
    #X_reduced = svd.fit_transform(X_transformed)

    # append the custom features for the full feature set
    #raw_k_means_features = np.concatenate((X_reduced, custom_features), axis=1)

    # normalize each column to have 0 mean and unit variance
    #k_means_features = normalize_matrix_so_cols_have_zero_mean_unit_variance(raw_k_means_features)
    #k_means_features = normalize_matrix_so_cols_have_zero_mean_unit_variance(X_transformed)
    #np.save('save/svd_1000', k_means_features)

    # Cluster the training sentences with K-means technique
    km = KMeans(n_clusters=N_CLUSTER)
    modelkmeans = km.fit(X_transformed)

    hist, bins = np.histogram(modelkmeans.labels_, bins=N_CLUSTER)
    print (hist)

    kmeans_dict = {get_hash_str(X_train[idx]): int(label) for idx, label in enumerate(modelkmeans.labels_)}

    with open(f'save/kmeans_topic_id_pair_{label}', 'w') as f:
        json.dump(kmeans_dict, f)

    # Build the matrix with cluster IDs as rows, topic IDs as columns
    topics_id = []
    for k, v in text_to_id_dict.items(): 
        if v not in topics_id:
            topics_id.append(v)
    num_topics = len(topics_id)
    co_occurance = np.zeros((num_topics, N_CLUSTER))
    for idx, cluster in enumerate(modelkmeans.labels_):
        topic_id = int(text_to_id_dict[X_train[idx]])
        co_occurance[topic_id][int(cluster)] += 1
    np.save(f'save/kmeans_co_occurance_{label}', co_occurance)
    '''
    K = range(4,100)
    Sum_of_squared_distances = []
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(k_means_features)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    '''


if __name__ == "__main__":
    main()
