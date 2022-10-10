# %%
#pip install -U sentence-transformers

# %% [markdown]
# ## Imports

# %%
from tqdm import tqdm
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from gensim.models import CoherenceModel
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from sqlalchemy import create_engine
import pandas as pd
import re
import gensim.corpora as corpora
from pprint import pprint
import contractions
import demoji
import string

import nltk
nltk.download('punkt')


# %% [markdown]
# # Database functions

# %%
def getSqlResult(sql):
    db_connection_str = 'mysql+pymysql://profesor:Tesis_2022@localhost/info_coursesdb'
    db_connection = create_engine(db_connection_str)
    return pd.read_sql(sql, con=db_connection)

# %% [markdown]
# # Processing data functions

# %%
def preprocess_text(text_col):
    # convert to lowercase
    text_col = text_col.apply(
        lambda x: ' '.join([w.lower() for w in x.split()]))

    # remove emojis
    text_col = text_col.apply(lambda x: demoji.replace(x, ""))

    # expand contractions
    text_col = text_col.apply(lambda x: ' '.join(
        [contractions.fix(word) for word in x.split()]))

    # remove punctuation
    text_col = text_col.apply(lambda x: ''.join(
        [i for i in x if i not in string.punctuation]))

    # remove numbers
    text_col = text_col.apply(lambda x: ' '.join(
        re.sub("[^a-zA-Z]+", " ", x).split()))

    # Stop words
    text_col = text_col.apply(
        lambda x: gensim.parsing.preprocessing.remove_stopwords(x))

    # Stem text
    text_col = text_col.apply(
        lambda x: gensim.parsing.preprocessing.stem_text(x))

    # Tokenizer
    text_col = text_col.apply(lambda x: word_tokenize(x))

    return text_col

# Numera los cursos para un usuario especifico


def create_index(pDf, pUser_id):
    df_courses_index = pDf.loc[pDf['user_id']
                               == pUser_id].reset_index().reset_index()
    df_courses_index["position"] = df_courses_index["level_0"]
    df_courses_index = df_courses_index.drop(columns=["level_0", "index"])
    df_courses_index["position"] = df_courses_index["position"]+1

    return df_courses_index
# Esta función clasifica entre train y test


def clasificator(pIndex, pCourses_viewed):
    return "test" if pIndex > (math.floor(pCourses_viewed * 0.7)) else "train"


def split_data_sets(pDf):
    frames_train = []
    frames_test = []

    # Recorre por cada uno de los usuarios clasificando si las filas van para train o test
    for i in pDf["user_id"].unique():
        data_user = create_index(pDf, i)
        data_user["clasification"] = data_user.apply(
            lambda x: clasificator(x["position"], x["courses_viewed"]), axis=1)

        frames_train.append(data_user[data_user["clasification"] == "train"])
        frames_test.append(data_user[data_user["clasification"] == "test"])

    df_train = pd.concat(frames_train)
    df_test = pd.concat(frames_test)

    return df_train, df_test


def filter_df_user(pDf_users, pDf_courses):
    r_df_users = pDf_users[pDf_users["course_name"].isin(
        pDf_courses["course_path"].values)]
    r_df_users["courses_viewed"] = r_df_users.groupby(
        "user_id")["course_name"].transform('nunique')
    r_df_users = r_df_users[r_df_users["courses_viewed"] > 1]
    r_df_users = r_df_users[r_df_users["courses_viewed"] < 50]
    return r_df_users


def vectorized_corpus_to_matrix_similarity(pVectorized_corpus, pFeatures):
    index_temp = get_tmpfile("index")
    matrix_similarity = Similarity(
        index_temp, corpus=pVectorized_corpus, num_features=pFeatures)
    return matrix_similarity


def create_common_corpus(field_name, pDf):
    tokens = preprocess_text(pDf[field_name])
    common_dictionary = Dictionary(tokens)
    print("dictionary len without remove", len(common_dictionary))
    common_dictionary.filter_extremes(no_above=0.95)
    print("dictionary len removed", len(common_dictionary))
    common_corpus = [common_dictionary.doc2bow(text) for text in tokens]
    return common_dictionary, common_corpus, tokens


def processing_data_users(pDf):
    df_users = pd.read_csv("final_data_set.csv", sep=",",
                           low_memory=False, index_col="Unnamed: 0")
    df_users = df_users.drop_duplicates(subset=["user_id", "course_name"])
    print("Shape before filter", df_users.shape)
    df_users = filter_df_user(df_users, pDf)
    print("Shape after filter", df_users.shape)
    df_users = filter_df_user(df_users, pDf)
    r_df_train, r_df_test = split_data_sets(df_users)
    return r_df_train, r_df_test


def create_df_courses():
    df = getSqlResult(
        'SELECT course_path,id_lesson,html,description_lesson,description_course FROM publish')
    df["html_course"] = df.groupby("course_path")[
        'html'].transform(lambda x: ' '.join(x))
    df["all_description_lesson"] = df.groupby(
        "course_path")['description_lesson'].transform(lambda x: ' '.join(x))
    df.drop_duplicates(subset=["course_path"], inplace=True, ignore_index=True)
    return df


# %% [markdown]
# # Only LDA and LSI functions

# %%
def generate_model_lda(common_corpus, num_topics, passes):
    return gensim.models.LdaMulticore(common_corpus,
                                      num_topics=num_topics,
                                      id2word=common_dictionary,
                                      passes=passes,
                                      minimum_probability=0,
                                      random_state=1)


def generate_sr_model_transformer(full_sentences, transformer_name, pDf_test, pDf_train, pDf_courses):
    model = SentenceTransformer(transformer_name)
    embeddings = model.encode(full_sentences)
    embeddings_tuple = [list(zip(range(0, len(embeddings[0])), vector))
                        for vector in embeddings]
    recommendations = recommended_by_course(
        embeddings_tuple, len(embeddings_tuple[0]), pDf_train, pDf_courses)
    final_array, len_courses = binary_array_recommendations(
        recommendations, pDf_test, pDf_courses)
    metric = custom_mean_precision(final_array, len_courses)
    print("custom ", metric)


def evaluate_model(pModel, pCorpus):
    # Compute Perplexity (lower is better)
    print('Perplexity: ', round(pModel.log_perplexity(pCorpus), 2))

    # Compute Coherence Score (higher is better)
    coherence_model_lda = CoherenceModel(
        model=pModel, texts=tokens, dictionary=common_dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', round(coherence_lda, 2))
    return round(pModel.log_perplexity(pCorpus), 2), round(coherence_lda, 2)


def plot_results(pPerplexity, pCoherence, pNumbers_topics):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(pNumbers_topics, pPerplexity, 'g-')
    ax2.plot(pNumbers_topics, pCoherence, 'b-')
    ax1.set_xlabel('Number topics')
    ax1.set_ylabel('perplexity', color='g')
    ax2.set_ylabel('coherence', color='b')
    plt.show()


def plot_results_topics(pMetric, pNumbers_topics):
    fig, ax1 = plt.subplots()
    ax1.plot(pNumbers_topics, pMetric)
    ax1.set_xlabel('Number topics')
    ax1.set_ylabel('Precisión')
    plt.show()


def find_best_topic(pCorpus, pMax_topics, pIterations):
    perplexity = []
    coherence = []
    for num_topics in range(10, pMax_topics, 5):
        print("number topics", num_topics)
        x, y = evaluate_model(generate_model_lda(
            pCorpus, num_topics, pIterations), pCorpus)
        perplexity.append(x)
        coherence.append(y)
    plot_results(perplexity, coherence, range(10, pMax_topics, 5))


def search_best_number_topics(top_topics, pCommon_corpus, model_name, pDf_courses, pDf_train, pDf_test):
    metrics = []
    num_topics_by_metric = []
    for num_topics in range(5, top_topics, 10):
        model = None
        if model_name == "LDA":
            model = generate_model_lda(pCommon_corpus, num_topics, 50)
        else:
            model = gensim.models.lsimodel.LsiModel(
                pCommon_corpus, num_topics=num_topics, id2word=common_dictionary, power_iters=50)
        vectorized_corpus = model[pCommon_corpus]
        print(type(vectorized_corpus))
        recommendations = recommended_by_course(
            vectorized_corpus, num_topics, pDf_train, pDf_courses)
        print("Number topics: ", num_topics)
        final_array, len_courses = binary_array_recommendations(
            recommendations, pDf_test, pDf_courses)
        metric = custom_mean_precision(final_array, len_courses)
        metrics.append(metric)
        num_topics_by_metric.append(num_topics)
        print("custom ", metric)
    plot_results_topics(metrics, num_topics_by_metric)


# %% [markdown]
# # Recommendations functions

# %%
def custom_mean_precision(r, k):
    precision = 0.0
    for i in range(0, len(r)):
        precision += np.sum(r[i]) / k[i]
    return precision / len(k)


def recommended_by_course(pVectorized_corpus, pNumber_topics, pDf_train, pDf_courses):
    matrix = vectorized_corpus_to_matrix_similarity(
        pVectorized_corpus, pNumber_topics)
    courses = pDf_courses["course_path"].tolist()
    recommendations = dict()
    for name, group in pDf_train.groupby("user_id"):
        matrix_similarity = []
        courses_index = []
        num_top_temp = len(group["course_name"]) + 5
        for course in group["course_name"]:
            index = courses.index(course)
            if len(matrix_similarity) == 0:
                matrix_similarity = matrix[pVectorized_corpus[index]]
            else:
                maxtrix_similarity = matrix_similarity + \
                    matrix[pVectorized_corpus[index]]
        matrix_similarity = matrix_similarity / len(group["course_name"])
        top_temp = maxtrix_similarity.argsort()[-num_top_temp:][::-1]
        top = [x for x in top_temp if x not in courses_index]
        recommendations[name] = top[:5]
    return recommendations


def binary_array_recommendations(recommendations, test_dataframe, pDf_courses):
    final_array = []
    len_courses = []
    courses = pDf_courses["course_path"].tolist()
    for name, group in test_dataframe.groupby("user_id"):
        courses_index = [courses.index(course)
                         for course in group["course_name"].tolist()]
        temp_array = []
        for course in recommendations[name]:
            temp_array.append(int(course in courses_index))
        final_array.append(temp_array)
        if len(courses_index) >= 5:
            len_courses.append(5)
        else:
            len_courses.append(len(courses_index))
    return final_array, len_courses


# %% [markdown]
# # Evaluation functions

# %%
def evaluate_model_matrix(pCommon_corpus, model_name, pDf_train, pDf_test, pDf_courses, field_name="html_course"):
    vectorized_corpus = None
    if model_name == "LDA":
        model = gensim.models.LdaModel.load("./models/LDA_model", mmap='r')
        vectorized_corpus = model[pCommon_corpus]
    elif model_name == "LSI":
        model = gensim.models.lsimodel.LsiModel.load("./models/LSI_model")
        vectorized_corpus = model[pCommon_corpus]
    else:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(pDf_courses[field_name])
        vectorized_corpus = [
            list(zip(range(0, len(embeddings[0])), vector)) for vector in embeddings]

    recommendations = recommended_by_course(
        vectorized_corpus, len(vectorized_corpus[0]), pDf_train, pDf_courses)
    final_array, len_courses = binary_array_recommendations(
        recommendations, pDf_test, pDf_courses)
    return final_array, len_courses


def save_models(pCommon_corpus):
    model = gensim.models.lsimodel.LsiModel(
        pCommon_corpus, num_topics=165, id2word=common_dictionary, power_iters=50)
    model.save("./models/LSI_model")

    model = generate_model_lda(pCommon_corpus, 225, 50)
    model.save("./models/LDA_model")


def random_split(dataframe, field_name, size):
    return dataframe.loc[dataframe[field_name].isin(np.random.choice(dataframe[field_name].unique(), size=size, replace=False))]


def evaluate_models_by_folders(pTrain_df, pTest_df, pDf_course, number_folders, size, file_name, pCommon_corpus):
    with open(file_name, mode='w', newline="") as data_user_file:
        doc_writer = csv.writer(data_user_file, delimiter=',', quotechar='"')
        doc_writer.writerow(["Folder", "Model", "Custom_metric"])
        for i in tqdm(range(0, number_folders)):
            df = random_split(pTrain_df, "user_id", size=size)
            pTest_df_filter = pTest_df[pTest_df["user_id"].isin(df["user_id"].values)]
            models = ["all-roberta-large-v1", "all-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2",
                      "all-MiniLM-L12-v2", "paraphrase-multilingual-MiniLM-L12-v2", "LDA", "LSI"]
            for model_name in tqdm(models, leave=False):
                final_array, len_courses = evaluate_model_matrix(
                    pCommon_corpus, model_name, df, pTest_df_filter, pDf_course)
                ct_metric = custom_mean_precision(final_array, len_courses)
                doc_writer.writerow([i, model_name, ct_metric])


def evaluate_model_matrix_transformer(pVectorized_corpus, pDf_train, pDf_test, pDf_courses):
    recommendations = recommended_by_course(pVectorized_corpus, len(
        pVectorized_corpus[0]), pDf_train, pDf_courses)
    final_array, len_courses = binary_array_recommendations(
        recommendations, pDf_test, pDf_courses)
    return final_array, len_courses


def array_vectorized_corpus(model_name, field_name, pDf_courses):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(pDf_courses[field_name])
    vectorized_corpus = [
        list(zip(range(0, len(embeddings[0])), vector)) for vector in embeddings]
    return vectorized_corpus


def generate_corpus(array_fields, model_name, pDf_courses):
    array = []
    for field in array_fields:
        array.append(array_vectorized_corpus(model_name, field, pDf_courses))
    return array


def evaluate_corpus_by_folders(pTrain_df, pTest_df, pDf_course, number_folders, size, file_name, corpus_names, pAll_vectorized_corpus):
    with open(file_name, mode='w', newline="") as data_user_file:
        doc_writer = csv.writer(data_user_file, delimiter=',', quotechar='"')
        doc_writer.writerow(["Folder", "Corpus", "Custom_metric"])
        for i in tqdm(range(0, number_folders)):
            df = random_split(pTrain_df, "user_id", size=size)
            pTest_df_filter = pTest_df[pTest_df["user_id"].isin(df["user_id"].values)]
            for index in tqdm(range(0, 3), leave=False):
                final_array, len_courses = evaluate_model_matrix_transformer(
                    pAll_vectorized_corpus[index], df, pTest_df_filter, pDf_course)
                ct_metric = custom_mean_precision(final_array, len_courses)
                doc_writer.writerow([i, corpus_names[index], ct_metric])


def general_test_corpus(pAll_vectorized_corpus, pDf_train, pDf_test, pDf_courses):
    for vectorized_corpus in pAll_vectorized_corpus:
        final_array, len_courses = evaluate_model_matrix_transformer(
            vectorized_corpus, pDf_train, pDf_test, pDf_courses)
        ct_metric = custom_mean_precision(final_array, len_courses)
        print(ct_metric)


# %% [markdown]
# # Global Variables

# %%
data_courses = create_df_courses()

# %%
data_courses.head()

# %%
common_dictionary, common_corpus, tokens = create_common_corpus("html_course",data_courses)

# %%
data_train , data_test = processing_data_users(data_courses)

# %% [markdown]
# ## Stadistical tests

# %%
save_models(common_corpus)

# %%
corpus_names= ["html_course","all_description_lesson","description_course"]

# %%
all_vectorized_corpus = generate_corpus(corpus_names, "all-mpnet-base-v2", data_courses)

# %%
evaluate_models_by_folders(data_train,data_test,data_courses,100,3000, "results_CB_models", common_corpus)

# %%
evaluate_corpus_by_folders(data_train,data_test,data_courses,100,3000, "results_CB_corpus.csv", corpus_names, all_vectorized_corpus)


