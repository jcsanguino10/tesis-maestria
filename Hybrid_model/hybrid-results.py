# %%
#pip install -U sentence-transformers

# %%
#pip install turicreate

# %% [markdown]
# ## Imports

# %%
from tqdm import tqdm
import csv
import math
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
import gensim
from sqlalchemy import create_engine
import pandas as pd

import turicreate as tc

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

def processing_data_users(pDf):
    df_users = pd.read_csv("final_data_set.csv", sep=",",
                           low_memory=False, index_col="Unnamed: 0")
    df_users = df_users.drop_duplicates(subset=["user_id", "course_name"])
    print("Shape before filter", df_users.shape)
    df_users = filter_df_user(df_users, pDf)
    print("Shape after filter", df_users.shape)
    r_df_train, r_df_test = split_data_sets(df_users)
    return r_df_train, r_df_test

def filter_df_user(pDf_users, pCourse_list):
    r_df_users = pDf_users[pDf_users["course_name"].isin(
        pCourse_list)]
    r_df_users["courses_viewed"] = r_df_users.groupby(
        "user_id")["course_name"].transform('nunique')
    r_df_users = r_df_users[r_df_users["courses_viewed"] > 1]
    r_df_users = r_df_users[r_df_users["courses_viewed"] < 50]
    return r_df_users

def create_df_courses():
    df = getSqlResult(
        'SELECT course_path,id_lesson,html,description_lesson,description_course FROM publish')
    df["html_course"] = df.groupby("course_path")[
        'html'].transform(lambda x: ' '.join(x))
    df = df[df['course_path'].values != '']
    df.drop_duplicates(subset=["course_path"], inplace=True, ignore_index=True)
    return df[df['course_path'].values != '']

# %%
def vectorized_corpus_to_matrix_similarity(pVectorized_corpus, pFeatures):
    index_temp = get_tmpfile("index")
    matrix_similarity = Similarity(
        index_temp, corpus=pVectorized_corpus, num_features=pFeatures)
    return matrix_similarity

def generate_sr_model_transformer(full_sentences, transformer_name, pDf_test, pDf_train, pDf_courses):
    model = SentenceTransformer(transformer_name)
    embeddings = model.encode(full_sentences)
    embeddings_tuple = [list(zip(range(0, len(embeddings[0])), vector))
                        for vector in embeddings]
    recommendations = generate_matrix_cb(
        embeddings_tuple, len(embeddings_tuple[0]), pDf_train, pDf_courses)
    return recommendations

def custom_mean_precision(r, k):
    precision = 0.0
    for i in range(0, len(r)):
        precision += np.sum(r[i]) / k[i]
    return precision / len(k)


def generate_matrix_cb(pVectorized_corpus, pNumber_topics, pDf_train, pDf_courses):
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
        recommendations[name] = matrix_similarity
    return recommendations

def binary_array_recommendations(matrix_recomendation, test_dataframe, train_dataframe, pDf_courses):
    final_array = []
    len_courses = []
    courses = pDf_courses["course_path"].tolist()
    for name, group in test_dataframe.groupby("user_id"):
        courses = group["course_name"].tolist()
        number_courses = 5 + len(courses)
        top_courses = matrix_recomendation.loc[name].nlargest(number_courses).index
        top_courses = [course for course in top_courses if course not in train_dataframe[train_dataframe["user_id"] == name]["course_name"].tolist()]
        top_courses = top_courses[:5]
        temp_array = []
        for course in top_courses:
            temp_array.append(int(course in courses))
        final_array.append(temp_array)
        if len(courses) >= 5:
            len_courses.append(5)
        else:
            len_courses.append(len(courses))
    return final_array, len_courses

def evaluate_matrix(matrix, test_dataframe, train_dataframe, pDf_courses):
    final_array, len_courses = binary_array_recommendations(matrix, test_dataframe, train_dataframe, pDf_courses)
    ct_metric = custom_mean_precision(final_array, len_courses)
    return ct_metric
    
def get_weights_matrix(matrix_cb, matrix_cf, test_dataframe, train_dataframe, pDf_courses, step):
    weights = np.arange (0, (1+step), step)
    for i in range(0,len(weights)):
        cb_weight = weights[i]
        cf_weight = weights[(len(weights) -(i+1))]
        full_matrix = (matrix_cb * cb_weight) + (matrix_cf * cf_weight)
        ct_metric = evaluate_matrix(full_matrix, test_df, train_df, df)
        print("Weight cb: {}, Weight cf: {} , precision: {}".format(cb_weight,cf_weight,ct_metric))

def get_weights_matrix_doc(matrix_cb, matrix_cf, test_dataframe, train_dataframe, pDf_courses, step, doc_writer, index_folder):
    weights = np.arange (0, (1+step), step)
    for i in tqdm(range(0,len(weights))):
        cb_weight = weights[i]
        cf_weight = weights[(len(weights) -(i+1))]
        full_matrix = (matrix_cb * cb_weight) + (matrix_cf * cf_weight)
        ct_metric = evaluate_matrix(full_matrix, test_dataframe, train_dataframe, pDf_courses)
        model_name ="cb:{}-cf:{}".format(cb_weight,cf_weight)
        doc_writer.writerow([index_folder, model_name, ct_metric])
        
def random_split(dataframe, field_name, size):
    return dataframe.loc[dataframe[field_name].isin(np.random.choice(dataframe[field_name].unique(), size=size, replace=False))]
    


# %%
def generate_model_CF(dataframe, field_name,len_courses):
    df = dataframe
    generate_rating(df)
    train_data = tc.load_sframe(df)
    model = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='course_name', target=field_name, similarity_type='cosine', verbose=False)
    model = model.recommend(users=train_df["user_id"].unique().tolist(), k = len_courses, verbose=False).to_dataframe()
    return model

def generate_rating(dataframe):
    dataframe["sqrt_ratings_by_user"] = np.sqrt((dataframe["course_porcentage"]/dataframe["course_porcentage_mean"]))*5
    dataframe.loc[dataframe.sqrt_ratings_by_user > 5, 'sqrt_ratings_by_user'] = 5
    return dataframe

# %% [markdown]
# # Global Variables

# %%
df = create_df_courses()

# %%
course_list= df["course_path"].values

# %%
train_df, test_df = processing_data_users(course_list)

# %%
model_cf = generate_model_CF(train_df,"sqrt_ratings_by_user",len(course_list))

# %%
model_cf

# %%
model_cf["course_name"].nunique()

# %%
model_cf.loc[model_cf['score'] > 1, 'score'] = 1

# %%
model_cb = generate_sr_model_transformer(df["html_course"], "all-mpnet-base-v2", test_df, train_df, df)

# %%
matrix_cb = pd.DataFrame.from_dict(model_cb, orient='index', columns=course_list)

# %%
matrix_cb[matrix_cb <0] = 0

# %%
matrix_cf = model_cf.pivot(values='score', index="user_id", columns='course_name')
matrix_cf = matrix_cf.reindex(columns=course_list, fill_value=0)

# %%
matrix_cf

# %%
evaluate_matrix(matrix_cf, test_df, train_df, df)

# %%
evaluate_matrix(matrix_cb, test_df, train_df, df)

# %%
#get_weights_matrix(matrix_cb,matrix_cf, test_df, train_df, df, 0.1)

# %%
def evaluate_models_by_folders(matrix_cb, matrix_cf, test_dataframe, train_dataframe, pDf_courses,size,number_folders,file_name):
    with open(file_name, mode='w', newline="") as data_user_file:
        doc_writer = csv.writer(data_user_file, delimiter=',', quotechar='"')
        doc_writer.writerow(["Folder", "Model", "Custom_metric"])
        for i in tqdm(range(0, number_folders)):
            pTrain_df = random_split(train_dataframe, "user_id", size=size)
            pTest_df = test_dataframe[test_dataframe["user_id"].isin(pTrain_df["user_id"].values)]
            get_weights_matrix_doc(matrix_cb,matrix_cf, pTest_df, pTrain_df, pDf_courses, 0.1, doc_writer, i)

# %%
evaluate_models_by_folders(matrix_cb,matrix_cf, test_df, train_df, df, 3000, 100,"hybrid-results.csv")

# %%



