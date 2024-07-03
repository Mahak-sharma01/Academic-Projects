#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D, Input
from tensorflow.keras.layers import Dropout
import keras.callbacks
import pickle

def load_data(sqlite_file):
    connection = sqlite3.connect(sqlite_file)
    df_all = pd.read_sql("SELECT * FROM tvmaze", con=connection)
    df_gen = pd.read_sql("SELECT * FROM tvmaze_genre", connection)
    df_req = pd.read_sql("SELECT t.description as Description, GROUP_CONCAT(tg.genre) as Genre from tvmaze t JOIN tvmaze_genre tg on t.tvmaze_id = tg.tvmaze_id group by description;", connection)
    connection.close()
    return df_all, df_gen, df_req

def data_cleaning(df_req):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # USing a copy to prevent warning
    df_req = df_req.copy()
    df_req['Description'] = df_req['Description'].apply(text_cleaning)
    df_req['Description'] = df_req['Description'].apply(remove_stopwords)
    df_req = df_req[~(df_req['Description'].isin(['', 'dupe:']) | df_req['Description'].str.endswith('dupe:'))]
    df_req.reset_index(drop=True, inplace=True)
    df_req['Description'] = df_req['Description'].apply(tokenize_and_lemmatize)
    df_req['Genre'] = df_req['Genre'].str.split(',')

    return df_req



def text_cleaning(text):
    if text is None:
        return ''
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    if isinstance(text, str):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(words)
    else:
        return text

def tokenize_and_lemmatize(text):
    if isinstance(text, str):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        text = text.lower()
        tokens = nltk.tokenize.word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    else:
        return text

def target_prep(df_req, df_gen):
    # Ensuring that 'Genre' column exists in df_gen
    if 'genre' not in df_gen.columns:
        raise KeyError("The 'Genre' column is not found in df_gen. Please check the DataFrame structure.")

    genres = df_gen['genre'].unique()
    target = np.zeros((df_req.shape[0], len(genres)))
    categories_forward_lookup = {genre: i for i, genre in enumerate(genres)}
    for i, cs in zip(df_req.index, df_req.Genre):
        for c in cs:
            category_number = categories_forward_lookup[c]
            target[i, category_number] = 1
    return target, categories_forward_lookup

def train_model(X_train, y_train, X_validation, y_validation):
    max_tokens = 50000
    output_sequence_length = 100
    embedding_dim = 32

    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
    vectorizer.adapt(X_train)

    inputs = Input(shape=(1,), dtype=tf.string)
    vectorized = vectorizer(inputs)
    embedded = Embedding(max_tokens + 1, embedding_dim)(vectorized)
    averaged = GlobalAveragePooling1D()(embedded)
    layer1 = Dense(128, activation='relu')(averaged)
    dropout = Dropout(0.5)(layer1)
    output = Dense(len(categories_forward_lookup), activation='sigmoid')(layer1)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=200, validation_data=(X_validation, y_validation), callbacks=[early_stopping])
    return model

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy

def save_model(model, lookup_dict):
    model.save('model')
    with open('categories_forward_lookup.pkl', 'wb') as f:
        pickle.dump(lookup_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model from SQLite data")
    parser.add_argument("--training-data", required=True, help="Path to the SQLite database file")
    args = parser.parse_args()

    df_all, df_gen, df_req = load_data(args.training_data)
    df_req = data_cleaning(df_req)
    target, categories_forward_lookup = target_prep(df_req, df_gen)

    X_train, X_test, y_train, y_test = train_test_split(df_req['Description'], target, test_size=0.2, random_state=142)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=142)

    model = train_model(X_train, y_train, X_validation, y_validation)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model training done. Accuracy: {accuracy:.2f} %")

    save_model(model, categories_forward_lookup)

