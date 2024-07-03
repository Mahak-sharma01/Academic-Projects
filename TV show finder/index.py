


import argparse
import sqlite3
import json
import pickle
import nltk

# Import any additional libraries or modules you may need

def read_index(sqlite_file):
    try:
        # Connect to the SQLite database
        connection = sqlite3.connect(sqlite_file)
        cursor = connection.cursor()

        # Update this query to reflect your database schema
        query = "SELECT tvmaze_id, showname, description FROM tvmaze"  # Assuming your table is named 'shows'
        cursor.execute(query)

        data_to_index = []
        texts = []

        for row in cursor.fetchall():
            processed_data = preprocess_data(row)
            data_to_index.append(processed_data)
            texts.append(processed_data['description'])

        # Create TF-IDF vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        # Save the indexed data to a separate file
        indexed_data = {
            'data': data_to_index,
            'tfidf_matrix': X,
            'vectorizer': vectorizer
        }

        output_file_path = 'indexed_tv_data.pkl'
        with open(output_file_path, 'wb') as index_file:
            pickle.dump(indexed_data, index_file)

        connection.close()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def preprocess_data(data):
    # Assumed that 'data' is a tuple with an ID, title, and description
    id, title, description = data
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = []
    if description is not None:
        tokens = nltk.word_tokenize(description.lower())
    
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_description = ' '.join(lemmatized_tokens)
    
    processed_data = {
        'id': id,
        'title': title,
        'description': lemmatized_description
    }

    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index data from an SQLite database")
    parser.add_argument("--raw-data", required=True, help="Path to the SQLite database file")
    args = parser.parse_args()
    read_index(args.raw_data)


