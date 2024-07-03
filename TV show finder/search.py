
import argparse
import json
import pickle
import nltk

# Import any additional libraries or modules you may need

def read_index(index_file_path):
    with open(index_file_path, 'rb') as index_file:
        index = pickle.load(index_file)
    return index

def preprocess_data(data):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = nltk.word_tokenize(data.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_description = ' '.join(lemmatized_tokens)
    return lemmatized_description

def search_tv_shows(query, index):
    # Preprocess the query
    preprocessed_query = preprocess_data(query)
    
    # Transform the query to a TF-IDF vector
    query_vector = index['vectorizer'].transform([preprocessed_query])
    
    # Calculate the cosine similarity between the query vector and the indexed data
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_vector, index['tfidf_matrix'])
    
    # Sort the shows based on their similarity scores
    sorted_shows = sorted(zip(similarities[0], index['data']), key=lambda x: -x[0])
    
    # Select the top 5 matches (or all matches if there are less than 5)
    top_5_matches = [{'tvmaze_id': show['id'], 'showname': show['title']} for _, show in sorted_shows[:3] if _ > 0]
    
    return top_5_matches


def search_tv_shows_from_file(input_file, output_json_file, index_file_path, encoding='UTF-8'):
    try:
        # Load the index
        index = read_index(index_file_path)
        
        # Read the search query from the input file
        with open(input_file, 'r', encoding=encoding) as f:
            search_query = f.read().strip()
        
        # Search for TV shows
        matched_shows = search_tv_shows(search_query, index)
        
        # Write the matched shows to the output JSON file
        with open(output_json_file, 'w', encoding=encoding) as json_file:
            json.dump(matched_shows, json_file, ensure_ascii=False)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for TV shows based on a query")
    parser.add_argument("--input-file", required=True, help="Path to the input file with the search query")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for matched shows")
    parser.add_argument("--index-file", required=True, help="Path to the indexed data file")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")

    args = parser.parse_args()

    search_tv_shows_from_file(args.input_file, args.output_json_file, args.index_file, args.encoding)


