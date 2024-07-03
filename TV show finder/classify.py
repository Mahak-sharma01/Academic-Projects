


import tensorflow as tf
import pickle
import argparse
import re
import nltk
import lime
from lime import lime_text
import numpy as np

# Load model and lookup table
def load_model_and_lookup():
    model = tf.keras.models.load_model('model')
    with open('categories_forward_lookup.pkl', 'rb') as f:
        categories_forward_lookup = pickle.load(f)
    return model, categories_forward_lookup

# Read show description from file
def read_description(file_path):
    with open(file_path, 'r') as f:
        description = f.read()
    return description

# Predict genre of a show based on its description
def predict_genre(model, categories_forward_lookup, description, threshold=0.5):
    predictions = model.predict([description])[0]
    genres = {genre: predictions[i] for genre, i in categories_forward_lookup.items() if predictions[i] >= threshold}
    sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
    return genres, sorted_genres[:3]

# Save predicted genres to a file
def save_results(genres, top_genres):
    with open('predicted_genres.txt', 'w') as f:
        
        f.write("\nTop 3 Genres:\n")
        for i, (genre, prob) in enumerate(top_genres, start=1):
            f.write(f"{i}. {genre}: {prob:.2f}\n")
        
        f.write("All Predicted Genres :\n")
        for genre, prob in genres.items():
            f.write(f"{genre}: {prob:.2f}\n")


# Clean text data
def text_cleaning(text):
    text = re.sub(r'http\S+', '', text)  # Example: Remove URLs
    text = ' '.join(text.split())
    return text

# Remove stopwords
def remove_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Tokenize and lemmatize text
def tokenize_and_lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text = text.lower()
    tokens = nltk.tokenize.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Preprocess description
def preprocessing_description(description):
    description = text_cleaning(description)
    description = remove_stopwords(description)
    description = tokenize_and_lemmatize(description)
    return description

# Function to make predictions
def predict_fn(texts):
    descriptions = [preprocessing_description(description) for description in texts]
    predictions = model.predict(descriptions)
    return predictions

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify the genre of a show based on its description")
    parser.add_argument("--description-file", required=True, help="Path to the text file containing the show description")
    args = parser.parse_args()

    model, categories_forward_lookup = load_model_and_lookup()
    description = read_description(args.description_file)
    genres, top_genres = predict_genre(model, categories_forward_lookup, description, threshold=0.5)
    save_results(genres, top_genres)
    print(top_genres)
    print("Genre predictions saved to predicted_genres.txt")

    # Create a LIME explainer object
    class_names = list(categories_forward_lookup.keys())
    explainer = lime_text.LimeTextExplainer(class_names=class_names)

    # Explain the prediction
    exp = explainer.explain_instance(description, predict_fn, num_features=4)
    
    # Save the explanation to a file
    exp.save_to_file('explanation.html')

    # Show the explanation in the console
    print('Explanation of predicted genres:')
    print('\n'.join(map(str, exp.as_list())))

