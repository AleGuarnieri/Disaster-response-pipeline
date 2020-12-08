import sys

import re
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

"""
This script is used to train and evaluate the Classifier model which will be
deployed into the web-app
"""

def load_data(database_filepath):
    #connecting to Database and load data
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable', engine)  

    X = df.message.values
    y = df.iloc[:, lambda df: ~df.columns.isin(['id', 'message', 'genre'])].values
    y_labels = df.iloc[:, lambda df: ~df.columns.isin(['id', 'message', 'genre'])].columns

    return X, y, y_labels

def tokenize(text):
    #remove url
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    #remove punctuation
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    #tokenize words
    tokens = word_tokenize(text)
    #remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    #lemmatize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    #using Pipeline to build classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #optimize classifier with GridSearch
    parameters = {
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False)
        #'vect__ngram_range': ((1, 1), (1, 2))
        'clf__estimator__n_estimators': [10, 50, 100],
        'clf__estimator__min_samples_split': [2, 3] 
        }
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
