from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split

from src import *


if __name__ == "__main__":
    print("You can choose 2 options: \n run a new model from src or show a model already executed")
    val = input("Do you want to train a new model? ")

    if val == 'SI':
        exec(open('../src/main.py').read())
    else:
        var = input("Do you want RF or NV")
        if( var == 'RF'):
            clf = load('../randomForest.joblib')
        else:

            clf  = load('../NaiveBayes.joblib')
        #data = pd.read_csv('../Data/tripadvisor_hotel_reviews.csv')
        #AfterProcess = PreProcessing(data)
        #X ,y = trainModel(AfterProcess, data)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data = input("Write Path")

        data = pd.read_csv(data)
        count_vectorizer = CountVectorizer(max_features=700)
        """
        You may need to change this to add the name of your columns insteaad of Review
        """
        sparce_matrix = count_vectorizer.fit_transform(data['Review']).toarray()
        x_train_counts = count_vectorizer.fit_transform(data['Review'])
        X = sparce_matrix
        y_predR = clf.predict(X)
        print(y_predR)


