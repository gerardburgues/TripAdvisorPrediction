from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from scoreModel import scoreNaiveBayes, scoreRandomForest


def trainModel(review_list, data):
    y = data['Rating']
    max_features = 700  # "number" most common(used) words in reviews

    count_vectorizer = CountVectorizer(max_features=max_features)

    sparce_matrix = count_vectorizer.fit_transform(review_list).toarray()
    x_train_counts = count_vectorizer.fit_transform(review_list)

    X = sparce_matrix
    splitText(X, y)
    return X

def splitText(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    chooseModel(X_train,X_test, y_train, y_test)

def chooseModel(X_train,X_test, y_train, y_test):
    val = input("Enter your model Naive Bayes (NB) or Random Forest) (RF): ")
    if val == 'NB':
        modelN = NaiveBayesModel(X_train,X_test, y_train, y_test)
        scoreNaiveBayes(X_test,y_test,modelN[0], modelN[1])
    elif val == 'RF':
        modelR = RandomForestModel(X_train,X_test, y_train, y_test)
        scoreRandomForest(X_test,y_test,modelR[0],modelR[1])
    else:
        print("Wrong Model")
def NaiveBayesModel(X_train,X_test,y_train, y_test):

    nb = GaussianNB()
    nb2 = BernoulliNB()
    nb_model = nb.fit(X_train, y_train)
    nb2_model = nb2.fit(X_train, y_train)
    return nb_model, nb2_model
def RandomForestModel(X_train,X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_predR = clf.predict(X_test)
    return y_predR,clf
