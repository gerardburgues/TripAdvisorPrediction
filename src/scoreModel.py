from sklearn.metrics import recall_score, precision_score


def scoreNaiveBayes(X_test,y_test,nb_model, nb2_model):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

    y_pred = nb_model.predict(X_test)
    y_pred2 = nb2_model.predict(X_test)

    print("Accuracy:", recall_score(y_test, y_pred, average='micro'))
    print("Precision:", precision_score(y_test, y_pred, average="micro"))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("**************************************************************")
    print("Accuracy_NB2:", recall_score(y_test, y_pred2, average='micro'))
    print("Precision_NB2:", precision_score(y_test, y_pred2, average="micro"))
    print("ConfusionMatrix 2: \n ", confusion_matrix(y_test, y_pred2))

def scoreRandomForest(X_test, y_test, y_predR ,clf):
    clf.score(X_test, y_test)
    print("Accuracy:", recall_score(y_test, y_predR, average='micro'))
    print("Precision:", precision_score(y_test, y_predR, average="micro"))
