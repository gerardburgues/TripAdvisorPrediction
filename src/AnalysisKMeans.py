import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.manifold import TSNE


def PCAFuntion(X):
    pca = PCA(n_components=5)
    ppca = pca.fit_transform(X)
    PCA_components = pd.DataFrame(ppca)
    fig, ax = plt.subplots()
    plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black', label="PCA - 1")
    plt.scatter(PCA_components[1], PCA_components[2], alpha=.1, color='blue', label="PCA - 2")
    plt.scatter(PCA_components[2], PCA_components[3], alpha=.1, color='orange', label="PCA - 3")
    plt.scatter(PCA_components[3], PCA_components[4], alpha=.1, color='green', label="PCA - 4")
    plt.title("Cluster by PCA ")
    leg = ax.legend()
    plt.show()
    return PCA_components,ppca


def Elbowrule(Components):
    ks = range(1, 10)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(Components.iloc[:, :3])

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()


def KMeansFunc(ppca):
    n = input("Choose how many clusters you want: ")
    kmeans = KMeans(n_clusters=n)

    # predict the labels of clusters.
    label = kmeans.fit_predict(ppca)

    print(label)
    return label

def TSNEFunc(X):

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    return tsne_results
def ElbowruleTSNE(Components):
    ks = range(1, 10)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(Components[:, :3])

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()


def AnalysisKMeans(X):
    n = input("Choose how you want to reduce dimension (PCA or TSNE): ")
    if n ==  'PCA':
        PCA_Components = PCAFuntion(X)
        Elbowrule(PCA_Components[0])
        KK = KMeansFunc(PCA_Components[1])
    elif n=='TSNE':
        tsne = TSNEFunc(X)
        ElbowruleTSNE(tsne)
        KK = KMeansFunc(tsne)
    print("This is KMeans sol: " , KK)
    return KK