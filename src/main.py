import nltk

from AnalysisKMeans import AnalysisKMeans
from PreProcessing import PreProcessing
import pandas as pd

from trainModel import trainModel

if __name__ == "__main__":
   nltk.download('wordnet')
   nltk.download('stopwords')
   nltk.download('punkt')
   data  = pd.read_csv('../Data/tripadvisor_hotel_reviews.csv')
   AfterProcess = PreProcessing(data)
   X = trainModel(AfterProcess, data)
   AnalysisKMeans(X)