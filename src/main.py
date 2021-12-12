import nltk
import pandas as pd
from src.AnalysisKMeans import AnalysisKMeans
from src.PreProcessing import PreProcessing
from src.trainModel import trainModel

if __name__ == "__main__":
   #nltk.download('wordnet')
   #nltk.download('stopwords')
   #nltk.download('punkt')
   data = pd.read_csv('Data/tripadvisor_hotel_reviews.csv')
   AfterProcess = PreProcessing(data)
   X , y = trainModel(AfterProcess, data)
   AnalysisKMeans(X)