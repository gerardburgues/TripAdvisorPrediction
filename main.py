from AnalysisKMeans import AnalysisKMeans
from PreProcessing import PreProcessing
import pandas as pd

from trainModel import trainModel

if __name__ == "__main__":
   data  = pd.read_csv('tripadvisor_hotel_reviews.csv')
   AfterProcess = PreProcessing(data)
   X = trainModel(AfterProcess, data)
   AnalysisKMeans(X)