# TripAdvisorPrediction

With a given dataset with comments from users and the start rating, my goal will be to predict the rating start when a user enter a comment. 
To do that I'm gonna apply various NLP steps:

1. CREATE DICTIONARY WITH ALL THE WORDS

    1.1 Transform to lowercase

    1.2 Tokenize

    1.3 Remove stopwords

    1.4 Lemmanization

   
   
2. Use Sklearn to vectorize and apply TF-IDF to see the value of the words 
3. Apply Naive Bayes to predict if comment would be 1-2-3-4-5 stars
4. PCA to show graphically the scatter plot of most of the words values (Vectorization - Feature Selection)
5. Use Kmeans to show how many stars a Rating should actually be.
6. Results and Conclusions. 


CSV: https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews


 
