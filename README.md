# TripAdvisorPrediction 
### Gerard Burgués
### Tripadvisor DataSet
### https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews

### Overview
Aquest dataset utlitza dades de comentaris de Tripadvisor amb les seves estrelles corresponents que els usuaris han afegit. Tenim 2 columne
This dataset uses data from a TripAdavisor comment rates. Includes 2 columns: The first one are comments of each user that gives an opinion. The second one are the starts of each comment. 

### Goal
The goal of this project is analyse the given data and to predict using different algorithms how many stars a given comment would be. 
An extra exercise I would be implementing is: How many stars the comments should be using?

### PreProcess and Model 
With a given dataset with comments from users and the start rating, my goal will be to predict the rating start when a user enter a comment. 
To do that I'm gonna apply various NLP steps:

1. CREATE DICTIONARY WITH ALL THE WORDS

    1.1 Transform to lowercase

    1.2 Tokenize

    1.3 Remove stopwords

    1.4 Lemmanization

   
   
2. Use Sklearn to vectorize and apply TF-IDF to see the value of the words 
3. Apply 
    3.1 Naive Bayes 
    3.2 " -...---"  
    to predict if comment would be 1-2-3-4-5 stars.
    
5. PCA to show graphically the scatter plot of most of the words values (Vectorization - Feature Selection)
6. Use Kmeans to show how many stars a Rating should actually be.
7. Results and Conclusions. 


CSV: https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews


 
