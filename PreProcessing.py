import matplotlib as plt
from gensim.parsing import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud


def DataAnalysis(data):
    print(data.head)
    data.isnull().values.any()
    data.groupby(['Rating']).size().reset_index(name='Count')

    plt.bar(data['Rating'], data['Count'])
    plt.show()


def PreProcessing(RawData):
    ReviewData = RawData['Review'].tolist()
    ReviewString = ' '.join(ReviewData)
    review_list = []
    for review in RawData['Review']:
        review = lowercase(review)
        review = stopWords(review)
        review = Lemmanize(review)
        review = Stemmer(review)
        review = ' '.join(review)
        review_list.append(review)
    return review_list


def lowercase(text):
    return text.lower()


def stopWords(text):
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(text)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    return ' '.join(wordsFiltered)

def Lemmanize(text):
    Lem = WordNetLemmatizer()
    return Lem.lemmatize(text)



def Stemmer(text):
    ps = PorterStemmer()
    CleanTextString = ps.stem(text)
    return CleanTextString.split()

def CloudWord(text):
    text = ' '.join(text)
    wordcloud = WordCloud(background_color="white",width=1024,height=768).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
