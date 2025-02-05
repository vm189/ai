import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('tripadvisor_hotel_reviews.csv') 
def create_sentiment(rating):
    
    res = 0 # neutral sentiment
    
    if rating==1 or rating==2:
        res = -1 # negative sentiment
    elif rating==4 or rating==5:
        res = 1 # positive sentiment
        
    return res
df['Sentiment'] = df['Rating'].apply(create_sentiment)
def clean_data(review):
    
    no_punc = re.sub(r'[^\w\s]', '', review)
    no_digits = ''.join([i for i in no_punc if not i.isdigit()])
    
    return(no_digits)
df['Review'] = df['Review'].apply(clean_data)
tfidf = TfidfVectorizer(strip_accents=None, 
                        lowercase=False,
                        preprocessor=None)
X = tfidf.fit_transform(df['Review'])
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,y)
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train,y_train)
preds = lr.predict(X_test)
print("Accuracuy Score: ", accuracy_score(preds,y_test))
