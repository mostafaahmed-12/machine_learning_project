import pandas as pd
import random

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import category_encoders as cs
import re
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("train.csv")
x_train = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
x = 'Entertainment, Strategy, Games, Family'
x_train['Genres'].fillna(x, inplace=True)

x_train["Languages"].fillna('EN', inplace=True)

#x_train["Languages"] = x_train["Languages"].apply(lambda x: "".join(x))
#x_train["Languages"] = x_train["Languages"].apply(lambda x:x.split(' '))
#print(x_train["Languages"][4427])



















def feature_selection():
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE
    model = LinearRegression()

    rfe = RFE(model, n_features_to_select=5)

    # Fit the RFE object to the data
    rfe.fit(x_train, y)

    print(rfe.ranking_)

    # Extract the selected features
    selected_features = x_train.loc[:, rfe.support_]
    print(selected_features)


import nltk.tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
def remove_comma(cell):
  new_cell=[]
  for val in cell:
    if val.isalnum():
       new_cell.append(val)
  return new_cell


x_train["Languages"]=x_train["Languages"].apply(lambda x:nltk.word_tokenize(x))
x_train["Languages"]=x_train["Languages"].apply(lambda x:remove_comma(x))
print(x_train['Languages'][4427])
#corpus = ' '.join(x_train['Languages'])
#corpus = corpus.lower()  # Convert all text to lowercase
#corpus = nltk.word_tokenize(corpus)  # Tokenize the text
#corpus = [word for word in corpus if word.isalnum()]  # Remove non-alphanumeric characters

# stop_words = set(stopwords.words('Languages'))  # Define the set of English stopwords
# corpus = [word for word in corpus if not word in stop_words]

# print(len(nltk.unique_list(corpus)))

