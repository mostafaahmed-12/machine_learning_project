from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, LabelEncoder, PolynomialFeatures
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split, cross_val_score,GridSearchCV
from sklearn.feature_selection import f_regression, RFE, SelectKBest, chi2
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import category_encoders as cs
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import nltk.tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import re
import string
#from numba import njit, jit
import seaborn as sns
from numpy.ma.core import absolute
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDRegressor

class preprocessing:
    def __init__(self, y):
        self.y = y
        self.encoder_target = cs.TargetEncoder(cols=["Primary Genre", 'Developer'])
        self.hot_encoding = OneHotEncoder(sparse_output=False)
        self.scaler = MinMaxScaler()
        self.count = MinMaxScaler()
        self.geners_encoding = MultiLabelBinarizer(sparse_output=False)
        self.lanauage_encoding = MultiLabelBinarizer(sparse_output=False)

    def clean_data(self, dt):
        cols = ["URL", "ID", "Name", "Icon URL", "Subtitle"]
        for x in cols:
            dt.drop(x, axis=1, inplace=True)
        dt["Languages"].fillna('EN', inplace=True)
        dt["Price"].fillna(dt["Price"].mean(), inplace=True)
        dt["Size"].fillna(dt["Size"].mean(), inplace=True)
        dt["Age Rating"].fillna('4+', inplace=True)
        dt['Primary Genre'].fillna('Games', inplace=True)
        x = 'Entertainment, Strategy, Games, Family'
        dt['Genres'].fillna(x, inplace=True)
        dt["Developer"].fillna("Tapps Tecnologia da Informa\\xe7\\xe3o Ltda.", inplace=True)
        dt['Original Release Date'].fillna("2/9/2016", inplace=True)
        dt["Current Version Release Date"].fillna("29/07/2019", inplace=True)

    def preporcessiong_for_description(self, dt):

      desc = dt['Description']
      size = len(desc)

      # remove '\.'
      abbrev = [re.sub(r"\.", " ", row) for row in desc]

      # apply tokenization
      tokens = (nltk.word_tokenize(row) for row in abbrev)

      # apply normaliztion
      # 1: convert to lowerCase
      # 2: remove punctuation
      normalized = [[token.translate(str.maketrans('', '', string.punctuation)).lower() for token in toks] for toks in tokens]

      # remove '', ' '
      normalized = list(list(token for token in row if (token != '' and token !=' ')) for row in normalized)

      # remove stop words
      stop_words = {word for word in stopwords.words("english")}
      filtered = [[word for word in row if word not in stop_words] for row in normalized]

      # apply lemmatized
      lemmatizer =WordNetLemmatizer()
      lemmatized = ((lemmatizer.lemmatize(word) for word in row) for row in filtered)

      # set unique words
      uniqwords = list(set(row) for row in lemmatized)

      # divid # of unique words by size of text after normalization and remove white space
      dt['modified_description'] = list(len(uniqwords[i])/len(normalized[i]) for i in range(size))


    # text to numerical
    def convert_to_numric(self, x):
        e = []
        for i in x:
            e.append(float(i))
            return e

    def preprocessin_In_pp_Purchases(self, dt):
        dt["In-app Purchases"].fillna(0, inplace=True)

        dt["In-app Purchases"] = dt["In-app Purchases"].astype(str)

        dt["In-app Purchases"] = dt["In-app Purchases"].apply(lambda x: x.split(','))

        dt["In-app Purchases"] = dt["In-app Purchases"].apply(lambda x: self.convert_to_numric(x))

        dt["Sum_of_purchases"] = dt["In-app Purchases"].apply(lambda x: sum(x))
        dt.drop("In-app Purchases", axis=1, inplace=True)

    def removeAbbreviation(self, companyName):
        companyName += " "
        return re.sub('(\.| |,)(inc|ltd|llc|corp|co|llp|pc|ltda|gmbh|sarl|i\.n\.c|s\.r\.l)(\.| |,|\b)', '', companyName)

    def removePuncAndSpecialChar(self, companyName):
        return re.sub('(\\|\/|\(|\)|\.?=[^a-zA-Z]|,|")', '', companyName)

    def handleDeveloperColumn(self, df):
        df["Developer"] = df["Developer"].astype(str)
        df['Developer'].apply(self.removeAbbreviation)
        df['Developer'].apply(self.removePuncAndSpecialChar)

    def shortenURL_Regex(URL):
      return re.sub('.*app/|id|-|/','',URL)

    def shortenURL(self,df):
      df['URL'] =df['URL'].apply(self.shortenURL_Regex)

    def getIconDimension(URL):
      return re.sub('.*source/','',URL)

    def shortenPicURL(self,df):
      df['Icon URL'] = df['Icon URL'].apply(self.getIconDimension)

    def encoding_weights(self, train, dt):
        if train:
            self.encoder_target.fit(dt, self.y)
            dt = self.encoder_target.transform(dt)
        else:
            dt = self.encoder_target.transform(dt)
        return dt

    def one_hot_encoding_age_rating(self, train, dt):
        if train:
            self.hot_encoding.fit(dt[['Age Rating']], self.y)
            encoded_data = self.hot_encoding.transform(dt[['Age Rating']])
        else:
            encoded_data = self.hot_encoding.transform(dt[['Age Rating']])
        features = pd.DataFrame(encoded_data, columns=self.hot_encoding.get_feature_names_out(['Age Rating']))
        dt.drop("Age Rating", axis=1, inplace=True)

        return pd.concat([dt, features], axis=1)

    def encoding_for_Genres(self, data, train):
        data["Genres"] = data["Genres"].apply(lambda x: nltk.word_tokenize(x))
        data["Genres"] = data["Genres"].apply(lambda x: self.remove_comma(x))
        if train:
            self.geners_encoding.fit(data["Genres"])
            encoded_data = self.geners_encoding.transform(data["Genres"])
        else:
            encoded_data = self.geners_encoding.transform(data["Genres"])
        features = pd.DataFrame(encoded_data, columns=self.geners_encoding.classes_)
        data.drop("Genres", axis=1, inplace=True)
        return pd.concat([data, features], axis=1)

    def preprocessing_date(self, date_name, df):
        df[date_name] = pd.to_datetime(df[date_name], dayfirst=True)
        df['year of' + str(date_name)] = df[date_name].dt.year
        df.drop(date_name, axis=1, inplace=True)

    def min_max_scaler(self, train, size, dt, col_name):
        if size:
            if train:
                self.scaler.fit(dt[[col_name]], self.y)
                scaler = self.scaler.transform(dt[[col_name]])
            else:
                scaler = self.scaler.transform(dt[[col_name]])
            dt[col_name] = scaler
        else:
            if train:
                self.count.fit(dt[[col_name]], self.y)
                scaler = self.count.transform(dt[[col_name]])
            else:
                scaler = self.count.transform(dt[[col_name]])
            dt[col_name] = scaler

    def remove_comma(self, cell):
        new_cell = []
        for val in cell:
            if val.isalnum():
                new_cell.append(val)
        return new_cell

    def encoding_for_Languages(self, data, train):
        data["Languages"] = data["Languages"].apply(lambda x: nltk.word_tokenize(x))
        data["Languages"] = data["Languages"].apply(lambda x: self.remove_comma(x))

        if train:
            self.lanauage_encoding.fit(data["Languages"])
            encoded_data = self.lanauage_encoding.transform(data["Languages"])
        else:
            encoded_data = self.lanauage_encoding.transform(data["Languages"])
        features = pd.DataFrame(encoded_data, columns=self.lanauage_encoding.classes_)
        data.drop("Languages", axis=1, inplace=True)
        return pd.concat([data, features], axis=1)

def start_preprocessing_training(xtrain,pre):
    pre.clean_data(xtrain)
    pre.preporcessiong_for_description(xtrain)
    xtrain = pre.encoding_for_Languages(xtrain, True)
    xtrain = pre.one_hot_encoding_age_rating(True, xtrain)
    pre.min_max_scaler(True, True, xtrain, "Size")
    pre.min_max_scaler(True, False, xtrain, "User Rating Count")
    pre.preprocessin_In_pp_Purchases(xtrain)
    pre.preprocessing_date("Original Release Date", xtrain)
    pre.preprocessing_date("Current Version Release Date", xtrain)
    pre.handleDeveloperColumn(xtrain)
    xtrain = pre.encoding_weights(True, xtrain)
    xtrain = pre.encoding_for_Genres(xtrain, True)
    xtrain.drop("Description", axis=1, inplace=True)
    return xtrain


def start_preprocessing_testing(xtest,pre):
    pre.clean_data(xtest)
    pre.preporcessiong_for_description(xtest)
    xtest = pre.encoding_for_Languages(xtest, False)
    xtest = pre.one_hot_encoding_age_rating(False, xtest)
    pre.min_max_scaler(False, True, xtest, "Size")
    pre.min_max_scaler(False, False, xtest, "User Rating Count")
    pre.preprocessin_In_pp_Purchases(xtest)
    pre.preprocessing_date("Original Release Date", xtest)
    pre.preprocessing_date("Current Version Release Date", xtest)
    pre.handleDeveloperColumn(xtest)
    xtest = pre.encoding_weights(False, xtest)
    xtest = pre.encoding_for_Genres(xtest, False)
    xtest.drop("Description", axis=1, inplace=True)
    return xtest


def spliting_data(X, Y):
                                                      #change random state to = 101
    x_train, x_validation, y_train, y_validation = train_test_split(X, Y, random_state=42, test_size=0.15, shuffle=True)
    y_train = pd.DataFrame(y_train)
    y_validation = pd.DataFrame(y_validation)

    x_train.reset_index(inplace=True)
    x_train.pop("index")
    x_validation.reset_index(inplace=True)
    x_validation.pop("index")

    y_train.reset_index(inplace=True)
    y_train.pop("index")
    y_validation.reset_index(inplace=True)
    y_validation.pop("index")
    print("X_train" + str(x_train.shape))
    print("y_train" + str(y_train.shape))
    print("x_validation" + str(x_validation.shape))
    print("y_validation" + str(y_validation.shape))

    return x_train, x_validation, y_train, y_validation
