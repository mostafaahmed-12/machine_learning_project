import pandas as pd
import numpy as np
import nltk
import pickle
import re
import string
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, LabelEncoder, PolynomialFeatures, OneHotEncoder,LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV

import hashlib


class preprocessing:
    def __init__(self):
        #self.y = y
        self.developer_encoder = LabelEncoder()
        self.primary_encoder = OneHotEncoder()
        self.hot_encoding = OneHotEncoder()
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
      abbrev = [re.sub(r"\.", " ", row) if isinstance(row, str) else row for row in desc]
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

   # def encoding_weights(self, train, dt):
    #    if train:
     #       self.encoder_target.fit(dt, self.y)
      #      dt = self.encoder_target.transform(dt)
       # else:
        #    dt = self.encoder_target.transform(dt)
        #return dt
   # def encoding_developer(self,train,dt):
    #    if train:
     #       self.developer_encoder.fit(dt["Developer"])
      #      dt["Developer"]=self.developer_encoder.transform(dt["Developer"])
       # else:
        #     dt["Developer"]=self.developer_encoder.transform(dt["Developer"])
        #return dt
    def encoding_primary_genre(self,train,dt):
        if train:
            self.primary_encoder.fit(dt[['Primary Genre']])
            encoded_data = self.primary_encoder.transform(dt[['Primary Genre']])
        else:
            encoded_data = self.primary_encoder.transform(dt[['Primary Genre']])
        features = pd.DataFrame(encoded_data.toarray(), columns=self.primary_encoder.get_feature_names_out(['Primary Genre']))
        dt.drop("Primary Genre", axis=1, inplace=True)

        return pd.concat([dt, features], axis=1)

    def one_hot_encoding_age_rating(self, train, dt):
        if train:
            self.hot_encoding.fit(dt[['Age Rating']])
            encoded_data = self.hot_encoding.transform(dt[['Age Rating']])
        else:
            encoded_data = self.hot_encoding.transform(dt[['Age Rating']])
        features = pd.DataFrame(encoded_data.toarray(), columns=self.hot_encoding.get_feature_names_out(['Age Rating']))
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
                self.scaler.fit(dt[[col_name]])
                scaler = self.scaler.transform(dt[[col_name]])
            else:
                scaler = self.scaler.transform(dt[[col_name]])
            dt[col_name] = scaler
        else:
            if train:
                self.count.fit(dt[[col_name]])
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

def encode_developer(df, column_name, num_bins=200):

  # determine the maximum number of unique values that can be hashed without collision
  max_unique_values = num_bins / 2

  # check if the number of unique values in the column exceeds the maximum
  if df[column_name].nunique() > max_unique_values:
      print("Warning: Number of unique values in column '{}' exceeds the maximum for safe hashing.".format(column_name))
      num_bins = df[column_name].nunique()

  # apply the hashing trick to the column
  df[column_name] = df[column_name].apply(lambda x: int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16) % num_bins)
  return df

def start_preprocessing_training(xtrain,pre):
    pre.clean_data(xtrain)
    pre.preporcessiong_for_description(xtrain)
    xtrain = pre.encoding_for_Languages(xtrain, True)
    xtrain = pre.one_hot_encoding_age_rating(True, xtrain)
    pre.min_max_scaler(True, True, xtrain, "Size")
    #pre.min_max_scaler(True, False, xtrain, "User Rating Count")
    pre.preprocessin_In_pp_Purchases(xtrain)
    pre.preprocessing_date("Original Release Date", xtrain)
    pre.preprocessing_date("Current Version Release Date", xtrain)
    pre.handleDeveloperColumn(xtrain)
    xtrain = encode_developer(xtrain, 'Developer', 500)

   # xtrain=pre.encoding_developer(True,xtrain)
    xtrain=pre.encoding_primary_genre(True,xtrain)
    xtrain = pre.encoding_for_Genres(xtrain, True)
    xtrain.drop("Description", axis=1, inplace=True)
    return xtrain

def start_preprocessing_testing(xtest,pre):
    pre.clean_data(xtest)
    pre.preporcessiong_for_description(xtest)
    xtest = pre.encoding_for_Languages(xtest, False)
    xtest = pre.one_hot_encoding_age_rating(False, xtest)
    pre.min_max_scaler(False, True, xtest, "Size")
   # pre.min_max_scaler(False, False, xtest, "User Rating Count")
    pre.preprocessin_In_pp_Purchases(xtest)
    pre.preprocessing_date("Original Release Date", xtest)
    pre.preprocessing_date("Current Version Release Date", xtest)
    pre.handleDeveloperColumn(xtest)
    xtest = encode_developer(xtest, 'Developer', 500)
    #xtest=pre.encoding_developer(False,xtest)
    xtest=pre.encoding_primary_genre(False,xtest)
    xtest = pre.encoding_for_Genres(xtest, False)
    xtest.drop("Description", axis=1, inplace=True)
    return xtest

def spliting_data(X, Y):
                                                      #change random state to = 101
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2, shuffle=True)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    x_train.reset_index(inplace=True)
    x_train.pop("index")
    x_test.reset_index(inplace=True)
    x_test.pop("index")

    y_train.reset_index(inplace=True)
    y_train.pop("index")
    y_test.reset_index(inplace=True)
    y_test.pop("index")
    print("X_train" + str(x_train.shape))
    print("y_train" + str(y_train.shape))
    print("x_test" + str(x_test.shape))
    print("y_test" + str(y_test.shape))

    return x_train, x_test, y_train, y_test
