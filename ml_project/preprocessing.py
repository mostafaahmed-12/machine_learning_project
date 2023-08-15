import pandas as pd
import random
from sklearn.model_selection import train_test_split
import category_encoders as cs
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import nltk.tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge




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


def start_preprocessing_training(xtrain, pre):
    pre.clean_data(xtrain)
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


def start_preprocessing_testing(xtest, pre):
    pre.clean_data(xtest)
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

remodel = LinearRegression()
rfe = RFE(remodel, n_features_to_select=10)


def feature_selection_rfe(xtrain, t, y):
    if t:
        # Fit the RFE object to the data
        rfe.fit(xtrain, y)
        # Extract the selected features
        xtrain = rfe.transform(xtrain)
    else:
        xtrain = rfe.transform(xtrain)
    # f = rfe.get_feature_names_out(xtrain.columns)
    # print(f)
    return xtrain


def anova(xtrain, ytrain):
    f_values, p_values = f_regression(xtrain, ytrain)

    selected_features = []
    for i in range(len(xtrain.iloc[0, :])):
        if p_values[i] < 0.0007:
            selected_features.append(i)
    colums = xtrain.columns
    c = []
    for i in selected_features:
        c.append(colums[i])
    print(c)

    return c

