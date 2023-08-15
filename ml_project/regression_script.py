import pickle
import regression_models as re

import preprocessing as p
with open("Primary_Genre_TargetEncoder_Developer_regression.pkl", "wb") as f:
    Primary_Genre_TargetEncoder_Developer_regression = pickle.load(f)
with open("hot_encoding_c_regression.pkl", "wb") as f:
    hot_encoding_c_regression = pickle.load(f)
with open("scaler_c_regression..pkl", "wb") as f:
    scaler_c_regression = pickle.load(f)
with open("count_c_regression.pkl", "wb") as f:
    count_c_regression = pickle.load(f)
with open("geners_encoding_c_regression.pkl", "wb") as f:
    geners_encoding_c_regression = pickle.load(f)
with open("language_encoding_c_regression.pkl", "wb") as f:
    language_encoding_c_regression = pickle.load(f)

with open("linear_regression_all_fatures.pkl", "wb") as f:
    linear_regression_all_fatures = pickle.load(f)

with open("ridge_regression.pkl", "wb") as f:
    ridge_regression = pickle.load(f)

walid = ['User Rating Count' 'Developer' 'Size' 'Primary Genre'
         'modified_description' 'AF' 'AR' 'BG' 'BS' 'CA' 'CS' 'CY' 'DA' 'DE' 'EN'
         'ES' 'ET' 'FA' 'FI' 'FR' 'HE' 'HY' 'ID' 'JA' 'KO' 'LT' 'MK' 'MS' 'NB'
         'NN' 'PL' 'PT' 'RO' 'SE' 'SI' 'SK' 'SL' 'SQ' 'SV' 'TH' 'TR' 'VI' 'ZH'
         'ZU' 'Age Rating_12+' 'Age Rating_17+' 'Age Rating_4+' 'Age Rating_9+'
         'year ofOriginal Release Date' 'year ofCurrent Version Release Date'
         'Action' 'Adventure' 'Board' 'Books' 'Business' 'Casual' 'Education'
         'Entertainment' 'Family' 'Fitness' 'Gaming' 'Health' 'Kids' 'Lifestyle'
         'Music' 'Networking' 'News' 'Photo' 'Productivity' 'Puzzle' 'Racing'
         'Reference' 'Simulation' 'Social' 'Sports' 'Stickers' 'Travel' 'Trivia'
         'Utilities' 'Video']





trainPath="train.csv"
testPath="test.csv"

train_data =re.p.pd.read_csv(trainPath)
test_data = re.p.pd.read_csv(testPath)

X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
x_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]

x_train, x_validation, y_train, y_validation =re.p.spliting_data(X_train,Y_train)

#pre=p.preprocessing()
#pre.encoder_target=Primary_Genre_TargetEncoder_Developer_regression
#pre.developer_encoder=Primary_Genre_TargetEncoder_Developer_regression
#pre.geners_encoding=geners_encoding_c_regression
#pre.lanauage_encoding=language_encoding_c_regression
#pre.primary_encoder=Primary_Genre_TargetEncoder_Developer_regression
#pre.hot_encoding=hot_encoding_c
#pre.scaler=scaler_c
#pre.count=count_c
#=target_encoding
