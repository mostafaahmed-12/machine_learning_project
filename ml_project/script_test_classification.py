import pickle
import preprocessing_classification as p

with open("count_c.pkl", "rb") as f:
    count_c = pickle.load(f)
with open("developer_encoder_c.pkl", "rb") as f:
    developer_encoder_c = pickle.load(f)
with open("geners_encoding_c.pkl", "rb") as f:
    geners_encoding_c = pickle.load(f)
with open("hot_encoding_c.pkl", "rb") as f:
    hot_encoding_c = pickle.load(f)
with open("language_encoding_c.pkl", "rb") as f:
    language_encoding_c = pickle.load(f)
with open("primary_encoder_c.pkl", "rb") as f:
    primary_encoder_c = pickle.load(f)
with open("scaler_c.pkl", "rb") as f:
    scaler_c = pickle.load(f)
with open("target_column_c.pkl", "rb") as f:
    target_encoding= pickle.load(f)
with open("svm_model.pkl", "rb") as f:
    svm=pickle.load(f)
with open("randomforest.pkl", "rb") as f:
    RandomForest=pickle.load(f)
with open("lr.pkl", "rb") as f:
    lr=pickle.load(f)


pre=p.preprocessing()
pre.developer_encoder=developer_encoder_c
pre.geners_encoding=geners_encoding_c
pre.lanauage_encoding=language_encoding_c
pre.primary_encoder=primary_encoder_c
pre.hot_encoding=hot_encoding_c
pre.scaler=scaler_c
pre.count=count_c
t=target_encoding
selected_features=['User Rating Count', 'Price', 'Developer', 'Size', 'AR', 'BG', 'BN',
       'CA', 'CS', 'CY', 'DE', 'EL', 'ES', 'FA', 'FR', 'GU', 'HI', 'HR', 'HU',
       'HY', 'ID', 'IT', 'JA', 'KN', 'KO', 'LT', 'LV', 'ML', 'MR', 'MS', 'NO',
       'PA', 'PT', 'RO', 'RU', 'SE', 'SK', 'SL', 'SQ', 'SR', 'TA', 'TE', 'TH',
       'TL', 'TR', 'UK', 'VI', 'ZH', 'ZU', 'Age Rating_12+', 'Age Rating_17+',
       'Age Rating_4+', 'Age Rating_9+', 'Sum_of_purchases',
       'Primary Genre_Book', 'Primary Genre_Business',
       'Primary Genre_Entertainment', 'Primary Genre_Finance',
       'Primary Genre_Health & Fitness', 'Primary Genre_Lifestyle',
       'Primary Genre_Medical', 'Primary Genre_Productivity',
       'Primary Genre_Reference', 'Primary Genre_Shopping',
       'Primary Genre_Social Networking', 'Primary Genre_Sports',
       'Primary Genre_Stickers', 'Primary Genre_Utilities', 'Action',
       'Adventure', 'Board', 'Books', 'Business', 'Casino', 'Casual', 'Drink',
       'Education', 'Entertainment', 'Family', 'Finance', 'Food', 'Gaming',
       'Kids', 'Lifestyle', 'Music', 'Networking', 'Playing', 'Productivity',
       'Puzzle', 'Racing', 'Reference', 'Role', 'Shopping', 'Simulation',
       'Social', 'Sports', 'Stickers', 'Travel', 'Trivia', 'Utilities']

def test_data(source):
  data=p.pd.read_csv(source)
  x = data.iloc[:, :-1]
  y = data['Rate']
  x=p.start_preprocessing_testing(x,pre)
  y=t.transform(y)
  labels1=svm.predict(x[selected_features])
  labels2=RandomForest.predict(x[selected_features])
  labels3=lr.predict(x[selected_features])
  acc1=p.accuracy_score(y,labels1)*100
  acc2=p.accuracy_score(y,labels2)*100
  acc3=p.accuracy_score(y,labels3)*100

  return acc1,acc2,acc3,len(labels1)

path="ms2-games-tas-test-v1.csv"
print(test_data(path))
