from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import preprocessing_classification as p
from imblearn.under_sampling import TomekLinks

path = "games-classification-dataset (1).csv"
data = p.pd.read_csv(path)
x = data.iloc[:, :-1]
y = data['Rate']
x_train, x_test, y_train, y_test = p.spliting_data(x, y)

pre = p.preprocessing()

x_train = p.start_preprocessing_training(x_train, pre)
x_test = p.start_preprocessing_testing(x_test, pre)
balance = True
target_column_encoder = p.LabelEncoder()

if balance == True:
    # handel imbalanced data
    tomek = TomekLinks(sampling_strategy='auto')
    x_train, y_train = tomek.fit_resample(x_train, y_train)

# feature selection
selector = SelectKBest(chi2, k=100)
selector.fit(x_train, y_train)
selected_indices = selector.get_support(indices=True)
selected_features = x_train.columns[selected_indices]

# encoding train
y_train = target_column_encoder.fit_transform(y_train)
y_test = target_column_encoder.transform(y_test)

c = [1, 2, 3]
kernel = ['linear', 'poly', 'rbf']
for i in c:
    for j in kernel:
        svm_model = SVC(C=i, kernel=j, max_iter=1000000)
        model2 = OneVsRestClassifier(svm_model)
        model2.fit(x_train[selected_features], y_train)
        # Make predictions on the test set
        y_pred = model2.predict(x_test[selected_features])
        # Calculate the accuracy score
        acc1 = accuracy_score(y_test, y_pred)
        # Print the accuracy score
        print("c :", i, "kernel :", j, 'Accuracy score: {:.2f}'.format(acc1))

final_svm_model = SVC(C=3, kernel='rbf', max_iter=1000000)

final_svm_model.fit(x_train[selected_features], y_train)
y_pred = final_svm_model.predict(x_test[selected_features])

acc1 = accuracy_score(y_test, y_pred)

print('Accuracy score: {:.2f}'.format(acc1))

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5)

grid_search.fit(x_train[selected_features], y_train)

print("Best hyperparameters: ", grid_search.best_params_)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(x_test[selected_features])

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}".format(accuracy))



best_params = {
    'C': 1,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'multi_class': 'auto',
    'class_weight': 'balanced',
    'max_iter': 100
}
param_grid2 = {
    'C': [0.001,0.01,0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear','sag'],
    'max_iter': [100, 500, 1000],
    'penalty': ['l1', 'l2', None]

}


def logisticTuning(dict2, current_hyperparameter, tuning_list, x_train,y_train, x_test,y_test,features):
  max_acc_test = -1e9
  max_acc_train = -1e9
  max_f1 = -1e9
  bestParamList = []
  dict = dict2.copy() # take copy-by-value only to not override the original dict
  print('current hyperparameter :  ', current_hyperparameter, "\n\n")
  for c in tuning_list:
    dict[current_hyperparameter] = c
    if dict['solver'] == 'lbfgs':
      dict['penalty'] = 'l2'
      print("NEW DICTIONARY : ", dict)

    print("before operations :  ", dict, "\n\n")
    model = LogisticRegression(C=dict['C'],max_iter=dict['max_iter'],solver=dict['solver'], penalty= dict['penalty'])
    ovrModel = OneVsRestClassifier(model)
    ovrModel.fit(x_train[features],y_train)
    y_pred_tst = ovrModel.predict(x_test[features])
    y_pred = ovrModel.predict(x_train[features])

    accuracy = accuracy_score(y_test, y_pred_tst)
    accuracy2 = accuracy_score(y_train, y_pred)
    # print the accuracy
    print('Accuracy for test:', accuracy,"accuracy for train: ",accuracy2, "\n")
    # calculate the F1 score of the model
    f1 = f1_score(y_test, y_pred_tst, average='micro')
    # print the F1 score
    print('F1-score:', f1, "\n\n\n")
    if max_acc_test < accuracy:
      max_acc_test = accuracy
      max_acc_train = accuracy2
      max_f1 = f1
      bestParamList = [dict['C'],dict['max_iter'],dict['solver'],dict['penalty']]
  return [max_acc_test, max_acc_train, max_f1, bestParamList]

max_acc_test = -1e9
max_acc_train = -1e9
max_f1  = -1e9
bestList = []
bestOVR = OneVsRestClassifier(LogisticRegression())
for key in param_grid2:
  print(key)
  print(best_params)
  ret = logisticTuning(best_params, key, param_grid2[key], x_train,y_train,x_test,y_test, selected_features)
  print('------------------------------------------------')
  print('RETURN VALUE : ', ret)
  print('MAX FOR NOW : ', max_acc_test)
  if max_acc_test < ret[0]:
    max_acc_test = ret[0]
    max_acc_train = ret[1]
    max_f1 = ret[2]
    bestList = ret[3]
print('MAXIMUM RESULT IS : ', max_acc_test, " ", max_acc_train, "  \n", max_f1, "\n\n\n\n")
print(bestList)
