import numpy as np

import preprocessing as pe

data=pe.pd.read_csv("train.csv")
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train, x_validation, y_train, y_validation=pe.spliting_data(x,y)
preprocessor=pe.preprocessing(y_train)
x_train=pe.start_preprocessing_training(x_train,preprocessor)
x_validation=pe.start_preprocessing_testing(x_validation,preprocessor)

test=pe.pd.read_csv('test.csv')
x_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]
y_test=np.array(y_test).reshape(-1,1)
x_test=pe.start_preprocessing_testing(x_test,preprocessor)


x_train=pe.feature_selection_rfe(x_train,True,y_train)
x_validation=pe.feature_selection_rfe(x_validation,False,y_train)
y_train=pe.np.array(y_train).reshape(-1,1)
y_validation=pe.np.array(y_validation).reshape(-1,1)
print(x_train.shape)
lasso = pe.Lasso(0.550562449002455)

# Fit model to training data
lasso.fit(x_train, y_train)

# Make predictions on testing data
y_pred_train = lasso.predict(x_train)
val = lasso.predict(x_validation)


# Evaluate model performance using mean squared error
mse =pe. mean_squared_error(y_train, y_pred_train)
print("Mean squared error tr:", mse)
mse =pe. mean_squared_error(y_validation, val)
print("Mean squared error v:", mse)
