import pickle

import regresssion_preprocessing as p

feature_to_select_var = 5

model = p.LinearRegression()

#rfe = p.RFE(model, n_features_to_select = feature_to_select_var)

def feature_selection_rfe(xtrain, t,y, rfe):

    if t:
        # Fit the RFE object to the data
        rfe.fit(xtrain, y)
        # Extract the selected features
        xtrain = rfe.transform(xtrain)
    else:
        xtrain = rfe.transform(xtrain)
    return xtrain



preprocessor=p.preprocessing(y_train)

x_train = p.start_preprocessing_training(x_train,preprocessor)
print(x_train.shape)

x_validation=p.start_preprocessing_testing(x_validation,preprocessor)
print(x_validation.shape)

x_test=p.start_preprocessing_testing(x_test,preprocessor)

print(x_test.shape)

lr= p.LinearRegression()
lr.fit(x_train, y_train)
y_pred_train = lr.predict(x_train)
y_pred_validat=lr.predict(x_validation)
y_pred_test=lr.predict(x_test)
print("train erorr",p.mean_squared_error(y_train, y_pred_train))
print("validation erorr",p.mean_squared_error(y_validation, y_pred_validat))
print("testing erorr",p.mean_squared_error(y_test, y_pred_test))


#alpha_values = [0.0001,0.001, 0.01, 0.1, 1, 10,30,50,70,100,130,150,200,250,300,400,500,600,700, 1000]
#n_features_values = [2,3,5,7,10,20,30,50,60,80,90,100,110,120,130,140,150]
#best_alpha = None
#best_n_features = None
#est_mse = np.inf
'''
for alpha in alpha_values:
    for n_features in n_features_values:
        break
        # Fit the Ridge model and RFE selector
        ridgeModel = Ridge(alpha=alpha)
        rfe = RFE(ridgeModel, n_features_to_select=n_features)
        x_Train_selected = feature_selection_rfe(x_train, True,  y_train, rfe)
        x_Test_selected = feature_selection_rfe(x_validation, False,  y_validation,rfe)

        # Train the model on the selected features
        ridgeModel.fit(x_Train_selected, y_train)

        # Predict the output variable for the validation data
        y_pred = ridgeModel.predict(x_Test_selected)

        # Calculate the mean squared error of the predictions
        mse = mean_squared_error(y_validation, y_pred)

        # Update the best alpha and number of features if the current MSE is lower
        if mse < best_mse:
            best_alpha = alpha
            best_n_features = n_features
            best_mse = mse
            print(best_alpha, " ", best_n_features, " ", best_mse, "\n")
'''



best_alpha = 70
best_n_features = 80
ridgeModel = p.Ridge(alpha=best_alpha)
rfe=p.RFE(ridgeModel, n_features_to_select=best_n_features)
x_Train_selected = feature_selection_rfe(x_train, True,  y_train, rfe)
x_val_selected = feature_selection_rfe(x_validation, False,  y_validation,rfe)
x_test_selected=feature_selection_rfe(x_test, False,  y_validation,rfe)
ridgeModel.fit(x_Train_selected, y_train)
y_pred_train = ridgeModel.predict(x_Train_selected)
y_pred_validat=ridgeModel.predict(x_val_selected)
y_pred_test=ridgeModel.predict(x_test_selected)
print("Mean squared error train:",p. mean_squared_error(y_train,y_pred_train))
print("Mean squared error validation:",p.mean_squared_error( y_validation,y_pred_validat))
print("Mean squared error test:",p.mean_squared_error( y_test,y_pred_test))

