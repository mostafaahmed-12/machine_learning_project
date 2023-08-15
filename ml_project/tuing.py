#pca=pe.PCA(100)
#pca.fit(x_train)
#x_train=pca.transform(x_train)
#x_validation=pca.transform(x_validation)
x_train=pe.feature_selection_rfe(x_train,True,y_train)
x_validation=pe.feature_selection_rfe(x_validation,False,y_train)
y_train=pe.np.array(y_train).reshape(-1,1)
y_validation=pe.np.array(y_validation).reshape(-1,1)

s=pe.PolynomialFeatures(2)
x_train=s.fit_transform(x_train)
x_validation=s.transform(x_validation)
lr=pe.LinearRegression()
lr.fit(x_train,y_train)
pre=lr.predict(x_validation)
print(pe.mean_squared_error(y_validation,pre))

lasso = pe.Lasso(alpha=0.01)

# Fit model to training data
lasso.fit(x_train, y_train)

# Make predictions on testing data
y_pred = lasso.predict(x_validation)

# Evaluate model performance using mean squared error
mse =pe. mean_squared_error(y_validation, y_pred)
print("Mean squared error:", mse)

# Perform 5-fold cross-validation on the training set
scores =pe.cross_val_score(lasso, x_train, y_train, cv=10, scoring='neg_mean_squared_error')

# Print the mean squared error for each fold
print("Mean squared error for each fold:", -scores)

# Print the average mean squared error across all folds
print("Average mean squared error:", -scores.mean())

ridge =pe. Ridge(alpha=0.00001)

# Set up a parameter grid for the regularization parameter 'alpha'
param_grid = {'alpha': pe.np.logspace(-5, 5, 50)}

# Create a GridSearchCV object
grid_search = pe.GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(x_train, y_train)
# Get the best hyperparameters
best_params = grid_search.best_params_

# Predict using the best model
y_pred = grid_search.predict(x_validation)

# Calculate the mean squared error
mse = pe.mean_squared_error(y_validation, y_pred)
print('Mean squared error:', mse)

model1 =pe. LinearRegression()

# Define the hyperparameter space
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}

# Create the grid search object
model_grid = pe.GridSearchCV(estimator=model1,
                          param_grid=param_grid,
                          cv=5,
                          verbose=1)

# Fit the grid search object to the data
model_grid.fit(x_train, y_train)

# Print the best hyperparameters and corresponding score
print("Best hyperparameters: ", model_grid.best_params_)
print("Best score: ", model_grid.best_score_)
