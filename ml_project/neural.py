import tensorflow as tf
from keras import Sequential
from  keras.layers import Dense,InputLayer,Dropout
from keras.regularizers import l2

import preprocessing as pe


data=pe.pd.read_csv("train.csv")
test=pe.pd.read_csv("test.csv")
x_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]

x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train, x_validation, y_train, y_validation=pe.spliting_data(x,y)
preprocessor=pe.preprocessing(y_train)
x_train=pe.start_preprocessing_training(x_train,preprocessor)
x_validation=pe.start_preprocessing_testing(x_validation,preprocessor)
x_train=pe.np.array(x_train)
y_train-pe.np.array(y_train)
x_validation=pe.np.array(x_validation)
y_validation=pe.np.array(y_validation)
x_test=pe.start_preprocessing_testing(x_test,preprocessor)
x_test=pe.np.array(x_test)
y_test=pe.np.array(y_test)
x_train=pe.feature_selection_rfe(x_train,True,y_train)
x_validation=pe.feature_selection_rfe(x_validation,False,y_train)

model=Sequential()
model.add(InputLayer(input_shape=(x_train.shape[1])))
model.add(Dense(7,activation="relu"))
model.add(Dense(4,activation="relu"))
#model.add(Dense(7,activation="relu"))




model.add(Dense(1,activation='linear'))

model.compile(loss='mse',
                optimizer=tf.keras.optimizers.SGD(0.01), metrics=['mse'])

hist=model.fit(x_train, y_train,validation_data=(x_validation,y_validation),epochs=20, batch_size=28)

x_test=pe.feature_selection_rfe(x_test,False,y_train)

y_pred = model.predict(x_test)

pe.plt.plot(hist.history['loss'], label='train')
pe.plt.plot(hist.history['val_loss'], label='test')
pe.plt.title('Model Training and Validation Loss')
pe.plt.xlabel('Epoch')
pe.plt.ylabel('Loss')
pe.plt.legend()
pe.plt.show()


# Evaluate model performance using mean squared error
mse =pe.mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)


y_pred =y_pred.flatten()
print(pe.r2_score(y_test,y_pred))
a = pe.plt.axes(aspect='equal')
pe.plt.scatter(y_test, y_pred)
pe.plt.xlabel('True Values ')
pe.plt.ylabel('Predictions ')
lims = [0,10]
pe.plt.xlim(lims)
pe.plt.ylim(lims)
_ = pe.plt.plot(lims, lims)
pe.plt.show()


