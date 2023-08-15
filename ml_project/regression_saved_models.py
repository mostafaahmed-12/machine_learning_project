import regression_models as r


with open("Primary_Genre_TargetEncoder_Developer_regression.pkl", "wb") as f:
    r.pickle.dump(r.preprocessor.encoder_target, f)

with open("hot_encoding_c_regression.pkl", "wb") as f:
  r.pickle.dump(r.preprocessor.hot_encoding, f)

with open("scaler_c_regression..pkl", "wb") as f:
   r. pickle.dump(r.preprocessor.scaler, f)

with open("count_c_regression.pkl", "wb") as f:
  r.pickle.dump(r.preprocessor.count, f)

with open("geners_encoding_c_regression.pkl", "wb") as f:
 r. pickle.dump(r.preprocessor.geners_encoding, f)

with open("language_encoding_c_regression.pkl", "wb") as f:
 r. pickle.dump(r.preprocessor.lanauage_encoding, f)

with open("linear_regression_all_fatures.pkl", "wb") as f:
 r. pickle.dump(r.lr, f)

with open("ridge_regression.pkl", "wb") as f:
 r. pickle.dump(r.ridgeModel, f)

#h=r.p.preprocessing(y_train)

#h.encoder_target=
