from rankDrugRev import RankDrugRev
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 

rev = RankDrugRev()

rev.load_data() 

rev.fit_preprocess()

X, y, idx = rev.transform_preprocess()


model, history = rev.lstm_model(X, y , pretrained = False)

rev.save_model("preprcc.dill", "lstm_model.h5", model = model)


rev, model = RankDrugRev.load_model("preprcc.dill", "lstm_model.h5")

X_test, y_test, idx_test = rev.transform_preprocess(data = 'test')

y_test_pred = model.predict(X_test)
tmp_df = pd.DataFrame({'obs' : y_test.ravel(), 'model' : y_test_pred.ravel()})





ax = sns.displot(data = tmp_df, x = 'obs', y = 'model', kind = 'kde' )

ax.ax.set_xlim(-5, 5)
ax.ax.set_ylim(-5, 5)

plt.plot(np.linspace(-5,5,100), np.linspace(-5,5,100), 'r.-')

plt.show()

transformed_seqs_, rating_, drug_OH_, cond_OH_, useC_, idx_ = rev.transform_preprocess(data = 'train')
useC_pred = model.predict([transformed_seqs_, drug_OH_, cond_OH_, rating_])

tmp_df = pd.DataFrame({'obs' : useC_.ravel(), 'model' : useC_pred_train.ravel()})




transformed_seqs_, rating_, drug_OH_, cond_OH_, useC_, idx_ = rev.transform_preprocess(data = 'train')