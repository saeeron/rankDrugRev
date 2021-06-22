import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, Concatenate, Flatten 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


import requests
import zipfile
import io   

import dill


class RankDrugRev:
	""" this class instantiates objects that can preprocess and model usefulness of reviews on drugs
	across a vroad range of drugs and conditions """

	data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip'

	def __init__(self, num_words = 5000, max_len = 2000, len_limit = None):
		self.num_words = num_words
		self.len_limit = len_limit
		self.max_len = max_len

		self._km_fitted = False
		self._seq_fitted = False



	def load_data(self):
		""" applying this function on self results in loading and storing data from online source, 
		as pandas dataframe """

		r = requests.get(RankDrugRev.data_url, verify = False)
		z = zipfile.ZipFile(io.BytesIO(r.content))
		df_train = pd.read_csv(io.BytesIO(z.read("drugsComTrain_raw.tsv")),sep = '\t', parse_dates = [5])
		df_train.rename(columns={"Unnamed: 0": "id"}, inplace = True)

		df_test = pd.read_csv(io.BytesIO(z.read("drugsComTest_raw.tsv")),sep = '\t', parse_dates = [5])
		df_test.rename(columns={"Unnamed: 0": "id"}, inplace = True)
	
		self.raw_data_train = df_train
		self.raw_data_test = df_test

		self._raw_data_loaded = True

	def fit_preprocess(self):
		""" applying this function on self fits a preprocessing pipline on the train dataset loaded
		by executing \"load_data(self)\" """

		if not self._raw_data_loaded:
			raise ValueError('raw data needs to be loaded first! ; execute \"load_data()\"')

		df = self.raw_data_train

		df = df.loc[~df["condition"].str.contains("</span>").astype(bool),:]

		low_cond_list = df["condition"].value_counts()[df["condition"].value_counts() < 50].index.values.tolist()
		low_drug_list = df["drugName"].value_counts()[df["drugName"].value_counts() < 50].index.values.tolist()

		idx_ = (~df["condition"].isin(low_cond_list)) & (~df["drugName"].isin(low_drug_list))

		df = df.loc[idx_,:]
		df.dropna(inplace = True)

		enc_cond = OneHotEncoder(handle_unknown='ignore')
		enc_cond.fit(df["condition"].values.reshape(-1,1))

		enc_drug = OneHotEncoder(handle_unknown='ignore')
		enc_drug.fit(df["drugName"].values.reshape(-1,1))

		tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = self.num_words, oov_token = "<UNK>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
		tokenizer.fit_on_texts(df["review"].values)
		tokenizer.index_word[0] = '<PAD>'

		df.loc[df["usefulCount"] == 0, "usefulCount"] = 0.1  # box_cox transformation requires strictly positive data 
		tmp = df["usefulCount"].values.reshape((-1,1))
		pw_useC = PowerTransformer(method='box-cox')
		pw_useC.fit(tmp)

		
		self._tokenizer = tokenizer
		self._enc_cond = enc_cond
		self._enc_drug = enc_drug

		self._pw_useC = pw_useC

		self._low_cond_list, self._low_drug_list = low_cond_list, low_drug_list

		self._seq_fitted = True


	def transform_preprocess(self, data = 'train', forModelorTest = True):
		"""

		Parameters
		----------
		data : 
			Default value = 'train') or 'train' for loaded datasets. Pass in a pandas dataframe if applying on an external dataset
										dataframe should include columns = "drugName", "condition", "review", "rating", and "usefulCount"
										if forModelorTest = False, usefulCount is not needed  
		forModelorTest :
			Default value = True) indicates whether 'data' has column usefulCount or not

		Returns
		-------
		A list of predictors for the model and usefulCounts as model output and index of 'data' for which preprocessing has an answer

		"""

		if isinstance(data, pd.DataFrame):
			df = data
		elif (data == 'train') & (self._raw_data_loaded) & (self._seq_fitted):
			df = self.raw_data_train
		elif (data == 'test') & (self._raw_data_loaded) & (self._seq_fitted):
			df = self.raw_data_test
		else:
			raise ValueError("no data are loaded!")

		index_orig = np.array(df.index)
		df = df.loc[~df["condition"].str.contains("</span>").astype(bool),:]

		enc_drug = self._enc_drug
		enc_cond = self._enc_cond
		
		drug_OH = enc_drug.transform(df["drugName"].values.reshape(-1,1)).toarray()
		cond_OH = enc_cond.transform(df["condition"].values.reshape(-1,1)).toarray()

		idx_1 = np.where(((drug_OH.sum(axis=1) == 0) | (cond_OH.sum(axis=1) == 0)).ravel())[0]
		idx_2 = np.where((df['rating'] < 0) | (df['rating'] > 10).values)[0]
		idx_3 = np.where((df['usefulCount'] < 0).values)[0]
		idx_4 = np.where((df['review'].str.strip() == "").values)[0]
		
		df.loc[df["usefulCount"] == 0, "usefulCount"] = 0.1
		df.loc[:, "rating"] = df.loc[:,"rating"].values / 10

		idx = np.setdiff1d(np.arange(df.shape[0]) , np.unique(np.concatenate((idx_1, idx_2, idx_3, idx_4))))

		df = df.iloc[idx,:] # cleaned up dataframe

		df.dropna(inplace = True)
	

		if df.empty:
			raise ValueError('there is no relevant data in \"data\"!')


		# preparing returns

		drug_OH = enc_drug.transform(df["drugName"].values.reshape(-1,1)).toarray()
		cond_OH = enc_cond.transform(df["condition"].values.reshape(-1,1)).toarray()

		tokenizer = self._tokenizer
		transformed_seqs = tokenizer.texts_to_sequences(df["review"].values)
		transformed_seqs = pad_sequences(transformed_seqs, maxlen = self.max_len)[:,:self.len_limit]

		rating = df["rating"].values.reshape((-1,1))

		if forModelorTest:
			pw_useC = self._pw_useC 
			useC = pw_useC.transform(df['usefulCount'].values.reshape(-1,1))
			assert transformed_seqs.shape[0] == rating.shape[0] == drug_OH.shape[0] == cond_OH.shape[0]\
				== useC.shape[0] == df.index.shape[0]
			return [transformed_seqs, drug_OH, cond_OH, rating], useC, df.index

		else:
			assert transformed_seqs.shape[0] == rating.shape[0] == drug_OH.shape[0] == cond_OH.shape[0]\
				== df.index.shape[0]
			return [transformed_seqs, drug_OH, cond_OH, rating], df.index

	def lstm_model(self, X, y):
		"""

		Parameters
		----------
		X : preprocessed inputs for lstm model 
		
		y:  preprocesses output for lstm model



		Returns
		-------
		trained model and traning history

		"""
		

		# defining model
		input_layer1 = keras.Input(shape=(None,), dtype="int32")
		embed1 = Embedding(X[0].max() + 1, 64)(input_layer1)
		lstm1 = Bidirectional(LSTM(64, return_sequences=True))(embed1)
		lstm2 = Bidirectional(LSTM(64))(lstm1)
		dense_1 = Dense(1)(lstm2)

		# input layer for drug 
		input_layer2 = keras.Input(shape=(X[1].shape[1],), dtype="int32")
		dense_2 = Dense(12, activation="relu")(input_layer2)
		dense_3 = Dense(1, activation="relu")(dense_2)

		# input layer for condition
		input_layer3 = keras.Input(shape=(X[2].shape[1],), dtype="int32")
		dense_4 = Dense(12, activation="relu")(input_layer3)
		dense_5 = Dense(1, activation="relu")(dense_4)

		# rating as an input
		input_layer4 = keras.Input(shape=(X[3].shape[1],), dtype="int32")  
		dense_6 = Dense(12, activation="relu")(input_layer4)
		dense_7 = Dense(1, activation="relu")(dense_6)

		concat_layer = Concatenate()([dense_1, dense_3, dense_5, dense_7])

		dense_8 = Dense(4, activation = "relu")(concat_layer)
		output = Dense(1)(dense_8)

		model = keras.Model(inputs = [input_layer1, input_layer2, input_layer3, input_layer4], outputs = output)

		model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.005), loss="mse")

		history = model.fit(x = X , y = y, epochs = 4, validation_split = 0.2)

		self._km_fitted = True

		return model, history




	def save_model(self, *fnames, model = None):
		"""

		Parameters
		----------
		*fnames : file names to which the trained model can be saved
		
		model :
			(Default value = None)
			lstm model object to be saved, otherwise only saving preprocessing model


		Returns
		-------
		None

		"""

		if len(fnames) == 0:
			raise TypeError('no file names were provided!')

		if (model is None) and len(fnames) == 1:
			with open(fnames[0], 'wb') as f:
				f.write(dill.dumps(self))

		elif (model is not None) & (len(fnames) == 2):
			if fnames[1].split('.')[1] != 'h5':
				raise ValueError('include \".h5\" for the keras model file name!')
			else:
				with open(fnames[0], 'wb') as f:
					f.write(dill.dumps(self))
				model.save(fnames[1])

		else:
			raise TypeError('no model was loaded! check fir correct arguments')

	@staticmethod
	def load_model(*fnames):
		"""

		Parameters
		----------
		*fnames : names of the files from which pretrained models can be loaded
		

		Returns
		-------
		model objects

		"""

		ret = [] 
		if len(fnames) >= 1:
			with open(fnames[0], 'rb') as f:
				ret.append(dill.loads(f.read()))

			if len(fnames) == 2:
				ret.append(keras.models.load_model(fnames[1]))

		if not ret:
			raise TypeError("no file names are provided")

		return tuple(ret)

	@staticmethod
	def plot_performance(y_obs, y_pred):
		"""
		plots model performance 
		Parameters
		----------
		y_obs : observed output as (-1,1) numpy array
		    param y_pred:
		y_pred : predicted output as (-1,1) numpy array
		    

		Returns
		-------
		None
		"""

		tmp_df = pd.DataFrame({'obs' : y_obs.ravel(), 'model' : y_pred.ravel()})
		ax = sns.displot(data = tmp_df, x = 'obs', y = 'model', kind = 'kde' )
		ax.ax.set_xlim(-4, 4)
		ax.ax.set_ylim(-4, 4)
		plt.plot(np.linspace(-4,4,100),np.linspace(-4,4,100),'r.-')

		plt.show()


def rank_reviews(df):
	"""
	A standalone function that uses RankDrugRev class to rank the reviews on drugs included in 'df'

	Parameters
	----------
	df : a pandas dataframe that needs to have columns : 'drugName', 'condition', 'review' ,'rating'


	Returns
	-------
	a pandas dataframe that is ranked based on an extra 'usefulness' column
	"""
	
	preprocc, lstm_model = RankDrugRev.load_model('preprcc.dill','lstm_model.h5')

	X, idx_known = preprocc.transform_preprocess(data = df, forModelorTest = False)
	y_pred = lstm_model.predict(X)
	df.loc[idx_known, 'usefulness'] = y_pred
	df.sort_values(by=['usefulness'], inplace = True, ascending = False)

	return df