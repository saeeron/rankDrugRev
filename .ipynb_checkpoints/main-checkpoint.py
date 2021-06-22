import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, Concatenate, Flatten 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import PowerTransformer, OneHotEncoder
import pandas as pd

import string

import requests
import zipfile
import io   


r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip", verify = False)
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(io.BytesIO(z.read("drugsComTrain_raw.tsv")),sep = '\t', parse_dates = [5])
df.rename(columns={"Unnamed: 0": "id"}, inplace = True)

df = df[~df["condition"].str.contains("</span>").astype(bool)]

# finding and removing low-count conditions
low_cond_list = df["condition"].value_counts()[df["condition"].value_counts() < 30].index.values
low_drug_list = df["drugName"].value_counts()[df["drugName"].value_counts() < 30].index.values

idx_ = (~df["condition"].isin(low_cond_list)) & (~df["drugName"].isin(low_drug_list))

df = df[idx_].copy()
df.dropna(inplace = True)

enc_cond = OneHotEncoder(handle_unknown='ignore')
enc_cond.fit(df["condition"].values.reshape(-1,1))
cond_OH = enc_cond.transform(df["condition"].values.reshape(-1,1)).toarray()

enc_drug = OneHotEncoder(handle_unknown='ignore')
enc_drug.fit(df["drugName"].values.reshape(-1,1))
drug_OH = enc_drug.transform(df["drugName"].values.reshape(-1,1)).toarray()


df.loc[df["usefulCount"] == 0, "usefulCount"] = 0.1  # box_cox transformation requires strictly positive data 
tmp = df["usefulCount"].values.reshape((-1,1))
pw_1 = PowerTransformer(method='box-cox')
usefulC_ = pw_1.fit_transform(tmp)


rating_ = df["rating"].values.reshape((-1,1)) / 10


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 5000, oov_token="<UNK>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
tokenizer.fit_on_texts(df["review"].values)
tokenizer.index_word[0] = '<PAD>'
train_seqs = tokenizer.texts_to_sequences(df["review"].values)
train_seqs = pad_sequences(train_seqs)

# defining model
input_layer1 = keras.Input(shape=(None,), dtype="int32")
embed1 = Embedding(train_seqs.max() + 1, 64)(input_layer1)
lstm1 = Bidirectional(LSTM(64, return_sequences=True))(embed1)
lstm2 = Bidirectional(LSTM(64))(lstm1)
dense_1 = Dense(1)(lstm2)

# input layer for drug 
input_layer2 = keras.Input(shape=(drug_OH.shape[1],), dtype="int32")
dense_2 = Dense(12, activation="relu")(input_layer2)
dense_3 = Dense(1, activation="relu")(dense_2)

# input layer for condition
input_layer3 = keras.Input(shape=(cond_OH.shape[1],), dtype="int32")
dense_4 = Dense(12, activation="relu")(input_layer3)
dense_5 = Dense(1, activation="relu")(dense_4)

# rating as an input
input_layer4 = keras.Input(shape=(rating_.shape[1],), dtype="int32")  
dense_6 = Dense(12, activation="relu")(input_layer4)
dense_7 = Dense(1, activation="relu")(dense_6)

concat_layer = Concatenate()([dense_1, dense_3, dense_5, dense_7])

dense_8 = Dense(4, activation = "relu")(concat_layer)
output = Dense(1)(dense_8)

model = keras.Model(inputs = [input_layer1, input_layer2, input_layer3, input_layer4], outputs = output)

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss="mse")

history = model.fit(x = [train_seqs, drug_OH, cond_OH, rating_] , y = usefulC_, epochs = 2, validation_split = 0.2)

y_ = model.predict(x = [train_seqs, drug_OH, cond_OH, rating_])
y_inv = pw_1.inverse_transform(y_)

tmp_df = pd.DataFrame({'obs' : usefulC_.ravel(), 'model' : y_.ravel()})

ax = sns.displot(data = tmp_df, x = 'obs', y = 'model', kind = 'kde' )

ax.ax.set_xlim(-5, 5)
ax.ax.set_ylim(-5, 5)

plt.show()

h = plt.hist2d(usefulC_.ravel(), y_.ravel(), bins = 100, range = [[-3, 3],[-3, 3]] )[0]


rmse = np.sqrt(((y_inv - tmp)**2).mean())

df_test = pd.read_csv(io.BytesIO(z.read("drugsComTest_raw.tsv")),sep = '\t', parse_dates = [5])
df_test.rename(columns={"Unnamed: 0": "id"}, inplace = True)

# removing low-count conditions learned from train
idx_ = (~df_test["condition"].isin(low_cond_list)) & (~df_test["drugName"].isin(low_drug_list))

df_test = df_test[idx_]
df_test.dropna(inplace = True)
df_test = df_test[~df_test["condition"].str.contains("</span>")]

df["condition"] = df["condition"].astype("category")
df["drugName"] = df["drugName"].astype("category")



model.predict()

# from main import *
