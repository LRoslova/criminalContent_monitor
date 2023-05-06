# !pip install num2words
# !pip install transformers
# !pip install nltk==3.2.4
import pandas as pd, numpy as np
import re
import string
import json
from num2words import num2words
import random
import math
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, \
    BatchNormalization, LayerNormalization
from tensorflow_addons.metrics import F1Score
from tensorflow.compat.v1.keras import backend as K

from transformers import BertConfig, TFAutoModel, AutoTokenizer, BertTokenizer
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
import nltk
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score
nltk.download('stopwords')
nltk.download('punkt')
import nltk.corpus
from nltk.corpus import stopwords
tqdm.pandas()
from sklearn.metrics import f1_score

import os
for dirname, _, filenames in os.walk('/home/larzi/DIPLOM/some'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

## Read the data into a dataframe

data=[]
with open('/home/larzi/DIPLOM/some/News_Category_Dataset_v3.json', 'r') as f:
    for line in f:
        content = json.loads(line)
        data.append(content)
df = pd.DataFrame(data=data)
print(df.head(3))

# make a copy of the dataframe

print(df.shape, df.columns)
df_copy = df.copy()
df.isna().sum()

print(df[df['short_description'].apply(lambda x: len(x)==0)].shape)

df = df[~df['short_description'].apply(lambda x: len(x)==0)]
print(df.shape)

df[df['headline'].apply(lambda x: len(x)==0)]

print(df['category'].nunique())
print(df['category'].unique())

print(df['category'].value_counts()[0:22])

## Trying out for top 20 news categories
cat_list=df['category'].value_counts()[0:22].index.tolist()
df = df[df['category'].isin(cat_list)].reset_index(drop=True)
df.shape

df = df[['headline', 'short_description', 'category']]
print(df.head(3))

df.loc[df['category']=='PARENTING', 'category'] = 'PARENTS'
df.loc[df['category']=='THE WORLDPOST', 'category'] = 'WORLD NEWS'
df.loc[df['category']=='BLACK VOICES', 'category'] = 'VOICES'
df.loc[df['category']=='QUEER VOICES', 'category'] = 'VOICES'
df.loc[df['category']=='WEDDINGS', 'category'] = 'WEDDINGS & DIVORCE'
df.loc[df['category']=='DIVORCE', 'category'] = 'WEDDINGS & DIVORCE'
df.loc[df['category']=='HEALTHY LIVING', 'category'] = 'WELLNESS'
print(df['category'].value_counts())

# df['input_data']= df.apply(lambda x: str(x['headline']) + str(x['short_description']), axis=1)
# def clean_text(txt):
#     if txt is np.NaN or txt == '' or len(txt) == '0' or txt is None:
#         return None
    
#     txt= re.sub('[^a-zA-Z0-9\.]', ' ', str(txt).lower())
#     txt = re.sub('\s+', ' ', txt)
#     txt = txt.replace('.','')
#     txt = re.sub('\n', ' ', txt)
#     txt = [nltk.word_tokenize(wrd) for wrd in txt.split() if wrd not in stopwords.words('english')]
#     txt = [item[0] for item in txt]
#     txt = ' '.join(txt)
#     ### convert numbers to words using num2words
#     nums = re.findall('[0-9]+\.?[0-9]*', txt)
#     try:
#         for num in nums:
#             txt = txt.replace(num, num2words(num))
#     except Exception as e:
#         print('exception', e)
#         print(f'for {num} in {txt}')
#     txt = re.sub('re.escape(string.punctuation)', ' ', txt)
#     txt = re.sub('\s+', ' ', txt)
#     txt = re.sub('-', ' ', txt)
#     #print(nums)
#     txt =txt.strip()
#     return txt

# df['input_data_cleaned'] = df['input_data'].progress_apply(lambda x :clean_text(x))

# df.to_csv('my_dataset.csv', index=False)
df = pd.read_csv('my_dataset.csv')
df.head(3)

### Encode the target labels with LabelBinarizer()
enc = LabelBinarizer()
df['category'] =df['category'].apply(lambda x: x.lower())
values = list(pd.unique(df['category']))

enc.fit(values)
print(enc.classes_)
df = df.reset_index()
df.head(3)

index_list = df['index'].values.tolist()
random.Random(42).shuffle(index_list)

train_index = math.floor(len(index_list)*0.80)
train_index_list = set(index_list[:train_index])
train_data = df[df['index'].isin(train_index_list)].reset_index(drop=True)

val_index = math.floor(len(index_list)*0.15)
val_index_list = set(index_list[train_index : train_index + val_index])
val_data = df[df['index'].isin(val_index_list)].reset_index(drop=True)

test_index_list = set(index_list[train_index + val_index :])
test_data = df[df['index'].isin(test_index_list)].reset_index(drop=True)

print('train data :', train_data.shape[0])
print('val data :', val_data.shape[0])
print('test data :', test_data.shape[0])

checkpoint = 'google/bert_uncased_L-4_H-512_A-8'

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint)

model_config = BertConfig.from_pretrained(checkpoint)
model_config.output_hidden_states = True
bert_encoder = TFAutoModel.from_pretrained(checkpoint, from_pt=True, config=model_config)

### model architecture
input_ids = tf.keras.layers.Input(shape=(200,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(200,), dtype=tf.int32, name="attention_mask")

model_inputs = [input_ids, attention_mask]
x = bert_encoder(model_inputs)[0]
x = Bidirectional(LSTM(60, activation="tanh",
                               recurrent_activation="sigmoid",
                               recurrent_dropout=0,
                               unroll=False,
                               use_bias=True,
                               return_sequences=True))(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
#x = BatchNormalization()(x)  ## to reduce overfitting
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = LayerNormalization(name='embedding_layer')(x)
x = Dense(len(enc.classes_), activation='softmax', name="output")(x)

mymodel = tf.keras.Model(inputs=[model_inputs], outputs = [x])

mymodel.summary()

## Involving early stopping option on minimum of val_loss with patience 2
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
mymetrics = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    F1Score(num_classes=len(enc.classes_), average='macro', name='f1_score')
]

## loss function is categorical Crossentropy suitable for multiclass classification
mymodel.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics = mymetrics, optimizer=Adam(learning_rate=0.00001))

## custom build data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, tokenizer, batch_size=32):
       
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.tokenizer = tokenizer
        self.on_epoch_end()

    def __len__(self):
        #print('in len')
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        #print('in getitem')
        start_idx = index * self.batch_size
        end_idx = min(((index + 1) * self.batch_size), len(self.df))

        selected_ind = self.indices[start_idx:end_idx]
        batch_data = self.df.loc[selected_ind].reset_index(drop=True)

        X, y = self.get_data(batch_data)
        return X, y

    def get_data(self, batch_data):

        batch_data = batch_data.fillna('')
        X = self.tokenizer(batch_data['input_data_cleaned'].tolist(), padding='max_length',
                           truncation=True, max_length=200, return_tensors='tf')
        #print(X)
        X = [X['input_ids'], X['attention_mask']]
        y = enc.transform(batch_data['category']).astype('float32')
        y = np.asarray(y).astype('float32')            
        return X, y
train_df = DataGenerator(train_data, tokenizer, 64)
val_df = DataGenerator(val_data, tokenizer, 64)

## save the model weights
filename = "model_weights.h5"
path = os.path.join(os.getcwd(), filename)
mymodel.save_weights(path)

test_data.head(3)

test_encode_X = tokenizer(test_data['input_data_cleaned'].tolist(), padding=True, return_tensors='tf')
test_encode_X = [test_encode_X['input_ids'], test_encode_X['attention_mask']]

y_true = test_data['category'].values
y_pred = mymodel.predict(test_encode_X)
y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred = [np.argmax(i) for i in y_pred]
y_pred = [enc.classes_[i] for i in y_pred]

mcm = multilabel_confusion_matrix(np.array(y_true), np.array(y_pred))
mcm = mcm.flatten().reshape(len(enc.classes_), 4)
mcm = pd.DataFrame(data=mcm, columns=['TN', 'FP', 'FN', 'TP'], index=enc.classes_)
print(mcm)
mcm.to_csv('multilabel_confusion_matrix.csv', index=False)

cl_rep = classification_report(np.array(y_true), np.array(y_pred), target_names=enc.classes_, output_dict=True)
cl_rep = pd.DataFrame.from_dict(cl_rep).transpose()
print(cl_rep)
cl_rep.to_csv('classification_report.csv', index=False)

test_data['category'].value_counts()

## predicting on some sample data from text_data
idx = [0, 356, 158, 1548, 739, 1356, 3156, 5230, 4872, 6234, 2753]
for i in test_data:
    count = 0
    if count < 3:
        if test_data.iloc[i]['category'] == 'crime':
            idx.append(i)
            count = count+1
    else: 
        break
    
y_true_list = []
y_pred_list = []
for i in idx:
    tokenized_text = tokenizer(test_data.iloc[i]['input_data_cleaned'], padding='max_length',
                               truncation=True, max_length=200, return_tensors='tf')
    tokenized_text = [tokenized_text['input_ids'], tokenized_text['attention_mask']]

    y_true_list.append(test_data.iloc[i]['category'])
    y_pred = mymodel.predict(tokenized_text)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_pred = [np.argmax(i) for i in y_pred]
    y_pred = [enc.classes_[i] for i in y_pred]
    y_pred_list.append(y_pred[0])
pd.DataFrame({'True Labels' : y_true_list, 'Predicted Labels': y_pred_list})
