#!/usr/bin/env python
# coding: utf-8

# In[200]:


#all of this is just to load the dataset back in 
import tensorflow as tf
import numpy as np
import pandas as pd
features = {
        'tickers': tf.io.FixedLenFeature(shape=[188],dtype=tf.float32,default_value=np.zeros(188)),
        'weekday': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64,default_value=0),
        'hour': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64,default_value=0),
        'month': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64,default_value=0),
        'target': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64,default_value=0)
    }


# In[201]:


#3.2.1
def parse_examples(serialized_examples):
    examples=tf.io.parse_example(serialized_examples,features)
    #this gets the last feature, which is our target which is what we will return
    #targets=examples['features'][:,-1]
    targets = examples.pop('target')
    #this returns two dictionaries. the first is examples, which holds tickers, weekday, hour, month and then it
    #also returns targets
    return examples, targets
#calls the parse_examples method and loads in the dataset 


# In[202]:


# 3.2.2
dataset=tf.data.TFRecordDataset(['dataset.tfrecord']).batch(256).map(parse_examples).cache()


# In[203]:




#3.2.4
# creating the keras inputs 
tickersInp = tf.keras.layers.Input(188, name='tickers', dtype=tf.float32)
weekdayInp = tf.keras.layers.Input((), name='weekday', dtype=tf.int64)
hourInp = tf.keras.layers.Input((), name='hour', dtype=tf.int64)
monthInp = tf.keras.layers.Input((), name='month', dtype=tf.int64)

#importing imputer layer
from customImputeLayerDefinition import myImputer
# chain imputer and normalizer, and adapt on numerical features

#3.2.5
myImp=myImputer()
myImp.adapt(dataset.map(lambda x,y: x['tickers']))

#3.2.6
normalizer=tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(dataset.map(lambda x,y: myImp(x['tickers'])))


# In[204]:


#3.2.3
#next split the dataset. I will use 80% training, 10% validation, and 10% testing 
dataset=tf.data.TFRecordDataset(['dataset.tfrecord'])
datLen = dataset.reduce(0,lambda x,y:x+1)
nv = int(datLen.numpy()*.1)
nTest = int(datLen.numpy()*.1)
nTrain = datLen.numpy()-nTest-nv

training = dataset.take(nTrain).batch(256).map(parse_examples,num_parallel_calls=8).cache()
testing = dataset.skip(nTrain).take(nTest).batch(256).map(parse_examples,num_parallel_calls=8).cache()
validation = dataset.skip(nTrain+nTest).take(nv).batch(256).map(parse_examples,num_parallel_calls=8).cache()


# In[205]:


#this is the next part of the problem  (3.3)
# aggregate into dictionary input fed model
#3.2.1
catEncoder=tf.keras.layers.IntegerLookup(max_tokens=12,num_oov_indices=0)
catEncoder.adapt(training.map(lambda x,y:x['month']))
inputDict={
        'tickers': tf.keras.Input((188), dtype=tf.float32),
        'weekday': tf.keras.Input((),dtype=tf.int64),
        'hour': tf.keras.Input((),dtype=tf.int64),
        'month': tf.keras.Input((),dtype=tf.int64)}


# In[206]:


#3.3.1
#we will tune output_dim because this is up to us to pick 

weekDayEmb = tf.keras.layers.Embedding(input_dim=7,output_dim = 8, input_length=1)(weekdayInp)
hourEmb = tf.keras.layers.Embedding(input_dim=24,output_dim = 8, input_length=1)(hourInp)
monthEmb = tf.keras.layers.Embedding(input_dim=12,output_dim = 8, input_length=1)(monthInp)

catInts=catEncoder(inputDict['month'])
embDim=32 
monthEmb=tf.keras.layers.Embedding(12,embDim)(catInts)
hourEmb=tf.keras.layers.Embedding(24,embDim)(inputDict['hour'])
weekDayEmb=tf.keras.layers.Embedding(7,embDim)(inputDict['weekday'])


# In[207]:


# now that embedded, same tf.float32 type as normalized, so can concat
preproced = tf.concat([normalizer(myImp(inputDict['tickers'])), weekDayEmb, hourEmb, monthEmb],axis=-1)


# In[208]:


restMod=tf.keras.Sequential([tf.keras.layers.Dense(22,activation='softmax')])


# In[209]:


decs=restMod(preproced)


# In[210]:


whole_model=tf.keras.Model(inputs=inputDict,outputs=decs)


# In[211]:


whole_model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[212]:


whole_model.fit(training,epochs=50,validation_data=validation,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)])
whole_model.evaluate(testing)


# In[213]:


tf.saved_model.save(whole_model, 'mySavedModel')

