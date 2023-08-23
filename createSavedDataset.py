#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Sasha Rieders
#Homework 1
#I used chatGPT to help me write the code. I understand all of the code that is implemented below.
import pandas
import numpy as np
import tensorflow as tf
inDat=pandas.read_pickle("appml-assignment1-dataset-v2.pkl")
dataFrameX = inDat['X']
dataFrameY = inDat['y']
# Compute the fractional change of the next hour's CAD-high versus the previous hour's CAD-close
fractionalChange = (dataFrameY - dataFrameX['CAD-close']) / dataFrameX['CAD-close']
# Define the boundaries for the bins
boundaries = np.linspace(-0.001, 0.001, num=21)
# Use the digitize function to assign each fractional change to a bin
bins = np.digitize(fractionalChange, boundaries)
dataFrameX['target'] = bins

#creating attribute for day of week
dataFrameX['weekday'] = dataFrameX['date'].dt.dayofweek
#print(dataFrameX.head())

#creating attribute for hour of day
dataFrameX['hour'] = dataFrameX['date'].dt.hour

#creating month of year attribute
dataFrameX['month'] = dataFrameX['date'].dt.month


import tensorflow as tf
print(tf.__version__)

from tensorflow.train import Feature, Features, Example, BytesList, FloatList, Int64List
with tf.io.TFRecordWriter('dataset.tfrecord') as f:
  for row in dataFrameX.iterrows():
    myDict={"tickers": Feature(float_list= FloatList(value=row[1][dataFrameX.columns[1:-4]].values)),
            "weekday": Feature(int64_list= Int64List(value=[row[1]['weekday']])),
            "hour": Feature(int64_list= Int64List(value=[row[1]['hour']])),
            "month": Feature(int64_list= Int64List(value=[row[1]['month']])),
            "target": Feature(int64_list= Int64List(value=[row[1]['target']]))}
    myExamp=Example(features=Features(feature=myDict))
    f.write(myExamp.SerializeToString())


correctData = tf.data.TFRecordDataset('correct.tfrecord')
myData = tf.data.TFRecordDataset('dataset.tfrecord')

for raw_record in correctData.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

for key, value in myDict.items():
    if value.HasField('float_list'):
        print(key, "shape:", [len(value.float_list.value)])
    elif value.HasField('int64_list'):
        print(key, "shape:", [len(value.int64_list.value)])
    elif value.HasField('bytes_list'):
        print(key, "shape:", [len(value.bytes_list.value)])

