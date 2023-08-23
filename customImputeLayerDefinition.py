#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
class myImputer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    #When you use this syntax, you can pass any number of keyword arguments to the function, and they will be collected ...
    # into a dictionary with the keyword as the key and the value as the corresponding value.
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
    def build(self,batch_input_shape):
        self.imps=self.add_weight(name='imps',shape=(batch_input_shape[-1]),
                                initializer="zeros",trainable=False)
        super().build(batch_input_shape)
    def call(self, X):
        return tf.where(tf.math.is_nan(X),self.imps,X)
    def adapt(self, dataset):
        self.build(dataset.element_spec.shape)
        sumOfNonNaNs=dataset.map(lambda z: tf.where(tf.math.is_nan(z),
                                tf.zeros_like(z),z)).reduce(
                                tf.zeros_like(self.imps),
                                lambda x,y: x+tf.reduce_sum(y,axis=0))
        numberOfNonNaNs=dataset.map(lambda z: tf.where(tf.math.is_nan(z),
                                tf.zeros_like(z),tf.ones_like(z))).reduce(
                                tf.zeros_like(self.imps),
                                lambda x,y: x+tf.reduce_sum(y,axis=0))
        self.imps.assign(tf.math.divide(sumOfNonNaNs,numberOfNonNaNs))
    def computer_output_shape(self,batch_input_shape):
        return batch_input_shape

