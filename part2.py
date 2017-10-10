
# coding: utf-8

# # import glob for iterating files in folder and use dataframe for structuring

# In[1]:

import glob
import pandas as pd

body=[]
list_of_files = glob.glob('spam_data/0*.txt')

for fileName in list_of_files:
    
    file = open(fileName, encoding = "ISO-8859-1")
    body.append(file.read())
    
    file.close()

body_file = pd.DataFrame(data = body,columns = ['post'])
body_file



# # extract label file

# In[3]:

#list_of_labels = glob.glob('/Users/ajinkya.parkar@ibm.com/Documents/deep/spam_data/labels.txt')
label_file = pd.read_csv('spam_data/labels.txt',header = None,delimiter=" ",names=["label",'Name'])
label_file 


# In[4]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[5]:

import itertools
import os

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, Dropout
from tensorflow.contrib.keras.python.keras.preprocessing import text, sequence
from tensorflow.contrib.keras.python.keras import utils

# This code was tested with TensorFlow v1.3
print("You have TensorFlow version", tf.__version__)


# In[6]:

body_file["post"]


# # Set index for train and test

# In[7]:

# Split data into train and test
train_size = int(len(body_file) * .5)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(body_file) - train_size))


# # Prepare train and test data in 50:50 ratio

# In[8]:


train_posts = body_file['post'][:train_size]
train_tags = label_file['label'][:train_size]

test_posts = body_file['post'][train_size:]
test_tags = label_file['label'][train_size:]


# In[9]:

train_posts


# File 1) a bunch of text
# Label 1) spam
# 
# File 2) a bunch more text
# Label 2) not spam
# 
# 
# more text
# 
# x_train 1) 0, 1
# x_train 2) 1, 1
# 
# testing example 1) more text
# x_test 1) 1, 1
# 
# 
# 
# 

# In[10]:

len(train_posts), len(train_tags)


# In[11]:

len(test_posts), len(test_tags)


# # Set bag of words array length

# In[12]:

max_words = 10
tokenize = text.Tokenizer(num_words=max_words, char_level=False)


# # train posts

# In[13]:

tokenize.fit_on_texts(train_posts) # only fit on train


# In[14]:

text = "the"
# tokenize the text, and retrieve the index of the first 10 tokens
tokenize.texts_to_matrix([text])


# In[15]:

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)


# In[16]:

print(tokenize.word_index)


# In[17]:

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


# In[18]:

np.max(y_train) + 1


# In[19]:

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# In[20]:

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# In[21]:

batch_size = 32
epochs = 5


# In[22]:

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[23]:

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


# In[24]:

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[25]:

y_train

