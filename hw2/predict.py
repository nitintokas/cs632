
# coding: utf-8

# In[29]:

""" This code demonstrates reading the test data and writing 
predictions to an output file.
It should be run from the command line, with one argument:
$ python predict.py [test_file]
where test_file is a .npy file with an identical format to those 
produced by extract_cats_dogs.py for training and validation.
(To test this script, you can use one of those).
This script will create an output file in the same directory 
where it's run, called "predictions.txt".
"""

import sys
import numpy as np
import random
import os
import keras
from keras.models import load_model

#constants
CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0
#TEST_FILE = sys.argv[1]

TEST_FILE = "validation1.npy"
# This file will be created if it does not exist
# and overwritten if it does

OUT_FILE = "predictions.txt"
BATCH_SIZE = 20

data = np.load(TEST_FILE).item()

x_train = images = data["images"]

# the testing data also contains a unique id
# for each testing image
if "ids" in data:
    ids = data["ids"]
else:
    #if it's not contained, a sequence is used
    ids = list(range(0,len(images)))

model = load_model("final_model.h5")

predictions = model.predict(x_train, BATCH_SIZE, verbose=1)
print(predictions)


# making a prediction on each image
# and writing output to disk
out = open(OUT_FILE, "w")
for i, image in enumerate(images):
    image_id = ids[i]
    prediction = round(float(predictions[i]))
    line = str(image_id) + " " + str(prediction) + "\n"
    out.write(line)
out.close()


# In[ ]:



