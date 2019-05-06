import os
import sys
from keras.models import load_model
model2 = load_model('my_model_ran1511.h5')

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical

test_df = pd.read_pickle('NvAndMelTestSetG2.zip')
test_df['idx'] = np.where(test_df['id'] == 'mel', 0 , 1)
features=test_df.drop(columns=['idx'], axis = 1)
target=test_df['idx']
features = np.asarray(features['image'].tolist())
target_cat = to_categorical(target, num_classes = 2)

import lime
from lime import lime_image

import matplotlib.pyplot as plt

image_data = test_df.iloc[0:400, 0]
images = []
for item in image_data:
  images.append(list(item))
  
images = np.asarray(images)

preds = model2.predict(images)
actual = test_df['idx'][0:400].values

index = 220
if preds[index][0] > preds[index][1]:
    diag_type = 0
    print("Predicted melanoma")
else:
    diag_type = 1
    print("Predicted nevi")
    
if actual[index] == 0:
    print("Actually melanoma")
else:
    print("Actually nevi")


explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[index], model2.predict, top_labels=10, hide_color=0, num_samples=1000)

from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(diag_type, positive_only=False, num_features=10, hide_rest=False)
imgplot = plt.imshow(mark_boundaries(temp, mask))
imgplot.axes.get_xaxis().set_visible(False)
imgplot.axes.get_yaxis().set_visible(False)
print(preds[index])

plt.show()