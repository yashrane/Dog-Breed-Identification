# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:42:22 2018

@author: yashr
"""

import pandas as pd
from sklearn.model_selection import train_test_split

labels = pd.read_csv('labels/labels.csv')

train, val = train_test_split(labels, train_size=0.8, random_state=0)

train.to_csv('labels/train.csv')
val.to_csv('labels/val.csv')