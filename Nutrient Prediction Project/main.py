#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 10:11:44 2019

@author: Arunkumar
"""

import numpy as np
import pandas as pd
import re

path = 'sr28abxl/ABBREV.xlsx'

dataset = pd.read_excel(path)

columns_to_drop = ['NDB_No','Shrt_Desc']
dataset1 = dataset.drop(columns_to_drop , axis=1)

total_columns = dataset1.columns
num_columns = dataset1._get_numeric_data().columns
cat_columns = list(set(total_columns)-set(num_columns))

labels = ['Calcium_(mg)','Iron_(mg)','Zinc_(mg)','Vit_A_IU','Vit_D_IU','Folate_Tot_(µg)']

def clean_text(data):
    if isinstance(data,str):
        clean = re.sub(r'\([^)]*\)','',data)
        clean = re.sub(r'\D',"",clean)
        return clean 
    else:
        return data

dataset1[cat_columns[0]] = dataset1[cat_columns[0]].apply(clean_text)
dataset1[cat_columns[1]] = dataset1[cat_columns[1]].apply(clean_text)

get_median = dataset1.median()
dataset1 = dataset1.fillna(get_median)

drop_columns = ['Vit_D_µg', 'Vit_A_RAE'] + labels

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#cols = dataset1.columns
#rows = dataset1.index
#dataset1 = sc.fit_transform(dataset1)
#dataset1 = pd.DataFrame(dataset1,index = rows,columns = cols)
y = dataset1[labels].values
x = dataset1.drop(drop_columns,axis=1).values

# BUILDING THE MODEL 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


 