# -*- coding: utf-8 -*-
"""Heart Disease prediction main project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uukmrpI7UDJeQ43s1ZtVhAp3RDxzGeL-

# Import

Import Libraries
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
data = pd.read_csv('/content/heart_2020_cleaned.csv')
df = pd.DataFrame(data)
df.isnull().sum()
df['HeartDisease']=df['HeartDisease'].replace({'No': 0,'Yes': 1})
df['Smoking']=df['Smoking'].replace({'No': 0,'Yes': 1})
df['AlcoholDrinking']=df['AlcoholDrinking'].replace({'No': 0,'Yes': 1})
df['Stroke']=df['Stroke'].replace({'No': 0,'Yes': 1})
df['DiffWalking']=df['DiffWalking'].replace({'No': 0,'Yes': 1})
df['PhysicalActivity']=df['PhysicalActivity'].replace({'No': 0,'Yes': 1})
df['Asthma']=df['Asthma'].replace({'No': 0,'Yes': 1})
df['KidneyDisease']=df['KidneyDisease'].replace({'No': 0,'Yes': 1})
df['SkinCancer']=df['SkinCancer'].replace({'No': 0,'Yes': 1})
df['Sex']=df['Sex'].replace({'Female': 0,'Male': 1})
label_encoder = LabelEncoder()
df["Race"] = label_encoder.fit_transform(df["Race"])
df["GenHealth"] = label_encoder.fit_transform(df["GenHealth"])
df["AgeCategory"] = label_encoder.fit_transform(df["AgeCategory"])
df["Diabetic"] = label_encoder.fit_transform(df["Diabetic"])
index_to_delete = 49997
df2 = df.drop(index_to_delete)

from sklearn.preprocessing import StandardScaler  # -3 to 3
scaler= StandardScaler()
X_standardlized=scaler.fit_transform(df2)
X_standardlized

x = df2.drop(columns='HeartDisease',axis=1)
y = df2['HeartDisease']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

print(x.shape,X_train,X_test.shape)


model = LogisticRegression()
model.fit(X_train, y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,y_test)
print('accuracy of testing data:',test_data_accuracy)