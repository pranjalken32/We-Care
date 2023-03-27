import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans

missing_value = ["N/a", "na", np.nan]
df = pd.read_csv("server/data.csv", na_values=missing_value)
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna()
df[['Experience']]=df[['Experience']]/56
df['Awards']=50*df['Padma_Vibhushan']+40*df['Padma_Bhushan']+30*df['Padma_Shri']+20*df['Dhanvantari_Award']+20*df['BC_Roy_National_Award']+30*df['Other_Awards']
df[['Awards']]=df[['Awards']]/df['Awards'].max()
df.drop('Padma_Vibhushan',
            axis='columns', inplace=True)
df.drop('Padma_Bhushan',
            axis='columns', inplace=True)
df.drop('Padma_Shri',
            axis='columns', inplace=True)
df.drop('Dhanvantari_Award',
            axis='columns', inplace=True)
df.drop('BC_Roy_National_Award',
            axis='columns', inplace=True)
df.drop('Other_Awards',
            axis='columns', inplace=True)
    
df['Experience']=df['Experience']*5
df['Awards']=df['Awards']*5

df['Rating']=df['Experience']*80+df['Awards']*20
df['Rating']=df['Rating']/100

df.drop('Awards',axis='columns', inplace=True)

df['Rating']=df['Rating']+0.40

df['Rating']=df['Rating'].apply(np.ceil)


km = KMeans(n_clusters=4)
    
y_predict=km.fit_predict(df[['Experience','Rating']])

df['cluster'] = y_predict

pickle.dump(km, open('model.pkl','wb'))