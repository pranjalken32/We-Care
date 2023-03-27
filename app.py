import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
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

    Doctor_Name = request.form.get("username")

    specialisaton = request.form.get("speciality")

    city = request.form.get("city")

    doctor_Experience = int(request.form.get("experience"))

    doctor_Awards_Points = int(request.values.get("rating"))
    Experience_Normalised = float(doctor_Experience/56)*5
    Awards_Point_Normalised = doctor_Awards_Points
    predicted_user = km.predict(
        [[Experience_Normalised, Awards_Point_Normalised]])
    final = []
    if(predicted_user < 4):  # for outliers
        for i in range((df.shape[0])):
            if(str(df.iloc[i, 2]).count(city) > 0 and str(df.iloc[i, 1]).count(specialisaton) > 0 and df.iloc[i, 6]<= predicted_user and Experience_Normalised < float(df.iloc[i, 4]) and Awards_Point_Normalised < float(df.iloc[i, 5])):
                final.append(df.iloc[i])

    hi=[]
    if(len(final)):
        for i in final:
            x = {"Name": i["Name"], "Specialisation": i["Specialisation"],
                 "City": i["City"], "Rating": i["Rating"]}
            hi.append(x)
    else:
        hi.append("Your Doctor is the best in your Area")

    return render_template('index.html', prediction_text=hi)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
