# from flask import Flask,render_template,request

# app = Flask(__name__)

# @app.route('/')
# def hell():
#     return render_template('index.html')

# @app.route('/submit', methods=['POST'])
# def submit_form():
#     # Access form data using request.form
#     department = request.form.get('department')
#     region = request.form.get('region')
#     education = request.form.get('education')
#     recruitment_channel = request.form.get('recruitment_channel')
#     gender = request.form.get('gender')
#     no_of_trainings = request.form.get('no_of_trainings')
#     length_of_service = request.form.get('length_of_service')
#     previous_year_rating = request.form.get('previous_year_rating')
#     avg_training_score = request.form.get('avg_training_score')
#     awards = request.form.get('awards')
#     KPIs_met = request.form.get('KPIs_met')
#     region = request.form.get('region')

#     # Do something with the values (e.g., print them)
#     print(f"Name: {department}, Email: {region},  education: {education}, recruitment_channel: {recruitment_channel}, gender: {gender}, no_of_trainings: {no_of_trainings}, length_of_service: {length_of_service}, previous_year_rating: {previous_year_rating}, avg_training_score: {avg_training_score}, awards:{awards}, KPIs_met : {KPIs_met}")
    
#     return f"Form submitted successfully! Name: {department}, Email: {region}education: {education}, recruitment_channel: {recruitment_channel}, gender: {gender}, no_of_trainings: {no_of_trainings}, length_of_service: {length_of_service}, previous_year_rating: {previous_year_rating}, avg_training_score: {avg_training_score}, awards:{awards}, KPIs_met : {KPIs_met}"

# if __name__=='__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

# Your existing route for rendering the form
@app.route('/')
def hell():
    return render_template('index.html')

# Load dataset

data = pd.read_csv('heart_2020_cleaned.csv')
data['HeartDisease']=data['HeartDisease'].replace({'No': 0,'Yes': 1})

# Encoding
def display(df):
    label_encoder = LabelEncoder()
    cols = ['Smoking','AlcoholDrinking','Stroke','DiffWalking','PhysicalActivity','Asthma','KidneyDisease','SkinCancer','Sex',"Race","GenHealth","AgeCategory","Diabetic"]
    for col in cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    # Normalization
    from sklearn.preprocessing import StandardScaler  # -3 to 3
    scaler= StandardScaler()
    X_standardlized=scaler.fit_transform(df)
    return X_standardlized

x = data.drop(columns='HeartDisease',axis=1)
y = data['HeartDisease']

x = display(x)

model = LogisticRegression()
model.fit(x, y)
# Your existing route for handling form submission
@app.route('/submit', methods=['POST'])
def submit_form():
    # Access form data using request.form
    bmi = request.form.get('bmi')
    smoking = request.form.get('smoking')
    drinking = request.form.get('drinking')
    disease = request.form.get('disease')
    physical = request.form.get('physical')
    mental = request.form.get('mental')
    diffwalking = request.form.get('diffwalking')
    gender = request.form.get('gender')
    ageCategory = request.form.get('ageCategory')
    race = request.form.get('race')
    diabetic = request.form.get('diabetic')
    physicalActivity = request.form.get('physicalActivity')
    genHealth = request.form.get('genHealth')
    sleepTime = request.form.get('sleepTime')
    asthma = request.form.get('asthma')
    kidneyDisease = request.form.get('kidneyDisease')
    skinCancer = request.form.get('skinCancer')

    # Preprocess the form data
    user_input_df = pd.DataFrame({
        'BMI': [bmi],
        'Smoking': [smoking],
        'AlcoholDrinking': [drinking],
        'Stroke': [disease],
        'PhysicalHealth': [physical],
        'MentalHealth': [mental],
        'DiffWalking' : [diffwalking],
        'Sex': [gender],
        'AgeCategory': [ageCategory],
        'Race': [race],
        'Diabetic': [diabetic],
        'PhysicalActivity': [physicalActivity],
        'GenHealth': [genHealth],
        'SleepTime': [sleepTime],
        'Asthma': [asthma],
        'KidneyDisease': [kidneyDisease],
        'SkinCancer': [skinCancer]
    })

    data = user_input_df.copy()
    data = display(data)
    pred = model.predict(data)
    if pred == 0:
        result = f"not diagnosed"
    else:
        result = f"diagnosed"
    return render_template('heart_disease.html',result_text=result)
if __name__=='__main__':
    app.run(debug=True)
