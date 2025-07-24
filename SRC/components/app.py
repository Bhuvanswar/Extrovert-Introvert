from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from ..pipeline.predict_pipeline import CustomData, PredictPipeline

import secrets

application=Flask(__name__)
application.secret_key = secrets.token_hex(16)

app=application

from flask import flash

@app.route('/')
def index():
    print("CWD:", os.getcwd())
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                Time_spent_Alone=request.form.get('Time_spent_Alone'),
                Stage_fear=request.form.get('Stage_fear'),
                Social_event_attendance=request.form.get('Social_event_attendance'),
                Going_outside=request.form.get('Going_outside'),
                Drained_after_socializing=request.form.get('Drained_after_socializing'),
                Friends_circle_size=float(request.form.get('Friends_circle_size')),
                Post_frequency=request.form.get('Post_frequency')
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")
            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")
            return render_template('home.html', results=results[0])

        except ValueError as e:
            flash('Please ensure all form fields are filled correctly.')
            return render_template('home.html')

    

if __name__=="__main__":
    #app.run(debug=True) 
    app.run(host="0.0.0.0",debug=True)        


