import sys
import os
import pandas as pd
from ..exception import CustomException
from ..utility import load_object
from ..logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=r"C:\Users\bhuva\OneDrive\Desktop\learn\python\New folder (2)\personality\artifacts\model.pkl"
            preprocessor_path=r'C:\Users\bhuva\OneDrive\Desktop\learn\python\New folder (2)\personality\artifacts\proprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            label_encoder = load_object(r"C:\Users\bhuva\OneDrive\Desktop\learn\python\New folder (2)\personality\artifacts\label_encoder.pkl")
            decoded_preds = label_encoder.inverse_transform(preds.astype(int))
            print("Classes:", label_encoder.classes_)
            print("Raw predictions (encoded):", preds)
            print("Decoded predictions:", decoded_preds)
            return decoded_preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 Time_spent_Alone,
                 Stage_fear,
                 Social_event_attendance,
                 Going_outside,
                 Drained_after_socializing,
                 Friends_circle_size,
                 Post_frequency):

        self.Time_spent_Alone = Time_spent_Alone
        self.Stage_fear = Stage_fear
        self.Social_event_attendance = Social_event_attendance
        self.Going_outside = Going_outside
        self.Drained_after_socializing = Drained_after_socializing
        self.Friends_circle_size = Friends_circle_size
        self.Post_frequency = Post_frequency

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Time_spent_Alone": [self.Time_spent_Alone],
                "Stage_fear": [self.Stage_fear],
                "Social_event_attendance": [self.Social_event_attendance],
                "Going_outside": [self.Going_outside],
                "Drained_after_socializing": [self.Drained_after_socializing],
                "Friends_circle_size": [self.Friends_circle_size],
                "Post_frequency": [self.Post_frequency]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
