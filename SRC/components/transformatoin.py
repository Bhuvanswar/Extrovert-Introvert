import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from ..exception import CustomException
from ..logger import logging
import os
from imblearn.over_sampling  import SMOTE
from ..utility import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = [
                  "Time_spent_Alone",
                  "Social_event_attendance",
                  "Going_outside",
                  "Friends_circle_size",
                  "Post_frequency"
                ]
            categorical_columns = ["Stage_fear","Drained_after_socializing"]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore",sparse_output=False)),  
                ("scaler", StandardScaler(with_mean=False))
            ]
        )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Personality"
            categorical_columns = ["Stage_fear","Drained_after_socializing"]

            target_encoder = LabelEncoder()
                        
            target_feature_train_df = target_encoder.fit_transform(train_df[target_column_name])
            target_feature_test_df = target_encoder.transform(test_df[target_column_name])

            save_object("artifacts/label_encoder.pkl", target_encoder)
            #drop coloumn 
            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            input_feature_test_df=test_df.drop(target_column_name,axis=1)

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            logging.info(f"Train Data Columns: {input_feature_train_df.columns}")
            logging.info(f"Test Data Columns: {input_feature_test_df.columns}")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("inputfeatures_array")
            smote = SMOTE(random_state=42)
            input_feature_train_arr, target_feature_train_df = smote.fit_resample(input_feature_train_arr, target_feature_train_df)
            logging.info("Applied SMOTE for data balancing.")

            #train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            #test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                #train_arr,test_arr,
                input_feature_train_arr,
                input_feature_test_arr,
                target_feature_train_df,
                target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
