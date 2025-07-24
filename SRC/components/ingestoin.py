import os 
import sys
import numpy as np
from ..logger import logging
from ..exception import CustomException
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from ..components.transformatoin import DataTransformation
from ..components.transformatoin import DataTransformationConfig
from ..components.model_train import ModelTrainerConfig
from ..components.model_train import ModelTrainer

warnings.filterwarnings('ignore', category=UserWarning)  # Ignores UserWarnings
warnings.filterwarnings('ignore', category=FutureWarning) 

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r"SRC\personality_datasert.csv")
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    input_feature_train_arr,input_feature_test_arr,target_feature_train_df,target_feature_test_df,_=data_transformation.initiate_data_transformation(train_data,test_data)
    print("Encoded class distribution:\n", pd.Series(target_feature_train_df).value_counts())

    train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
    test_arr = np.c_[input_feature_test_arr, target_feature_test_df]
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
