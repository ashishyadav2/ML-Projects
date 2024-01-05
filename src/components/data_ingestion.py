import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from data_transformation import DataTransformation
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
# dataclass is a decorator, used when class has only variables
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact','train.csv')
    test_data_path: str = os.path.join('artifact','test.csv')
    raw_data_path: str = os.path.join('artifact','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try: 
            df = pd.read_csv('./notebook/data/stud.csv')
            logging.info('read csv file dataset as dataframe(df)')
            
            # if already exist don't do anything else create one
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train test split initiated')
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,            
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, x = data_transformation.initiate_data_transformation(
        obj.ingestion_config.train_data_path,
        obj.ingestion_config.test_data_path
    )
    modeltrainer = ModelTrainer()
    r2_scoree = modeltrainer.initiate_model_trainer(train_arr,test_arr)
    print(r2_scoree)
    