import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor
import yaml
 
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('Model Building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'Data Ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data Loaded from %s with shape %s',file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('It failed to parse the training and test data for model building')
        raise
    except FileNotFoundError as e:
        logger.error('Files not found to train and test data')
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s',e)
        raise
    
def train_model(X_train : np.ndarray ,y_train :np.ndarray ,params : dict) -> RandomForestRegressor:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of parameter in X_train and Y_train must be same.")
        
        logger.debug('Initalizing Randomforest model with parameters : %s',params)
        rf_reg_model = RandomForestRegressor(n_estimators=params['n_estimators'],random_state=params['random_state'])
        
        logger.debug('Model Training Started with %d samples',X_train.shape[0])
        rf_reg_model.fit(X_train,y_train)
        logger.debug('Model Training Completed')
        
        return rf_reg_model
    except ValueError as e:
        logger.error('ValueError during model training: %s',e)
        raise
    except Exception as e:
        logger.error('Erro During model training: %s',e)
        raise
    

def save_model(model,file_path: str)-> str:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved to %s',file_path)
    except FileNotFoundError as e:
        logger.debug('File Path not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error while saving the model :%s',e)
        raise
    

def main():
    try:
        params = load_params('params.yaml')['model_training']
        train_data = load_data('data/raw/train.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        rf_reg_model = train_model(X_train,y_train,params)
        
        model_save_path = 'models/model.pkl'
        save_model(rf_reg_model,model_save_path)
    except Exception as e:
        logger.error('Failed to complete the model building process: %s',e)
        print(f"Error:{e}")

if __name__ == '__main__':
    main()
        
        