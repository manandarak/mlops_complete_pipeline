import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import yaml
import logging

#ensure the "logs" directory exits
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#logging confrigation
logger = logging.getLogger('Data Ingestion')
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


def load_data(data_url : str ) -> pd.DataFrame :
    try :
        df = pd.read_csv(data_url)
        logger.debug('Data Loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('It failed to prase the csv file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexcepted error during preprocessing: %s',e)
        raise
    
def preprocess_data(df : pd.DataFrame) -> pd.DataFrame:
    try :
        df.drop(columns=['Q-o-Q','Growth Type'], inplace= True)
        df['Month'] = df['Quarter'].str.split(' ', expand=True)[0] 
        df['Year'] = df['Quarter'].str.split(' ', expand=True)[1]
        df.drop(columns=['Quarter'],inplace=True) 
        df['Average Price'] = df['Average Price'].replace('-', np.nan)
        df['Average Price'] = df['Average Price'].str.replace(',', '').astype(float)
        df['Price Range'] = df['Price Range'].replace('-', '0-0').fillna('0-0')
        df['Min Price'] = df['Price Range'].str.split('-').str[0].str.replace(',', '').replace('', '0').astype(float)
        df['Max Price'] = df['Price Range'].str.split('-').str[1].str.replace(',', '').replace('', '0').astype(float)
        df = df.dropna(subset=['Average Price', 'Min Price', 'Max Price'])
        df.drop(columns=['Price Range','City','Type'],inplace=True)
        df['Year'] = df['Year'].astype('float64')
        df = df.drop(df[df['Month'] == 'April-June'].index)
        
        #scaling prices 
        columns_to_scale = ['Average Price', 'Min Price', 'Max Price']
        scaler = MinMaxScaler(feature_range=(1, 10))
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        
        #locality encoding and then scaling
        encoder = LabelEncoder()
        df['Locality'] = encoder.fit_transform(df['Locality'])
        scaler = MinMaxScaler(feature_range=(1,10))
        df[['Locality']] = scaler.fit_transform(df[['Locality']])

        month_mapping = {
            "Jan-Mar": 1,
            "Apr-Jun": 2,
            "Jul-Sep": 3,
            "Oct-Dec": 4
        }
        df['Month_Mapped'] = df['Month'].map(month_mapping)
        df.drop(columns=['Month'])
        df['Year_sin'] = np.sin(2 * np.pi * (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min()))
        df['Year_cos'] = np.cos(2 * np.pi * (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min()))
        df.drop(columns=['Year','Month','Month_Mapped'],inplace=True)
        logger.debug('Data Preprocessing completed.')
        return df
    except KeyError as e:
        logger.error('Missing coloumn in the DataFrame: %s',e)
        raise
    except Exception as e:
        logger.error('Unexcepted error occured during preprocessing :%s',e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str) -> None:
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug("Train and Test Data Saved to %s",raw_data_path)
    except Exception as e:
        logger.error('Unexcepted error occured while saving the data %s',e)
        raise
    
def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/manandarak/manandarak/refs/heads/main/Mumbai%20Property%20Data.csv'
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the Data Ingestion process %s',e)
        print(f'Error:{e}')
        
if __name__ == '__main__':
    main()