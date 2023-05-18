import pickle
import pandas as pd
import numpy as np

class HealthInsurance:
    
    def __init__(self):
        self.home_path = ''
        self.annual_premium_scaler =                                 pickle.load(open(self.home_path + 'paramater/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler =                                            pickle.load(open(self.home_path + 'paramater/age_scaler.pkl', 'rb'))
        self.vintage_scaler =                                        pickle.load(open(self.home_path + 'paramater/vintage_scaler.pkl', 'rb'))
        self.target_encoder_gender_scaler =                          pickle.load(open(self.home_path + 'paramater/target_encoder_gender_scaler.pkl', 'rb'))
        self.target_encoder_region_code_scaler =                     pickle.load(open(self.home_path + 'paramater/target_encoder_region_code_scaler.pkl', 'rb'))
        self.fe_policy_sales_channel_scaler =                        pickle.load(open(self.home_path + 'paramater/fe_policy_sales_channel_scaler.pkl', 'rb'))       
        
    def data_cleaning(self, data):
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 'vehicle_damage', 'annual_premium',
       'policy_sales_channel', 'vintage', 'response']
        
        data.columns = cols_new
        
        return data
    
    def feature_engineering(self, data):
        data['vehicle_age'] = data['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_years' if x == '1-2 Year' else 'below_1_year')

        data['vehicle_damage'] = data['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        return data
        
    def data_preparation (self, data):
        # annual_premium
        data['annual_premium'] = self.annual_premium_scaler.transform(data[['annual_premium']].values)
        
        # Age
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # vintage
        data['vintage'] = self.vintage_scaler.transform(data[['vintage']].values)
        
        # gender
        data.loc[:,'gender'] = data['gender'].map(self.target_encoder_gender_scaler)

        # region_code - Target Encoding / Frequency Encoding
        data.loc[:, 'region_code'] = data['region_code'].map(self.target_encoder_region_code_scaler)

        # vehicle_age
        data = pd.get_dummies(data, prefix='vehicle_age', columns=['vehicle_age'])

        # policy_sales_channel
        data.loc[:, 'policy_sales_channel'] = data['policy_sales_channel'].map(self.fe_policy_sales_channel_scaler)
        
        # Feature Selection
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured',
                 'policy_sales_channel']
        
        return data[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # Model Prediction
        pred = model.predict_proba(test_data)
        
        #Join prediction to original data
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json(orient = 'records', date_format='iso')
