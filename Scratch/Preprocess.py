import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.exceptions import DataConversionWarning

import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
class Preprocessor:
    def __init__(self, drop=['Plot', 'Year', 'Date', 'Range', 'Row', 'left', 
                             'top', 'right', 'bottom', 'Mean.Yld.bu.ac', 
                             'Yld Vol(Dr', 'Crop Flw(M', 'Crop Flw(V', 'Elevation(',
                             'Moisture(%'
                            ], inplace=False, test_size=0.3, target=['Yld Mass(D'], random_state=42):
        self.drop = drop
        self.inplace = inplace
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
    
    def __drop__(self):
        self.train_validate_fields.drop(columns=self.drop, inplace=True)

    def __scale__(self):
        scaler = MinMaxScaler()
        scalecols = [col for col in self.train_validate_fields if col not in ['X', 'Y', 'FIELD_ID']]
        self.train_validate_fields[scalecols] = scaler.fit_transform(self.train_validate_fields[scalecols])

    def __combine_train_indices__(self):
        for i, field in enumerate(self.train_validate_fields):
            field['FIELD_ID'] = i+1
        
        self.train_validate_fields = pd.concat(self.train_validate_fields).reset_index(drop=True)
        
    def transform(self, train_validate_fields: list):
        self.train_validate_fields = train_validate_fields
        self.__combine_train_indices__()
        self.__drop__()
        self.__scale__()

        xCols = [col for col in self.train_validate_fields if col not in self.target and col not in ['id_old']]
        X_train_validate = self.train_validate_fields[xCols]
        Y_train_validate = self.train_validate_fields[self.target]

        return X_train_validate, Y_train_validate

