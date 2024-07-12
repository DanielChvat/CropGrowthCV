import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.exceptions import DataConversionWarning

import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
class CombinedPreprocessor:
    def __init__(self, drop=['Plot', 'Year', 'Date', 'Range', 'Row', 'left', 
                             'top', 'right', 'bottom', 'Mean.Yld.bu.ac', 
                             'Yld Vol(Dr', 'Crop Flw(M', 'Crop Flw(V'
                             ], inplace=False, test_size=0.3, target=['Yld Mass(D'], random_state=42, combination_method='average'):
        self.drop = drop
        self.inplace = inplace
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.combination_method = combination_method
    
    def __drop__(self):
        if self.combination_method == 'average':
            self.combined_mat.drop(columns=self.drop, inplace=True)
        else:
            pass

        self.test_indices.drop(columns=self.drop, inplace=True)

    def __scale__(self):
        self.scaler = MinMaxScaler()
        scalecols = [col for col in self.combined_mat if col not in ['X', 'Y']]
        self.combined_mat[scalecols] = self.scaler.fit_transform(self.combined_mat[scalecols])
        self.test_indices[scalecols] = self.scaler.transform(self.test_indices[scalecols])

    def __combine_train_indices__(self):
        allowed_methods = ['average', 'concatenate']
        if self.combination_method == 'average':
            self.combined_mat = np.zeros(self.train_validate_indices[0].shape)
            
            for indices in self.train_validate_indices:
                self.combined_mat += indices.to_numpy()

            self.combined_mat /= len(self.train_validate_indices)

            self.combined_mat = pd.DataFrame(self.combined_mat, columns=self.train_validate_indices[0].columns)

        elif self.combination_method == 'concatenate':
            pass
        else:
            raise ValueError(self.combination_method, " Not Found In Allowed Methods: ", allowed_methods)
        

    def transform(self, train_validate_indices: list, test_indices: pd.DataFrame):
        self.train_validate_indices = []
        for indices in train_validate_indices:
            self.train_validate_indices.append(indices if self.inplace else indices.copy(deep=True))

        self.test_indices = test_indices

        self.__combine_train_indices__()
        self.__drop__()
        self.__scale__()

        xCols = [col for col in self.combined_mat.columns if col not in self.target and col not in ['id_old']]

        X_train_validate = self.combined_mat[xCols]
        X_test = self.test_indices[xCols]

        Y_train_validate = self.combined_mat[self.target]
        Y_test = self.test_indices[self.target]

        return X_train_validate, X_test, Y_train_validate, Y_test

    def inverse_transform(self):
        pass