import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

class GridCV:
    def __init__(self, models, scoring = ['neg_root_mean_squared_error', 'r2'], random_state=42):
        self.models_ = models
        self.scoring_ = scoring
        self.random_state = random_state
        self.fold_results = {}

        self.results_table = pd.DataFrame(columns=['Spatial Grid', 'TYPE', 'RMSE_AVG', 'R2_AVG', 'RMSE STD', 'R2 STD'])

    def build_folds(splits: list, total_indices: np.int32, n_fields: np.int32, splits_per_field: np.int32, method: str):

        if method == 'single': return splits

        folds = []
        for split in range(splits_per_field):
            validate_indices = []
            for field in range(n_fields):
                validate_indices.extend(splits[split + splits_per_field * field][1])
            
            train_indices = np.array([i for i in range(total_indices) if i not in validate_indices])
            folds.append((train_indices, validate_indices))

        return folds
        

    def split(fields: list, n_splits: np.int32 = 6, method='single'):
        allowed_methods = ['single', 'representative']
        assert method in allowed_methods, 'Fold Building Method must be in allowed methods [' + ', '.join(allowed_methods) + ']'


        splits = []
        index_offset = 0
        n_fields = len(fields)
        splits_per_field = n_splits // n_fields
        assert n_fields * splits_per_field == n_splits, "Folds must be evenly divisible by number of fields"

        total_indices = np.sum(len(field) for field in fields)
        for field in fields:
            X_min = np.min(field['X'])
            X_max = np.max(field['X'])
            r = np.abs(X_min - X_max) / splits_per_field
            for i in range(splits_per_field):
                left = X_min + (i * r)
                right = X_min + ((i + 1) * r)

                validate_indices = np.array(field[field['X'].between(left, right)].index) + index_offset
                train_indices = np.array([i for i in range(total_indices) if i not in validate_indices])
                splits.append((train_indices, validate_indices))
            
            index_offset += len(field)

        return GridCV.build_folds(splits, total_indices, n_fields, splits_per_field, method)
    


    def __run__(self):
        for name in self.models_:
            model = self.models_[name]
            scores = cross_validate(model, self.X_train_validate, self.y_train_validate, scoring=self.scoring_, cv=self.folds, return_estimator=True)
            cv_RMSE_ = -scores['test_neg_root_mean_squared_error']
            cv_R2 = scores['test_r2']


            cv_RMSE_std = np.std(cv_RMSE_)
            cv_R2_std = np.std(cv_R2)

            self.fold_results[name] = {'RMSE': cv_RMSE_, 'R2': cv_R2}

            self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'CV', np.average(cv_RMSE_), np.average(cv_R2), cv_RMSE_std, cv_R2_std]
        

    def results(self, X_train: pd.DataFrame, y_train: pd.DataFrame, folds: list):
        spatial_cols = ['X', 'Y']
        feature_cols = [col for col in X_train.columns if col not in spatial_cols and col != 'FIELD_ID']
        self.X_train_validate = X_train[feature_cols]
        self.y_train_validate = y_train
        self.train_spatial_coordinates = X_train[spatial_cols]
        self.folds = folds

        self.__run__()

        return self.results_table


    def display_fold(self, fold_index: np.int32):
        fold_train_indices = self.folds[fold_index][0]
        fold_validate_indices = self.folds[fold_index][1]

        train = self.train_spatial_coordinates[np.in1d(self.train_spatial_coordinates.index, fold_train_indices)]
        validate = self.train_spatial_coordinates[np.in1d(self.train_spatial_coordinates.index, fold_validate_indices)]

        plt.scatter(train['X'], train['Y'], color='black', label='Train')
        plt.scatter(validate['X'], validate['Y'], color='red', label='Validate')
        plt.legend()
        plt.title("Spatial Grid CV Fold " + str(fold_index+1))
        plt.show()

        print('Train Samples   : ', len(fold_train_indices))
        print('Validate Samples: ', len(fold_validate_indices))
        print('LR Fold CV RMSE Scores: ', self.fold_results['LR']['RMSE'])
        print('LR Fold CV R2 Scores  : ', self.fold_results['LR']['R2'])
        print('RF Fold CV RMSE Scores: ', self.fold_results['RF']['RMSE'])
        print('RF Fold CV R2 Scores  : ', self.fold_results['RF']['R2'])