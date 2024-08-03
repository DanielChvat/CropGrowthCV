import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_validate, StratifiedShuffleSplit
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

class RandomCV:
    def __init__(self, models, n_folds=4, scoring = ['neg_root_mean_squared_error', 'r2'], date='', test_size=0.3, random_state=42):
        self.models_ = models
        self.n_folds = n_folds
        self.scoring_ = scoring
        self.test_size = test_size
        self.random_state = random_state
        self.fold_results = {}

        self.results_table = pd.DataFrame(columns=['Random', 'TYPE', 'RMSE_AVG', 'R2_AVG', 'RMSE STD', 'R2 STD'])

    def __run__(self):
        self.cv_method = StratifiedShuffleSplit(n_splits=self.n_folds, test_size=self.test_size, random_state=self.random_state)
        splits = self.cv_method.split(self.X_train_validate, self.groups)
        train_indices, test_indices = [list(traintest) for traintest in zip(*splits)]
        folds = [*zip(train_indices,test_indices)]
        for name in self.models_:
            model = self.models_[name]
            scores = cross_validate(model, self.X_train_validate, self.y_train_validate, scoring=self.scoring_, cv=folds, return_estimator=True)
            cv_RMSE_ = -scores['test_neg_root_mean_squared_error']
            cv_R2 = scores['test_r2']

            cv_RMSE_std = np.std(cv_RMSE_)
            cv_R2_std = np.std(cv_R2)

            self.fold_results[name] = {'RMSE': cv_RMSE_, 'R2': cv_R2}

            self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'CV', np.average(cv_RMSE_), np.average(cv_R2), cv_RMSE_std, cv_R2_std]
        
    def display_fold(self, fold_index: np.int32):
        splits = self.cv_method.split(self.X_train_validate, self.groups)
        train_indices, test_indices = [list(traintest) for traintest in zip(*splits)]
        folds = [*zip(train_indices,test_indices)]

        fold_train_indices = folds[fold_index][0]
        fold_validate_indices = folds[fold_index][1]

        train = self.train_spatial_coordinates[np.in1d(self.train_spatial_coordinates.index, fold_train_indices)]
        validate = self.train_spatial_coordinates[np.in1d(self.train_spatial_coordinates.index, fold_validate_indices)]

        plt.scatter(train['X'], train['Y'], color='black', label='Train')
        plt.scatter(validate['X'], validate['Y'], color='red', label='Validate')
        plt.legend()
        plt.title("Random CV Fold")
        plt.show()

        print('Train Samples   : ', len(fold_train_indices))
        print('Validate Samples: ', len(fold_validate_indices))
        print('LR Fold CV RMSE Scores: ', self.fold_results['LR']['RMSE'])
        print('LR Fold CV R2 Scores  : ', self.fold_results['LR']['R2'])
        print('RF Fold CV RMSE Scores: ', self.fold_results['RF']['RMSE'])
        print('RF Fold CV R2 Scores  : ', self.fold_results['RF']['R2'])

    def results(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        spatial_cols = ['X', 'Y']
        self.groups = X_train['FIELD_ID']
        feature_cols = [col for col in X_train.columns if col not in spatial_cols and col != 'FIELD_ID']
        self.X_train_validate = X_train[feature_cols]
        self.y_train_validate = y_train
        self.train_spatial_coordinates = X_train[spatial_cols]

        self.__run__()

        return self.results_table