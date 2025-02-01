from enum import Enum
import time 
import pandas as pd 
import numpy as np
import sys 
import tkinter as tk
from tkinter import ttk
from frontend import ModelApp

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

class Steps(Enum):
    INTRO = 1
    READFILE = 2
    DISPLAYFILES = 3
    SETPROBLEM = 4
    CLEANDATA = 5
    INITMODEL = 6

class CustomModel:
    
    def __init__(self, df, modelType, x_features, y_feature):
        self.df = df
        self.modelType = modelType
        self.X = self.df[x_features]
        self.y = self.df[y_feature]

    def generatePreprocessor(self, scalerType=StandardScaler):
        numerical_pipeline = Pipeline([('Imputer', KNNImputer(n_neighbors=5)),
                               ('Scaler', scalerType())])

        categorical_pipeline = Pipeline([('Encoder', OrdinalEncoder()),
                                 ('Imputer', KNNImputer(n_neighbors=5))])

        data_pipeline = ColumnTransformer([('numerical', numerical_pipeline, self.X.select_dtypes(exclude='object').columns),
                                   ('categorical', categorical_pipeline, self.X.select_dtypes(include='object').columns)])

        return data_pipeline
    
    def generateModel(self, preprocessor, modelType='linear'):
        if modelType == 'Linear':
            model = LinearRegression()
            param_grid = {'fit_intercept': [True, False]}

        elif modelType == 'Logistic':
            model = LogisticRegression()
            c_space = np.logspace(-5, 8, 15)
            param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

        elif modelType == 'RandomForest':
            model = RandomForestRegressor()
            param_grid = {
                'n_estimators': [25, 50, 100],  
                'max_features': ['auto', 'sqrt'], 
                'max_depth': [5, 10, 20, None],  
                'min_samples_split': [2, 5],  
                'min_samples_leaf': [1, 2],  
                'bootstrap': [True]  
            }
            
        elif modelType == 'SVM':
            model = SVR()
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf']}

        elif modelType == 'DecisionTree':
            model = DecisionTreeClassifier()
            param_grid = {'criterion': ['gini', 'entropy'],
                          'splitter': ['best', 'random'],
                          'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 2, 4]}
            
        return GridSearchCV(model, param_grid, cv=5)
    
    def trainModel(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        
        preprocessor = self.generatePreprocessor()
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        model = self.generateModel(preprocessor, modelType=self.modelType)
        
        model.fit(X_train_preprocessed, y_train)
        print(f"Best Parameters: {model.best_params_}")
        
        return model, X_test, y_test
    
def run():
    opModeIsActive = True
    currentStep = Steps.INTRO
    
    while opModeIsActive:
        match currentStep.name:
            case "INTRO":
                print("Welcome to blackbox AI, are you ready?")
                time.sleep(3)
                currentStep = Steps.READFILE
                
            case "READFILE":
                # f = input("Input filename: ")
                f = "car_price_prediction.csv" # Testing
                try:
                    df = pd.read_csv(f)
                    currentStep = Steps.DISPLAYFILES
                except FileNotFoundError:
                    print("That file was not found. Try again.")
                    currentStep = Steps.READFILE
                    
            case "DISPLAYFILES":
                print(f"\nFile: {f} loaded.")
                print("Here's a summary of your file: \n")
                print(df.info())
                print("\n\n")
                print(df.head(5))
                currentStep = Steps.SETPROBLEM
    
            case "SETPROBLEM":
                # predictionType = input("Select ML Type: ")
                predictionType = "RandomForest" # Testing
                print(f"Available Features: {list(df.columns)}")
                # features = input("Select features: ").split(', ')
                features = ['Car ID', 'Brand', 'Year', 'Engine Size', 'Fuel Type', 'Transmission', 'Mileage', 'Condition', 'Model'] # Testing
                print(f"\nAvailable Response Vars: {set(df.columns)-set(features)}")
                # response = input("Select response: ")
                response = 'Price' # Testing
                currentStep = Steps.INITMODEL
                
            case "INITMODEL":
                model = CustomModel(df, predictionType, features, response)
                trainedModel, X_test, y_test = model.trainModel()
                y_pred = trainedModel.predict(X_test)
                print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")
                currentStep = Steps.INTRO
                opModeIsActive = False
                
if __name__ == "__main__":
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()

