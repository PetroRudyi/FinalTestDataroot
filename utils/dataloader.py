import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):

        # drop_columns
        drop_list = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Location', 'MinTemp', 'WindGustDir',
                     'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Temp9am']
        self.dataset = self.dataset.drop(drop_list, axis=1)

        title_list = ['Date', 'Rainfall', 'RainToday',
                      'MaxTemp', 'WindGustSpeed', 'Humidity9am',
                      'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp3pm',
                      'Date_year', 'Date_month', 'Date_day', 'Date_weekday']

        # fill_nans mean
        listOfChangeFload = ['MaxTemp', 'WindGustSpeed', 'Humidity9am',
                             'Humidity3pm', 'Pressure9am',
                             'Pressure3pm', 'Temp3pm', 'Rainfall']
        for i in listOfChangeFload:
            self.dataset[i] = self.dataset[i].fillna(self.dataset[i].mean())

        # fill_nans mode
        listOfChangeObj = ['RainToday']

        for i in listOfChangeObj:
            self.dataset[i] = self.dataset[i].fillna(self.dataset[i].mode()[0])

        # replace
        '''
        mean1 = (self.dataset[self.dataset['RainToday'] == "Yes"][['Rainfall']].mean())[0]
        self.dataset.loc[self.dataset['RainToday'] == "Yes", 'Rainfall'] = \
        self.dataset[self.dataset['RainToday'] == "Yes"]['Rainfall'].fillna(
            mean1)

        # Якщо RainToday No
        mean1 = (self.dataset[self.dataset['RainToday'] == "No"][['Rainfall']].mean())[0]
        self.dataset.loc[self.dataset['RainToday'] == "No", 'Rainfall'] = \
        self.dataset[self.dataset['RainToday'] == "No"]['Rainfall'].fillna(mean1)

        # заповнемо все інше
        mean1 = (self.dataset[['Rainfall']].mean())[0]
        self.dataset['Rainfall'] = self.dataset['Rainfall'].fillna(mean1)
        '''
        # encode labels

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()

        le.fit(self.dataset['RainToday'])
        self.dataset['RainToday'] = le.transform(self.dataset['RainToday'])  # obj


        # date
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'], format="%Y-%m-%d")

        self.dataset['Date_year'] = pd.DatetimeIndex(self.dataset['Date']).year
        self.dataset['Date_month'] = pd.DatetimeIndex(self.dataset['Date']).month
        self.dataset['Date_weekday'] = pd.DatetimeIndex(self.dataset['Date']).weekday
        self.dataset['Date_day'] = pd.DatetimeIndex(self.dataset['Date']).dayofyear

        # Normalization

        listOfChangeFload = ['Rainfall', 'RainToday',
                             'MaxTemp', 'WindGustSpeed', 'Humidity9am',
                             'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp3pm',
                             'Date_year', 'Date_month', 'Date_weekday', 'Date_day']

        for column in listOfChangeFload:
            self.dataset[column] = (self.dataset[column] - self.dataset[column].min()) / (
                    self.dataset[column].max() - self.dataset[column].min())

        #DROP Date
        self.dataset = self.dataset.drop('Date', axis=1)

        return self.dataset
