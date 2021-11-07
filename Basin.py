import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from datetime import datetime
from pandas import DataFrame
import matplotlib.dates as dates
from matplotlib.dates import DateFormatter

""" todo: 
timestamps
create X,y
split train, val, pred
make model
make plot"""


class Basin:
    """
    This class represents a single basin. 
    """

    # counts the number of basins
    counter = 0

    def __init__(self, id, file_flow, stations_info=None):
        Basin.counter += 1

        self.id = np.int64(id)

        # basin files
        self.f_flow = file_flow
        self.info = stations_info
        # self.f_prcp = file_prcp

        self.area = None
        self.name = None
        self.latitude = None  # m
        self.elevation = None  # m

        self.data = None
        self.X = None  # features
        self.y = None  # labels
        self.f = {}  # features
        self.timestamp = None

    def process_station_info(self):
        df = pd.read_excel(self.info, index_col=None)
        if self.id in df['id'].values:
            df = df[df['id'] == self.id]
            self.name = df['name'].iloc[0]
            self.area = df['area'].iloc[0]

    def process_flow_file(self):
        df = pd.read_csv(self.f_flow, sep='\t')
        timestamp = pd.DataFrame({'year': df.year,
                                  'month': df.month,
                                  'day': df.day,
                                  'hour': df.time})
        timestamp = pd.to_datetime(timestamp)
        df['date'] = timestamp
        # todo take out year/month/day/time
        # df = df.drop(['year','month','day','time'], 1)
        df = df.dropna(axis=0)
        self.data = df

    def preprocess_data(self):
        """
        taking out unreliable data points based on sudden change in water
        levels.
        """
        level = self.data['Level']

        # take out data point with many consecutive values
        df = level.diff().ne(0).cumsum()
        self.data['flag'] = df.groupby([level, df]).transform('size').ge(50).astype(int)
        self.data.loc[self.data.flag == 1, 'Level'] = np.nan
        self.data = self.data.drop(['flag'], 1)

        # change values of zero to be Nan
        self.data['Level'] = level.replace(0.0, np.NaN)

        # identify median of dry months for base threshold
        dry_season = self.data[(level != np.nan) & (self.data['month'].between(6, 9))]
        threshold = dry_season['Level'].median()

        # remove data points that are much below threshold
        # todo based on גודל יחסי , and should i make threshold or Nan? 17135
        # todo if exceeds Q1-Q3?
        # todo 19185 threshold -0.2
        self.data.loc[self.data.Level - threshold <= -0.2, 'Level'] = np.nan


        # todo 14120 2013-01-31 9:00 jump of over one, but gradual desend - in winter!

        # todo 17135 no major jumps - flatline problem
        # todo 2016-04-03 14:00 2016-11-01 10:00

    def split_data(self):
        """
        split data into training, validation and test sets
        """
        # todo add validation split, and add to features
        # todo talk about size of split (ratio)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        # X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

        split = round(self.data.shape[0] // 78 * (0.8))
        X_train, X_test = np.split(self.X, [split], axis=0)
        y_train, y_test = np.split(self.y, [split], axis=0)
        timestamp_train, self.timestamp = np.split(self.timestamp, [split], axis=0)
        self.f = {"X_train": X_train, "y_train": y_train,
                  "X_test": X_test, "y_test": y_test}

    def create_features(self):
        """
        feature - 72 hours of streamflow data
        label - streamflow in 6 hours
        """
        d = 72
        lag = 5  # 6 hours ahead
        m = self.data.shape[0] // 78
        X = np.zeros((m, d))
        flow = self.data['Level']

        for i in range(m):
            X[i] = flow.iloc[i:i + d]
        y = flow[d + lag:d + m + lag].reset_index(drop=True)
        self.timestamp = self.data['date'][d + lag:d + m + lag].reset_index(drop=True)
        self.X = X
        self.y = y

    def model(self):
        """
        regression model for stream prediction.
        """
        # regularization: minus min divide by difference max min
        X_train = self.f['X_train']
        X_min, X_max = X_train.min(), X_train.max()
        X_train = (X_train - X_min) / (X_max - X_min)

        y_train = self.f['y_train']
        y_min, y_max = y_train.min(), y_train.max()
        y_train = (y_train - y_min) / (y_max - y_min)

        # linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        X_test = (self.f['X_test'] - X_min) / (X_max - X_min)
        y_pred = model.predict(X_test)
        y_pred = y_pred * (y_max - y_min) + y_min
        return y_pred

    def NSE(self, y_pred, y_test):
        """
        uses NSE as the prediction assessment for model
        """
        # todo fix function
        error_variance = np.sum(np.power((y_pred - y_test), 2))
        variance = np.sum(np.power((y_test - np.mean(y_test)), 2))
        NSE = 1 - (error_variance / variance)
        return NSE

    def asses_models(self, y_pred):
        """
        assessing model performance using NSE coefficient.
        """
        # unregularize

        # with open(self.nse_file, 'a', newline='') as f:
        #     writer = csv.writer(f, delimiter=',')
        #     if Basin.counter == 1:
        #         writer.writerow(["NSE"])
        #     writer.writerow([str(self.NSE(y_pred, self.y_test))])
        print(self.NSE(y_pred, self.f['y_test']))

    def plot_prediction(self, y_pred):
        """
        plots the streamflow prediction vs reality per day.
        """
        fig, ax = plt.subplots()
        plt.title(label="Predicted and actual streamflow per day")
        # ax.plot(self.f['y_test'], label="actual")
        ax.set_xlabel("year")
        ax.set_ylabel("streamflow mm/day")
        # ax.plot(self.timestamp,y_pred, label="predicted")
        ax.legend()
        plt.show()

    def clean_data(self):
        pass

    def plot_streamflow(self):
        """
        plots the streamflow of all available data.
        """
        fig, ax = plt.subplots()
        plt.title(label="streamflow of Basin: " + self.name + " id: " + str(self.id))
        ax.plot(self.data['date'], self.data['Level'])
        ax.set_xlabel("year")
        ax.set_ylabel("Level m")
        plt.show()


if __name__ == '__main__':
    pass
    # file_flow = "C:\\Users\\asus\\Desktop\\Israel_flood_prediction\\data_14120.csv"
    # stations_id = "data//stations.xlsx"
    # station_info = "data//קטלוג_תחנות_הידרומטריות.xlsx"
    # b = Basin(id, file_flow)
    # b.process_flow_file()
    # b.create_features()
    # b.split_data()
    # y_pred = b.model()
    # b.asses_models(y_pred)
    # # b.plot_prediction(y_pred)
