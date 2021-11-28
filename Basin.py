import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
    total_nse = []

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
        self.features = None
        self.labels = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
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
        self.data = df

    def preprocess_data_17168(self):
        level = self.data['Level']

        # change value of sudden unrealistic drop
        index_min = self.data.Level.idxmin()
        self.data.iloc[index_min,6] = 11.9

        # missing data for year 2015 months 7 and 8
        index_min = self.data.Level.idxmin()
        self.data.iloc[index_min,6] = 11.9


        df = self.data
        diff = 12.13-11.9
        df.loc[((df['year'] ==2018)&(df['month'] >= 10))|((df['year'] ==2019) & (df['month'] <= 5)), 'Level'] -= diff

        index_min = self.data.Level.idxmin()
        self.data.iloc[index_min,6] = 12.8

        # todo make better - lower first half of 2018 (check if all summer)

        # correct rows with Null value
        average = (13.32+13.15)/2
        df.loc[(df['year'] ==2015)&(df['month'] == 1)&((df['day'] ==8) & (df['time'] == 11)),'Level'] = average
        df.loc[(df['year'] == 2015) & (df['month'] == 1) & ((df['day'] == 15) & (df['time'] == 22)), 'Level'] = 12.48
        df.loc[(df['year'] == 2015) & (df['month'] == 5) & ((df['day'] == 3) & (df['time'] == 3)), 'Level'] = 11.9
        df.loc[(df['year'] == 2015) & (df['month'] == 5) & ((df['day'] == 18) & (df['time'] == 15)), 'Level'] = 11.9



    def preprocess_data_15120(self):
        """
        taking out unreliable data points based on sudden change in water
        levels.
        """
        level = self.data['Level']
        # take out data point with many consecutive values

        # df = level.diff().ne(0).cumsum()
        # self.data['flag'] = df.groupby([level, df]).transform('size').ge(50).astype(int)
        # self.data.loc[self.data.flag == 1, 'Level'] = np.nan
        # self.data = self.data.drop(['flag'], 1)

        # identify median of dry months for base threshold
        dry_season = self.data[(level != 0.0) & (self.data['month'].between(6, 9))]
        threshold = dry_season['Level'].median()

        # change values of zero to be Nan
        self.data['Level'] = level.replace(0.0, threshold)

        # remove data points that are much below threshold
        self.data.loc[self.data.Level - threshold <= -0.2, 'Level'] = threshold

        # correct rows with Null value
        df = self.data
        df.loc[(df['year'] == 2016) & (df['month'] == 12) & ((df['day'] == 8) & (df['time'] == 19)), 'Level'] = 4.87
        df.loc[(df['year'] == 2019) & (df['month'] == 7) & ((df['day'] == 10) & (df['time'] == 12)), 'Level'] = 4.99


    def create_features(self,lead_time):
        """
        feature - 72 hours of streamflow data
        label - streamflow in 6 hours
        """
        # remove problematic data points that were set to Nan
        # df = self.data.dropna()

        df = self.data
        # take only wet season: 1st of october till may 31st
        first_year = df['year'].min()
        last_year = df['year'].max()
        labels = []
        features = []

        for i in range(first_year, last_year):
            cur_features = []
            # LABELS - wet season per year
            cur_label = df[(df['year'] == i) & (df['month'] >= 10) | (df['year'] == i + 1) & (df['month'] <= 5)]
            first_index = cur_label.index[0]
            last_index = cur_label.index[-1]
            labels.append(cur_label)

            # FEATURES
            m = cur_label.shape[0]  # number of samples
            d = 72  # number of features
            feature_index = first_index - (d+lead_time)
            for j in range(m):
                cur_features.append(df['Level'].iloc[feature_index + j:feature_index + j + d].reset_index(drop=True))
            features.append(pd.concat(cur_features, axis=1).transpose())

        # test set are values of last hydrological year
        self.x_test = features.pop()
        self.y_test = labels.pop()
        self.features = features
        self.labels = labels

    def cross_val(self):
        """
        The final year of data (2021) is used us test set.
        The linear regression model uses cross validation, so that each year
        is used in turn for validation.
        """
        # linear regression
        years = len(self.features)
        weights = pd.DataFrame(np.zeros((10, 72)))
        f = self.features
        l = self.labels
        intercept = []
        NSE = []
        for i in range(years):
            model = LinearRegression()  # one model for each year left out
            LOO_f = f[:i] + f[i + 1:]  # leave out one
            LOO_l = l[:i] + l[i + 1:]
            X = pd.concat(LOO_f)
            y = pd.concat(LOO_l)
            y = y['Level']

            model.fit(X, y)  # todo uses MSE?
            weights.iloc[i] = model.coef_.T
            intercept.append(model.intercept_)
            y_pred = model.predict(self.features[i])
            # print(self.NSE(y_pred, self.labels[i]['Level']))
            NSE.append(self.NSE(y_pred, self.labels[i]['Level']))

        bias = np.mean(intercept)
        weighted_w = weights.mean(axis=0)
        # todo not supposed to use average for weights - then what to use?
        y_pred = np.matmul(self.x_test, weighted_w) + bias
        self.y_pred = y_pred
        Basin.total_nse.append(NSE)
        # y_test = self.y_test['Level']
        # print(self.NSE(y_pred, y_test))

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

    def plot_prediction(self):
        """
        plots the streamflow prediction vs observed per day.
        """
        fig, ax = plt.subplots()
        plt.title(label="streamflow of Basin: " + self.name + " id: " + str(self.id))
        ax.plot(self.y_test['date'], self.y_test['Level'], label="actual")
        ax.set_xlabel("year")
        ax.set_ylabel("streamflow mm/day")
        ax.plot(self.y_test['date'], self.y_pred, label="predicted")
        ax.legend()
        plt.show()

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
    # todo normalize min and max

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
