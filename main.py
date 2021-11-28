import csv
import pandas as pd
from Basin import Basin
import matplotlib.pyplot as plt

TIME = {100: '01', 200: '02', 300: '03', 400: '04', 500: '05', 600: '06', 700: '07'
    , 800: '08', 900: '09', 1000: '10', 1100: '11', 1200: '12', 1300: '13', 1400: '14'
    , 1500: '15', 1600: '16', 1700: '17', 1800: '18', 1900: '19', 2000: '20'
    , 2100: '21', 2200: '22', 2300: '23', 2400: '24'}


def process_data(stations_id, metatok):
    """
    extract data for each one of selected basins to csv file
    """
    # selected basins
    df = pd.read_excel(stations_id)
    id = df['ID'].to_list()
    selected_basins = {}
    header = ['station', 'year', 'month', 'day', 'hour', 'is_row_bad', 'Level', 'Battery']
    for i in id:
        has_header = False
        file_name = 'data_' + str(i) + '.csv'
        selected_basins[i] = file_name

        # for file in metatok:
        #     if not has_header:
        #         df = pd.read_csv(file, nrows=0)
        #         header = df.columns
        #         df.to_csv(file_name,mode='a',header=header,index=False,sep="\t")
        #         has_header = True
        #
        #     df = pd.read_csv(file)
        #
        #     if i in df['station'].values:
        #         df = df[df['station'] == i]
        #         df = df[df['time'] % 100 == 0]  # takeout not round hours
        #         df = df.replace({'time': TIME})
        #         df['time'] = df['time'].astype(int)
        #         df['Level'] = df['Level'].astype(float)
        #         df.to_csv(file_name, mode='a', sep='\t',header=False,index=False,index_label=True)
    return selected_basins


def create_basins(selected_basins, stations_info):
    for id, file_flow in selected_basins.items():
        b = Basin(id, file_flow, stations_info)

        b.process_station_info()
        b.process_flow_file()
        if id == 15120:
            b.preprocess_data_15120()
        elif id == 17168:
            b.preprocess_data_17168()

        x = range(3,7)
        for i in x:
            b.create_features(i)
            b.cross_val()

        # b.plot_streamflow()
        # b.plot_prediction()

        # y_pred = b.model()
        # b.asses_models(y_pred)
        return x,Basin.total_nse,b

def plot_NSE(x,y,basin):
    for xe, ye in zip(x, y):
        plt.scatter([xe] * len(ye), ye)

    plt.xticks([3, 4,5,6])
    plt.title(label="NSE per lead time " +str(basin.name)+ " id: " + str(basin.id))
    plt.xlabel("lead time(hr)")
    plt.ylabel("NSE")
    plt.show()

if __name__ == '__main__':
    # general station files
    stations_id = "data//stations.xlsx"
    station_info = "data//קטלוג_תחנות_הידרומטריות.xlsx"

    # metatok data per year
    metatok_1 = "data//ihs_1hr_2008__2009.csv"
    metatok_2 = "data//ihs_1hr_2010__2014.csv"
    metatok_3 = "data//ihs_1hr_2015__2018.csv"
    metatok_4 = "data//ihs_1hr_2019__2021.csv"
    metatok = [metatok_1, metatok_2, metatok_3, metatok_4]

    selected_basins = process_data(stations_id, metatok)
    single = {int('17168'): 'data_17168.csv'}
    # todo problem with intercept, leave out a year that problematic (+3.1)
    single = {int('15120'): 'data_15120.csv'}

    x,y,basin = create_basins(single, station_info)
    plot_NSE(x,y,basin)

"""



last year leave out for test
and train is for cross validation - each time leave out a whole year
linear regression, ridge or lasso.
NSE for each basin for each year, get the התפלגות of the nse per year
LSTM 
"""
