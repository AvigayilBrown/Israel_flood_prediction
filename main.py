import csv
import pandas as pd
from Basin import Basin

TIME = {100: '01', 200: '02', 300: '03', 400: '04', 500: '05', 600: '06', 700: '07'
    , 800: '08', 900: '09', 1000: '10', 1100: '11', 1200: '12', 1300: '13', 1400: '14'
    , 1500: '15', 1600: '16', 1700: '17', 1800: '18', 1900: '19', 2000: '20'
    , 2100: '21', 2200: '22', 2300: '23', 2400: '0'}


def process_data(stations_id, metatok):
    """
    extract data for each one of selected basins to csv file
    """
    # selected basins
    df = pd.read_excel(stations_id)
    id = df['ID'].to_list()
    selected_basins = {}
    header = ['station', 'year', 'month', 'day', 'time', 'is_row_bad', 'Level', 'Battery']
    for i in id:
        file_name = 'data_' + str(i) + '.csv'
        selected_basins[i] = file_name

        # with open(file_name, 'w',newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(header)

        # for file in metatok:
        #     df = pd.read_csv(file)
        #     if i in df['station'].values:
        #         df = df[df['station'] == i]
        #         df = df[df['time'] % 100 == 0] # takeout not round hours
        #         df = df.replace({'time': TIME})
        #         df['time'] = df['time'].astype(int)
        #
        #         # df['date'] = pd.to_datetime(df['year'] * 10000 + df['month'] * 100 + df['day'], format='%Y%m%d')
        #         df.to_csv(file_name, mode='a', sep='\t', index=False, header=False)
    return selected_basins


def create_basins(selected_basins, stations_info):
    for id, file_flow in selected_basins.items():
        b = Basin(id, file_flow, stations_info)
        b.process_flow_file()
        b = Basin(id, file_flow)
        b.process_flow_file()
        b.create_features()
        b.split_data()
        y_pred = b.model()
        b.asses_models(y_pred)
        b.plot_prediction(y_pred)

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
    create_basins(selected_basins, station_info)
