import csv
import pandas as pd
from Basin import Basin


def process_data(stations_id, stations_info, metatok):
    """
    extract data for each one of selected basins to csv file
    """
    # selected basins
    df = pd.read_excel(stations_id)
    id = df['ID'].to_list()
    selected_basins = {}
    header = ['station', 'year', 'month', 'day', 'time', 'is_row_bad', 'Level', 'Battery']
    print(id)
    for i in id:
        file_name = 'data_' + str(i) + '.csv'
        selected_basins[i] = file_name

        with open(file_name, 'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        for file in metatok:
            df = pd.read_csv(file)
            if i in df['station'].values:
                df2 = df[df['station'] == i]
                print(str(i) + " in " + str(file_name))
                print(df2)
                df2.to_csv(file_name, mode='a', sep=',', index=False, header=False)
    return selected_basins


def create_basins(selected_basins, stations_info):
    for id, file_flow in selected_basins.items():
        Basin(id, file_flow, stations_info)
    print(Basin.counter)


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

    selected_basins = process_data(stations_id, station_info, metatok)
    # create_basins(selected_basins, station_info)
