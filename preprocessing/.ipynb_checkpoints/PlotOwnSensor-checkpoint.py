from json.decoder import NaN
from statistics import correlation

import requests
import os
import statistics
import numpy
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from numpy.ma.core import indices
from pathlib import Path

# Where to find the input files
input_directory = '../SensingFiles/'
output_directory = '../GeneratedOutput/'
file_extenstion = '.csv'
#Look at https://geo.irceline.be/sos/api/v1/timeseries/ to get updated IDs for the sensors
# File from sensors to be parsed
filename = "sensordata_2026_01_26.csv"
#filename = "sensordata2_2025_12_31.csv"

targets = ['Ozon_value(mV)', 'NO2_value(mV)', 'PM10.0_AVG_PPM']
UTC_MINIMAL = 1735686000 # corresponds to 2025-01-01-00:00:00
#uncomment the right station below to get the right pollutants
#uccle
#pollutants = {'Ozon_value(mV)': '6625', 'NO2_value(mV)': '6622', 'PM10.0_AVG_PPM': '6626', 'PM2.5': '6627' }
#Sint-Pieters-Leeuw
pollutants = {'Ozon_value(mV)': '6719', 'NO2_value(mV)': '6718', 'PM10.0_AVG_PPM': '6626', 'PM2.5': '6627' }
"""------------------------------------------------------------------------------------"""
def load_sensor_data(bestandsnaam):
    try:
        df = pd.read_csv(bestandsnaam, sep=',', engine='python', index_col=False)
        df.columns = df.columns.str.strip()
        print(f"📊 File contains {df.shape[0]} rows and {df.shape[1]} cols.")

        #we have to invert the polarity of the gas sensors
        #df['Ozon_value(mV)'] = -df['Ozon_value(mV)']
        #df['NO2_value(mV)'] = -df['NO2_value(mV)']
        return df

    except FileNotFoundError:
        print(f"❌ Fout: Het bestand '{bestandsnaam}' werd niet gevonden.")
        return None
    except Exception as e:
        print(f"❌ Er is een onverwachte fout opgetreden: {e}")
        return None
"""------------------------------------------------------------------------------------"""
def plot_polluenten(output_directory, df):
    ylabels_pollutants = ['mV','mV','PPM','PPM']
    ylabels_meteo = ['hPa', '%RH', '*C', 'Lx']
    data = numpy.array(df.values)
    # Check welke kolommen daadwerkelijk in de dataset zitten
    targets_pollutants = ['Ozon_value(mV)', 'NO2_value(mV)', 'PM10.0_AVG_PPM', 'CO2(ppm)']
    targets_meteo = ['BME68X_PRES(hPa)','BME68X_HUM(%RH)', 'BME68X_TEMP(*C)','Lux']

    # plot air pollutants
    indices = df.columns.get_indexer(targets_pollutants)
    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 10), sharex=True)
    # Als er maar 1 grafiek is, zorg dat axes een lijst is
    if len(indices) == 1: axes = [axes]

    for i, cols in enumerate(indices):
        # We plotten elke stof in een eigen kleur, data[250:, indices[i]]
        kleuren = ['purple', 'orange', 'green']
        axes[i].plot(df['UTC_TIMESTAMP'][250:],df[targets_pollutants[i]][250:], color=kleuren[i % 3], linewidth=0.7)
        axes[i].set_title(f"{df.columns[indices[i]]}")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel(ylabels_pollutants[i])

    plt.xlabel("Date & Time")
    plt.tight_layout()
    plt.savefig(f'{output_directory}/polluents.pdf')
    plt.show()
    #plot meteo data
    indices = df.columns.get_indexer(targets_meteo)
    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 10), sharex=True)
    # Als er maar 1 grafiek is, zorg dat axes een lijst is
    if len(indices) == 1: axes = [axes]

    for i, cols in enumerate(indices):
        # We plotten elke stof in een eigen kleur, data[250:, indices[i]]
        kleuren = ['purple', 'orange', 'green']
        axes[i].plot(df['UTC_TIMESTAMP'][250:],df[targets_meteo[i]][250:], color=kleuren[i % 3], linewidth=0.7)
        axes[i].set_title(f"{df.columns[indices[i]]}")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel(ylabels_meteo[i])

    plt.xlabel("Date & Time")
    plt.tight_layout()
    plt.savefig(f'{output_directory}/meteo_data.pdf')
    plt.show()

"""------------------------------------------------------------------------------------"""
def recover_time(df):
    df['DATE_DT'] = df['DATE'] + ' ' + df['TIME']
    #df['GPS_TIMESTAMP'] = pd.to_datetime(df['DATE_DT'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    df['GPS_TIMESTAMP'] = pd.to_datetime(df['DATE_DT'], yearfirst=True, errors='coerce')
    # convert to UTC values
    df['UTC'] = df['GPS_TIMESTAMP'].astype('int64') // 10 ** 9
    # now correct the UTC times based on the ms times
    # divide SYS_MILLIS_TIME (ms) by 1000 to get on a second basis
    df['TIME_SEC'] = df['SYS_MILLIS_TIME (ms)']/1000
    # compute the delta between TIME_SEC, which must be positive
    df['TIME_SEC_STEP'] = df['TIME_SEC'].diff()
    df.loc[0,'TIME_SEC_STEP']=0
    df['TIME_SEC_STEP'] = df['TIME_SEC_STEP'].astype(int)
    #check if our internal time is still correct
    # all timings must incremental
    neg_incr_found = True
    median_inc = statistics.median(df['TIME_SEC_STEP'])
    while neg_incr_found:
        time_sec_neg_step = numpy.where(df['TIME_SEC_STEP'] <0)[0]
        if (len(time_sec_neg_step)>0):
            highest_index = time_sec_neg_step[len(time_sec_neg_step)-1]
            for i in range(highest_index,-1,-1):
                df.loc[i,'TIME_SEC'] = df.loc[i+1,'TIME_SEC']-median_inc
        else:
            neg_incr_found = False
        df['TIME_SEC_STEP'] = df['TIME_SEC'].diff()
        df.loc[0, 'TIME_SEC_STEP'] = 0
        df['TIME_SEC_STEP'] = df['TIME_SEC_STEP'].astype(int)
    #correct the first missing times from the GPS
    utc_valid_indices = numpy.where(df['UTC']>=UTC_MINIMAL)[0]
    if len(utc_valid_indices)>0:
        start_index = utc_valid_indices[0]
        #if our first index does not equal the zero index we have to recompute the missing times
        if start_index>0:
            for i in range(start_index,-1,-1): #index -1 will never be done
                df.loc[i,'UTC']=df.loc[i+1,'UTC'] - df.loc[i,'TIME_SEC_STEP']

    #iterate to correct the UTC timestamps as long as negative deltas are found
    neg_delta_found = True
    while neg_delta_found:
        # compute UTC time differences
        df['UTC_STEP'] = df['UTC'].diff()
        # replace the NaNs with 0
        indices_ = numpy.where(df['UTC_STEP'].isna())[0]
        df.loc[indices_, 'UTC_STEP'] = 99999
        df['UTC_STEP'] = df['UTC_STEP'].astype('int64')
        # search for timestep discrepancies
        # first search where time evolution is lower than  0, we cannot backwards in time
        indices_ = numpy.where(df['UTC_STEP'] < 1)[0]
        if len(indices_) == 0:
            neg_delta_found = False
        else:
            # correct these times based on the df['TIME_SEC_STEP']
            for n in indices_:
                df.loc[n, 'UTC'] = df.loc[n - 1, 'UTC'].astype('int64') + df.loc[n, 'TIME_SEC_STEP'].astype('int64')

    pos_delta_found = True
    while pos_delta_found:
        # compute UTC time differences
        df['UTC_STEP'] = df['UTC'].diff()
        # replace the NaNs with 0
        indices_ = numpy.where(df['UTC_STEP'].isna())[0]
        df.loc[indices_, 'UTC_STEP'] = 0
        df['UTC_STEP'] = df['UTC_STEP'].astype('int64')
        # search for timestep discrepancies
        # first search where time evolution is lower than  0, we cannot backwards in time
        indices_ = numpy.where(df['UTC_STEP'] > 100)[0]
        if len(indices_) == 0:
            pos_delta_found = False
        else:
            # correct these times based on the df['TIME_SEC_STEP']
            for n in indices_:
                df.loc[n, 'UTC'] = df.loc[n - 1, 'UTC'].astype('int64') + df.loc[n, 'TIME_SEC_STEP'].astype('int64')

    df['UTC_TIMESTAMP'] = pd.to_datetime(df['UTC'], utc=True, unit='s')
    return df
"""------------------------------------------------------------------------------------"""
def average_per_hour(df):
    #find the start of the hours so we can synchronize with irceline
    #every hour starts at a multiple of 3600 seconds
    hour_multiples = round(df['UTC']/ 3600-0.5)
    #compute the locations when the hours do increment
    hour_inc = hour_multiples.diff()
    hour_inc.loc[0]=1 #reset the first one to 1
    # Find the indices where the hours do change
    indices_ = numpy.where(hour_inc > 0)[0]
    indices_ = np.append(indices_, len(df['UTC']) - 1)
    #now compute the averages between 2 indices, store the hour multiple in a new table
    table_names = ['UTC','TIMESTAMP','T_IRCELINE']
    table_names = np.append(table_names,targets)
    row_depth = len(indices_) - 1
    averaged = np.zeros((row_depth,len(table_names)))
    averaged = pd.DataFrame(averaged, index=range(row_depth),columns=table_names)
    averaged['UTC'] = [0]*row_depth

    averaged['UTC'] = averaged['UTC'].astype('int64')
    averaged['TIMESTAMP'] = averaged['TIMESTAMP'].astype('int64')
    averaged['T_IRCELINE'] = averaged['T_IRCELINE'].astype('int64')
    for i in range(3,len(table_names)):
        averaged[table_names[i]] = [0]*row_depth
        averaged[table_names[i]] = averaged[table_names[i]].astype('float64')

    for n in range(row_depth):
        averaged.loc[n,'UTC']=df['UTC'][indices_[n]]-df['UTC'][indices_[n]]%3600
        averaged.loc[n,'TIMESTAMP'] = pd.to_datetime(averaged.loc[n,'UTC'], utc=True, unit='s')
        averaged.loc[n,'T_IRCELINE'] = pd.to_datetime(averaged.loc[n, 'UTC'], utc=True, unit='s').isoformat()

    for i in range(len(targets)):
        for n in range(row_depth):
            #compute from index indices[n] to indices[n+1]-1
            sum_ = np.sum(df[targets[i]][indices_[n]:indices_[n + 1] - 1])
            averaged.loc[n,targets[i]]=(sum_ / (indices_[n + 1] - indices_[n])).astype('float64')

    return averaged
"""------------------------------------------------------------------------------------"""
def obtain_start_stop_time(df):
    t_start = df['T_IRCELINE'][0]
    t_end = df['T_IRCELINE'][len(df['T_IRCELINE'])-1]
    return t_start + '/' + t_end
"""------------------------------------------------------------------------------------"""
def download_irceline(df):
    all_dataframes = []
    timespan = obtain_start_stop_time(df) # example"2025-12-31T00:00:00/2026-01-05T09:00:00"
    base_url = "https://geo.irceline.be/sos/api/v1/timeseries/"
    headers = {'User-Agent': 'Mozilla/5.0'}

    print(f"Downloading data between timestamps {timespan}")

    for name, ts_id in pollutants.items():
        print(f"Obtaining {name} (ID: {ts_id}) from Irceline")
        url = f"{base_url}{ts_id}/getData"

        try:
            response = requests.get(url, params={"timespan": timespan}, headers=headers, verify=False)
            response.raise_for_status()

            values = response.json().get('values', [])
            if values:
                temp_df = pd.DataFrame(values)
                temp_df.columns = ['timestamp', name]
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms')

                # We zetten de timestamp als index om het samenvoegen makkelijker te maken
                temp_df.set_index('timestamp', inplace=True)
                all_dataframes.append(temp_df)
            else:
                print(f"oops: no data for {name} could be downloaded")

        except Exception as e:
            print(f"Error with {name}: {e}")

    if all_dataframes:
        # 'outer' join zorgt dat we geen data verliezen als een sensor vaker meet dan een andere
        df_irceline = pd.concat(all_dataframes, axis=1, join='outer').sort_index()
        return df_irceline
    else:
        print("No data could be downloaded.")
        return 0
"""------------------------------------------------------------------------------------"""
def plot_combined(output_directory,df_average, df_irceline):
    """Maakt de grafieken voor Ozon, NO2 en PM10 op basis van de index."""
    ylabels = ['mV','mV','PPM']
    kleuren = ['purple', 'orange']
    indices_ = df_average.columns.get_indexer(targets)
    #fig, axes = plt.subplots(len(indices), 1, figsize=(12, 10), sharex=True)
    # Als er maar 1 grafiek is, zorg dat axes een lijst is
    #if len(indices) == 1: axes = [axes]
    # only plot the first n irceline points
    max_len = len(df_irceline['Ozon_value(mV)'])
    #if len(df_irceline[Ozon_value(mV)])<len(df_average['Ozon_value(mV)']):
    #    max_len = len(df_irceline[Ozon_value(mV)])

    for i, cols in enumerate(indices_):
        # We plotten elke stof in een eigen kleur, data[250:, indices[i]]
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(df_average['TIMESTAMP'][2:max_len],df_average[targets[i]][2:max_len], color=kleuren[0], linewidth=0.7)
        axes[0].set_title(f"Measurements PCB: {df_average.columns[indices_[i]]}")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel(ylabels[i])
        axes[1].plot(df_average['TIMESTAMP'][2:max_len], df_irceline[targets[i]][2:max_len], color=kleuren[1], linewidth=0.7)
        axes[1].set_title(f"Measurements Irceline")#: {df_irceline.columns[indices[i]]}")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel(ylabels[i])
        plt.xlabel("Date & Time")
        plt.tight_layout()
        plt.savefig(f'{output_directory}/{targets[i]}.pdf')
        plt.show()
"""------------------------------------------------------------------------------------"""
def list_files(directory):
    # path = Path(directory)
    # files = []
    # # List all files and directories
    # for entry in path.iterdir():
    #     new_path = directory + entry.name
    #     if entry.is_dir():
    #         new_path = new_path + '/'
    #         files.extend(list_files(new_path))

    # files.extend([f for f in path.iterdir() if f.is_file()])    
    path = Path(directory)
    files = []

    # Controleer of het pad bestaat om crashes te voorkomen
    if not path.exists():
        return files

    for entry in path.iterdir():
        if entry.is_dir():
            # Gebruik het 'entry' object direct voor de recursie
            files.extend(list_files(entry))
        elif entry.is_file():
            # Controleer of het bestand eindigt op .csv (case-insensitive)
            if entry.suffix.lower() == '.csv':
                # Voeg het volledige pad toe als string (of houd het als Path object)
                files.append(entry)
    return files
"""------------------------------------------------------------------------------------"""
def create_output_directories(input_directory, output_directory, input_files):
    # create new output direcotries only if they do not exists 
    output_directory_name = os.path.dirname(output_directory)
    input_directory_name = os.path.dirname(input_directory)

    output_directories = []
    for file in input_files:
        dir = str(file).replace(input_directory_name,output_directory_name)
        # create from the filename itself a directory
        dir = dir.replace(file_extenstion, '')
        # create the directory on disk
        d = Path(dir)
        if d.exists() and d.is_dir():
            #remve the input file
            input_files.remove(file)
            continue  # Skip this file and move to the next
        #creating output directory
        d.mkdir(parents=True, exist_ok=True)
        output_directories.append(dir)

    return (output_directories)
"""------------------------------------------------------------------------------------"""
def store_cleaned_source_files(output_directory, file, data):
    #clean out GPS data, write zeros instead of actual locations
    data_save = data.copy()
    data_save.loc[:,'GPS_LAT'] = 0
    data_save.loc[:,'GPS_LON'] = 0
    data_save.loc[:,'GPS_ALT(m)'] = 0

    outputfile = output_directory + '/' +  file
    print(f"Writing cleaned file to {outputfile}")
    #store this to file
    data_save.to_csv(outputfile,
              sep=',',  # Use commas instead of semicolons
              index=False)
              #encoding='utf-8-sig',  # Ensures special characters show up correctly in Excel
              #columns=['A', 'B'])
"""------------------------------------------------------------------------------------"""
def main():
    # 2. Data inladen
    files = list_files(input_directory)
    print(files)
    output_directories = create_output_directories(input_directory, output_directory, files)
    print(output_directories)

    for i in range(len(files)):
        data = load_sensor_data(files[i])
        store_cleaned_source_files(output_directories[i], files[i].name, data)
        # 3. Als het laden gelukt is, ga verder met plotten
        if data is not None:
            print('Recovering missing time records')
            time_receovered_data = recover_time(data)
            print('Computing averages per hour')
            average_data = average_per_hour(time_receovered_data)
            print('Downloading corresponding timed data from Irceline')
            ircel_data = download_irceline(average_data)
            print('Plotting data')
            plot_polluenten(output_directories[i], time_receovered_data)
            plot_combined(output_directories[i], average_data, ircel_data)

"""------------------------------------------------------------------------------------"""
# Dit zorgt ervoor dat de main() functie alleen draait als je dit script direct uitvoert
if __name__ == "__main__":
    main()