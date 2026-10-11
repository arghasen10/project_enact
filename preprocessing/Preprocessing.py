from json.decoder import NaN
import re
from statistics import correlation

import requests
import os
import statistics
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
from numpy.ma.core import indices
from pathlib import Path

# Place folders into the input directory for each device containing the CSV files
input_directory = '../SensingFiles/Halle_2'
output_directory = '../GeneratedOutput/'
#Look at https://geo.irceline.be/sos/api/v1/timeseries/ to get updated IDs for the sensors

targets = ['Ozon_value(mV)', 'NO2_value(mV)', 'PM10.0_AVG_PPM', 'PM1.0_AVG_PPM', 'PM2.5_AVG_PPM', 'CO2(ppm)', 'CO_value(mV)', 
        'BME68X_PRES(hPa)', 'BME68X_HUM(%RH)', 'BME68X_TEMP(*C)', 'BME68X_VOC(kOhm)', 'Lux', 'UVA', 'UVB', 'UVC']
UTC_MINIMAL = 1764547200 # corresponds to 2025-12-01-00:00:00

#uccle (Ukkel)
UCCLE_POLLUTANTS = {'Ozon_ref(mV)': '6625', 'NO2_ref(mV)': '6622', 'PM10_ref(PPM)': '6626', 'PM2.5_ref(PPM)': '6627'}
#Sint-Pieters-Leeuw (Halle)
HALLE_POLLUTANTS  = {'Ozon_ref(mV)': '6719', 'NO2_ref(mV)': '6718', 'PM10_ref(PPM)': '6626', 'PM2.5_ref(PPM)': '6627'}
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

def group_files_by_device(directory):
    """Group every CSV by its first subdirectory beneath the input root."""
    input_path = Path(directory).resolve()
    device_files = {}

    for file in list_files(input_path):
        file = Path(file).resolve()
        relative_path = file.relative_to(input_path)

        # Expected structure: SensingFiles/<device>/<file>.csv
        device = relative_path.parts[0] if len(relative_path.parts) > 1 else "Unknown"
        device_files.setdefault(device, []).append(file)

    for files in device_files.values():
        files.sort()

    return device_files
"""------------------------------------------------------------------------------------"""

def get_pollutants(device):
    """Return the correct Irceline station configuration for a device."""
    normalized_device = device.lower()

    if normalized_device.startswith('halle'):
        return HALLE_POLLUTANTS

    if normalized_device.startswith(('ukkel', 'uccle')):
        return UCCLE_POLLUTANTS

    print(f"Warning: Device {device!r} is not recognized. Using Uccle reference station.")
    return UCCLE_POLLUTANTS
"""------------------------------------------------------------------------------------"""

def create_output_directories(input_directory, output_directory, input_files):
    input_path = Path(input_directory).resolve()
    output_path = Path(output_directory).resolve()

    output_directories = []

    for file in input_files:
        file = Path(file).resolve()

        # Get relative path from input root
        relative_path = file.relative_to(input_path)

        # Remove .csv extension and build output dir
        output_dir = output_path / relative_path.with_suffix('')

        output_dir.mkdir(parents=True, exist_ok=True)
        output_directories.append(str(output_dir))

    return output_directories
"""------------------------------------------------------------------------------------"""

def load_sensor_data(bestandsnaam):
    try:
        df = pd.read_csv(bestandsnaam, sep=',', engine='python', index_col=False)
        df.columns = df.columns.str.strip()
        print(f"\n📊 File '{bestandsnaam.name}' contains {df.shape[0]} rows and {df.shape[1]} cols.")

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

def reformat_data(data):
    corrected_date_col = []

    # Remove outliers in O3 and NO2
    data.loc[data['Ozon_value(mV)'] > 100, 'Ozon_value(mV)'] = 0
    data.loc[data['NO2_value(mV)'] > 100, 'NO2_value(mV)'] = 0
    data.loc[data['Ozon_value(mV)'] < 0, 'Ozon_value(mV)'] = 0
    data.loc[data['NO2_value(mV)'] < 0, 'NO2_value(mV)'] = 0

    for raw_date in data['DATE']:
        raw_date = str(raw_date)

        if raw_date.startswith("80/"):
            corrected_date_col.append(raw_date)
            continue

        try:
            part1, part2, part3 = raw_date.split("/")

            # part1 = YY → make it 20YY
            corrected_year = "20" + part1.zfill(2)

            # part3 = 20DD → make it DD
            corrected_day = part3[-2:]

            corrected_date = f"{corrected_year}/{part2}/{corrected_day}"
            corrected_date_col.append(corrected_date)

        except Exception:
            corrected_date_col.append(raw_date)

    data['DATE'] = corrected_date_col
    return data
"""------------------------------------------------------------------------------------"""

def recover_time(output_directory, file, df):
    df['DATE_DT'] = df['DATE'] + ' ' + df['TIME']
    df['GPS_TIMESTAMP'] = pd.to_datetime(df['DATE_DT'], yearfirst=True, errors='coerce')
    # convert to UTC values
    df['UTC'] = df['GPS_TIMESTAMP'].astype('int64') // 10 ** 9
    df['TIME_SEC'] = df['SYS_MILLIS_TIME (ms)']/1000
    df['TIME_SEC_STEP'] = df['TIME_SEC'].diff()
    df.loc[0,'TIME_SEC_STEP']=0
    df['TIME_SEC_STEP'] = df['TIME_SEC_STEP'].astype(int)
    # store_file(output_directory, 'UTC_added', df)

    # Correct negative increments in TIME_SEC_STEP by applying an offset to all subsequent times
    median_inc = statistics.median(df.loc[df['TIME_SEC_STEP'] > 0, 'TIME_SEC_STEP'])
    while True:
        df['TIME_SEC_STEP'] = df['TIME_SEC'].diff()
        df.loc[0, 'TIME_SEC_STEP'] = 0
        neg_indices = numpy.where(df['TIME_SEC_STEP'] < 0)[0]

        if len(neg_indices) == 0:
            print("No negative time increments found, time correction complete.")
            break
        else:
            print(f"{len(neg_indices)} negative time increments found, applying offset correction...")

        idx = neg_indices[0]
        prev_time = df.loc[idx - 1, 'TIME_SEC']
        curr_time = df.loc[idx, 'TIME_SEC']
        offset = prev_time - curr_time + median_inc

        df.loc[idx:, 'TIME_SEC'] += offset
        print(f"Applied offset of {offset:.2f} seconds "f"starting from index {idx}")
    df['TIME_SEC_STEP'] = df['TIME_SEC'].diff()
    df.loc[0, 'TIME_SEC_STEP'] = 0
    df['TIME_SEC_STEP'] = df['TIME_SEC_STEP'].astype(int)
    # store_file(output_directory, 'Time_sec_incremental', df)

    # Check for valid GPS timestamps and perform backward reconstruction of missing UTC values
    utc_valid_indices = numpy.where(df['UTC']>=UTC_MINIMAL)[0]
    if len(utc_valid_indices)>0:
        start_index = utc_valid_indices[0]
        print(f"First valid UTC timestamp found at index {start_index}")
        #if our start index does not equal the zero index we have to recompute the missing times
        if start_index>0:
            for i in range(start_index-1,-1,-1): 
                df.loc[i,'UTC']=df.loc[i+1,'UTC'] - df.loc[i,'TIME_SEC_STEP']
    else:
        print("No valid GPS timestamps found, time is reconstructed using only system time.")

    # print(df['UTC'].describe())
    # print(df['UTC'].nlargest(10))
    # print(df['TIME_SEC_STEP'].nlargest(10))

    # Perform forward reconstruction to correct any negative or spiking UTC steps
    utc_values = df['UTC'].values
    time_sec_steps = df['TIME_SEC_STEP'].values

    for i in range(1, len(utc_values)):
        step = utc_values[i] - utc_values[i-1]
        
        # If the UTC step is negative, missing, or leaps forward by more than 1hr - use time_sec_step, or if that is also invalid - use median_inc
        if pd.isna(step) or step < 1 or step > 3600:
            safe_step = time_sec_steps[i]
            if safe_step < 1 or safe_step > 3600:
                safe_step = median_inc
                
            utc_values[i] = utc_values[i-1] + safe_step
            
    df['UTC'] = utc_values
    df['UTC_STEP'] = df['UTC'].diff().fillna(0).astype('int64')
    
    # Compute final UTC timestamps
    df['UTC_TIMESTAMP'] = pd.to_datetime(df['UTC'], utc=True, unit='s')

    print("Duration from TIME_SEC:",(df['TIME_SEC'].iloc[-1] - df['TIME_SEC'].iloc[0]) / 86400," days")
    print("Duration from UTC:",(df['UTC'].iloc[-1] - df['UTC'].iloc[0]) / 86400," days")
    print(f"Duration is {df.loc[0, 'UTC_TIMESTAMP']} to {df.loc[len(df)-1, 'UTC_TIMESTAMP']}")

    df.drop(columns=["UTC_STEP", "TIME_SEC_STEP", "TIME_SEC", "GPS_TIMESTAMP", "DATE_DT", 
                    "NO2_measured(mV)", "NO2_vref(mV)", "Ozon_measured(mV)", "Ozon_vref(mV)", "CO_vref(mV)", "CO_measured(mV)", 
                    "ACC_X(m/s2)", "ACC_Y(m/s2)", "ACC_Z(m/s2)", "SYS_MILLIS_TIME (ms)", "TIME", "DATE", 
                    "GPS_NSATS", "GPS_ALT(m)", "GPS_LON", "GPS_LAT", "GPS_FIX", "Unnamed: 0", "MIC(dBSPL)", 
                    "PM_N_NAN", "PM_N_OBSTRUCTIONS", "PM_N_SUCCESS", "BATTERY_VOLT", "BATTERY_PERCENT"], 
                    axis=1, inplace=True, errors="ignore")

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
    averaged['TIMESTAMP'] = averaged['TIMESTAMP'].astype('object')
    averaged['T_IRCELINE'] = averaged['T_IRCELINE'].astype('object')
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

def download_irceline(df, pollutants):
    all_dataframes = []
    timespan = obtain_start_stop_time(df) # example"2025-12-31T00:00:00/2026-01-05T09:00:00"
    base_url = "https://geo.irceline.be/sos/api/v1/timeseries/"
    headers = {'User-Agent': 'Mozilla/5.0'}

    print(f"Downloading data between timestamps {timespan}")

    for name, ts_id in pollutants.items():
        print(f"Obtaining {name} (ID: {ts_id}) from Irceline\n")
        url = f"{base_url}{ts_id}/getData"

        try:
            response = requests.get(url, params={"timespan": timespan}, headers=headers, verify=False)
            response.raise_for_status()

            values = response.json().get('values', [])
            if values:
                temp_df = pd.DataFrame(values)
                temp_df.columns = ['timestamp', name]
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms')

                # Set timestamp as index to make merging easier
                temp_df.set_index('timestamp', inplace=True)
                all_dataframes.append(temp_df)
            else:
                print(f"oops: no data for {name} could be downloaded")

        except Exception as e:
            print(f"Error with {name}: {e}")

    if all_dataframes:
        # 'outer' join ensures that we do not lose data if a sensor measures more often than another
        df_irceline = pd.concat(all_dataframes, axis=1, join='outer').sort_index().reset_index()
        result_df = pd.concat([df, df_irceline], axis=1)
        return result_df
    else:
        print("No data could be downloaded.")
        return 0
"""------------------------------------------------------------------------------------"""

def store_file_csv(output_directory, file_name, data):
    # Splits 'data.csv' into ('data', '.csv') or 'custom_name' into ('custom_name', '')
    root, ext = os.path.splitext(file_name)
    csv_file = f"{root}.csv"
    outputfile = os.path.join(output_directory, csv_file)

    print(f"Writing file to {outputfile}")
    data.to_csv(outputfile, sep=',', index=True)
"""------------------------------------------------------------------------------------"""

def plot_data(df, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure datetime format and sort
    df['UTC_TIMESTAMP'] = pd.to_datetime(df['UTC_TIMESTAMP'])
    df = df.sort_values('UTC_TIMESTAMP')
    
    # Define the groupings and their respective columns
    plot_groups = {
        'Meteo_data': ['BME68X_TEMP(*C)', 'BME68X_HUM(%RH)', 'BME68X_PRES(hPa)'],
        'PM_values': ['PM1.0_AVG_PPM', 'PM2.5_AVG_PPM', 'PM10.0_AVG_PPM'],
        'Light_radiation': ['Lux', 'UVA', 'UVB', 'UVC'],
        'Gases': ['Ozon_value(mV)', 'NO2_value(mV)', 'CO2(ppm)', 'CO_value(mV)', 'BME68X_VOC(kOhm)']
    }
    
    for filename, columns in plot_groups.items():
        num_plots = len(columns)
        fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(12, 3 * num_plots), sharex=True)
        
        if num_plots == 1:
            axes = [axes]
            
        for ax, col in zip(axes, columns):
            # Base logic for standard plots
            ax.plot(df['UTC_TIMESTAMP'], df[col], label=col, color='tab:blue')
            ax.set_ylabel(col)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Special case for Gases: Add Temperature to a secondary y-axis
            if filename == 'Gases':
                ax_temp = ax.twinx()
                ax_temp.plot(df['UTC_TIMESTAMP'], df['BME68X_TEMP(*C)'], label='Temp (*C)', color='tab:red', alpha=0.5)
                ax_temp.set_ylabel('Temp (*C)', color='tab:red')
                ax_temp.tick_params(axis='y', labelcolor='tab:red')
                
                # Merge legends for the two axes
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_temp.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax.legend(loc='upper right')
            
        axes[-1].set_xlabel('UTC Timestamp')
        fig.suptitle(f'{filename.replace("_", " ")} Over Time', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}.pdf'))
        plt.close(fig)
"""------------------------------------------------------------------------------------"""

def plot_avg_data(df, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure datetime format and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    x = df['timestamp']
    temp = df['BME68X_TEMP(*C)']
    
    # -----------------------------------------
    # 1. Gases_ref (O3 + Temp, NO2 + Temp)
    # -----------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: O3
    ax1 = axes[0]
    ax1.plot(x, df['Ozon_value(mV)'], label='Ozon_value(mV)', color='tab:blue')
    ax1.plot(x, df['Ozon_ref(mV)'], label='Ozon_ref(mV)', color='tab:blue', linestyle=':')
    ax1.set_ylabel('Ozone (mV)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Secondary y-axis for Temperature
    ax1_temp = ax1.twinx()
    ax1_temp.plot(x, temp, label='Temp (*C)', color='green', alpha=0.4)
    ax1_temp.set_ylabel('Temperature (*C)', color='green')
    
    # Combine legends for ax1
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_temp.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Subplot 2: NO2
    ax2 = axes[1]
    ax2.plot(x, df['NO2_value(mV)'], label='NO2_value(mV)', color='tab:purple')
    ax2.plot(x, df['NO2_ref(mV)'], label='NO2_ref(mV)', color='tab:purple', linestyle=':')
    ax2.set_ylabel('NO2 (mV)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Secondary y-axis for Temperature
    ax2_temp = ax2.twinx()
    ax2_temp.plot(x, temp, label='Temp (*C)', color='green', alpha=0.4)
    ax2_temp.set_ylabel('Temperature (*C)', color='green')
    ax2.set_xlabel('Timestamp')
    
    # Combine legends for ax2
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_temp.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

    fig.suptitle('Gases Reference Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gases_ref.pdf'))
    plt.close(fig)

    # -----------------------------------------
    # 2. PM_ref (PM2.5, PM10)
    # -----------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: PM2.5
    axes[0].plot(x, df['PM2.5_AVG_PPM'], label='PM2.5_AVG_PPM', color='tab:blue')
    axes[0].plot(x, df['PM2.5_ref(PPM)'], label='PM2.5_ref(PPM)', color='tab:blue', linestyle=':')
    axes[0].set_ylabel('PM 2.5 (PPM)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right')

    # Subplot 2: PM10
    axes[1].plot(x, df['PM10.0_AVG_PPM'], label='PM10.0_AVG_PPM', color='tab:green')
    axes[1].plot(x, df['PM10_ref(PPM)'], label='PM10_ref(PPM)', color='tab:green', linestyle=':')
    axes[1].set_ylabel('PM 10 (PPM)')
    axes[1].set_xlabel('Timestamp')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')

    fig.suptitle('Particulate Matter Reference Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'PM_ref.pdf'))
    plt.close(fig)
"""------------------------------------------------------------------------------------"""

def main():
    input_path = Path(input_directory).resolve()
    output_path = Path(output_directory).resolve()
    device_files = group_files_by_device(input_path)

    if not device_files:
        print(f"No CSV files found under {input_path}")
        return

    print(f"Found {sum(len(files) for files in device_files.values())} CSV files across {len(device_files)} devices.")

    for device, files in sorted(device_files.items()):
        print(f"\n{'=' * 70}")
        print(f"PROCESSING DEVICE: {device} ({len(files)} files)")
        print(f"{'=' * 70}")

        device_output_directory = output_path / device
        device_output_directory.mkdir(parents=True, exist_ok=True)

        device_pollutants = get_pollutants(device)
        all_data = []
        all_average_data = []

        for file in files:
            relative_path = file.relative_to(input_path)

            # Remove the device directory from the relative path.
            if len(relative_path.parts) > 1:
                relative_file_path = Path(*relative_path.parts[1:])
            else:
                relative_file_path = Path(file.name)

            file_output_directory = device_output_directory / relative_file_path.with_suffix('')
            file_output_directory.mkdir(parents=True, exist_ok=True)

            print(f"\nProcessing: {file}")

            data = load_sensor_data(file)
            if data is None:
                print(f"Skipping {file.name}: loading failed.")
                continue

            print("Reformatting date column and removing outliers")
            data = reformat_data(data)

            print("Recovering missing time records")
            time_recovered_data = recover_time(file_output_directory, file.name, data)
            store_file_csv(file_output_directory, 'time_recovered', time_recovered_data)
            all_data.append(time_recovered_data)

            print("Computing averages per hour")
            print(f"Length of time_recovered_data: {len(time_recovered_data)}")
            average_data = average_per_hour(time_recovered_data)
            print(f"Length of average_data: {len(average_data)}")

            print(f"Downloading Irceline reference data for {device}")
            ircel_data = download_irceline(average_data, device_pollutants)

            if isinstance(ircel_data, pd.DataFrame):
                store_file_csv(file_output_directory, 'average_data', ircel_data)
                all_average_data.append(ircel_data)
                plot_avg_data(ircel_data, file_output_directory)
            else:
                print(f"Warning: No valid Irceline data returned for {file.name}.")

            print("Plotting sensor data")
            plot_data(time_recovered_data, file_output_directory)
            print(f"\n{'-' * 70}")

        # Combine raw time-recovered data separately for each device.
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)

            if 'UTC_TIMESTAMP' in combined_df.columns:
                combined_df['UTC_TIMESTAMP'] = pd.to_datetime(combined_df['UTC_TIMESTAMP'], utc=True, errors='coerce')
                combined_df = combined_df.sort_values('UTC_TIMESTAMP').reset_index(drop=True)

            store_file_csv(device_output_directory, 'combined_data_all_files', combined_df)

            if len(all_data) > 1:
                print(f"Plotting combined sensor data for {device}")
                plot_data(combined_df, device_output_directory)
        else:
            print(f"No sensor data was successfully processed for {device}.")

        # Combine hourly data and references separately for each device.
        if all_average_data:
            average_df = pd.concat(all_average_data, ignore_index=True)

            if 'timestamp' in average_df.columns:
                average_df['timestamp'] = pd.to_datetime(average_df['timestamp'], errors='coerce')
                average_df = average_df.sort_values('timestamp').reset_index(drop=True)

            store_file_csv(device_output_directory, 'average_data_all_files', average_df)

            if len(all_average_data) > 1:
                print(f"Plotting combined hourly/reference data for {device}")
                plot_avg_data(average_df, device_output_directory)
        else:
            print(f"No hourly/reference data was successfully processed for {device}.")

        print(f"Completed device: {device}")
    print(f"\n{'=' * 70}\nAll devices processed. Output saved to {output_directory}\n{'=' * 70}")
"""------------------------------------------------------------------------------------"""

# Dit zorgt ervoor dat de main() functie alleen draait als je dit script direct uitvoert
if __name__ == "__main__":
    main()