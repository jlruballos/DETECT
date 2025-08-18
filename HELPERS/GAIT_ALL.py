import pandas as pd
import os
import glob

program_name = 'GAIT'

#------ Set up paths ------
base_path = '/mnt/d/DETECT'
output_dir = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_dir, exist_ok=True)

NYCE_PATH = os.path.join(base_path, 'DETECT_DATA_080825', 'Sensor_Data', 'NYCE_Data')

# Initialize empty lists to store all results
all_sensor_data = []
all_summary_data = []

# Get list of all CSV files in the folder
csv_files = glob.glob(os.path.join(NYCE_PATH, 'NYCE_Area_Data_DETECT_*.csv'))

# Process each CSV file
for csv_file in csv_files:
    try:
        # Skip files that already have 'GAIT' in their name
        if 'GAIT' in csv_file:
            continue
            
        print(f"Processing {os.path.basename(csv_file)}...")
        
        # Load data
        line_sensor_data = pd.read_csv(csv_file)
        
        # Filter by areaid 63
        line_sensor_data = line_sensor_data[line_sensor_data['areaid'] == 63]
        
        # If no data after filtering, skip to next file
        if len(line_sensor_data) == 0:
            print(f"No area 63 data in {os.path.basename(csv_file)}, skipping...")
            continue
        
        # Convert 'stamp' to datetime
        line_sensor_data['stamp'] = pd.to_datetime(line_sensor_data['stamp'])
        
        # Reset index to ensure continuous integer-based indexing
        line_sensor_data = line_sensor_data.reset_index(drop=True)
        
        # Identify sequences where 'event' is 1 four times in a row
        line_sensor_data['walking'] = line_sensor_data['event'].rolling(window=4).sum()
        
        # Find valid indices where 'walking' is 4
        valid_indices = line_sensor_data[line_sensor_data['walking'] == 4].index
        
        # List to store valid walking sequences
        valid_walking_indices = []
        
        for idx in valid_indices:
            start_idx = idx - (4 - 1)  # First index in the sequence
            if start_idx >= 0 and start_idx < len(line_sensor_data):
                unique_itemids = line_sensor_data.loc[start_idx:idx, 'itemid'].nunique()
                
                # Check if all four itemids are unique
                if unique_itemids == 4:
                    valid_walking_indices.append(idx)
        
        # Process valid walking sequences
        if valid_walking_indices:
            for idx in valid_walking_indices:
                start_idx = idx - (4 - 1)  # First index in the sequence
                homeid = line_sensor_data.iloc[start_idx]["homeid"]
                start_time = line_sensor_data.iloc[start_idx]["stamp"]
                end_time = line_sensor_data.iloc[idx]["stamp"]
                duration = (end_time - start_time).total_seconds()
                #gait_speed = 6/duration  # divided by 6 feet distance between sensors
                gait_speed = 182.88/duration  # divided by 182.88 cm (6 feet) distance between sensors
                
                # Add the filename as a source column
                all_summary_data.append({
                    "homeid": homeid, 
                    "start_time": start_time, 
                    "duration": duration, 
                    "gait_speed": gait_speed,
                    "source_file": os.path.basename(csv_file)
                })
                
                # Append homeid to results
                line_sensor_data.loc[start_idx:idx, 'homeid'] = line_sensor_data.loc[start_idx, 'homeid']
                # Append calculated values to DataFrame
                line_sensor_data.loc[start_idx:idx, 'duration'] = duration
                line_sensor_data.loc[start_idx:idx, 'GAIT'] = gait_speed
                line_sensor_data.loc[start_idx:idx, 'source_file'] = os.path.basename(csv_file)
        
        # Append the processed data to our main list
        all_sensor_data.append(line_sensor_data)
        
        print(f"Completed processing {os.path.basename(csv_file)}")
        
    except Exception as e:
        print(f"Error processing {os.path.basename(csv_file)}: {str(e)}")
        continue

# Combine all processed data
if all_sensor_data:
    combined_sensor_data = pd.concat(all_sensor_data, ignore_index=True)
    combined_summary_data = pd.DataFrame(all_summary_data)
    
    # Export combined data to CSV
    combined_path = os.path.join(output_dir, 'COMBINED_NYCE_Area_Data_DETECT_GAIT.csv')
    summary_path = os.path.join(output_dir, 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')

    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path),  exist_ok=True)

    combined_sensor_data.to_csv(combined_path, index=False)
    combined_summary_data.to_csv(summary_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Total records in main file: {len(combined_sensor_data)}")
    print(f"Total records in summary file: {len(combined_summary_data)}")
else:
    print("No data was processed!")