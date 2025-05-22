import pandas as pd

#load the full daily export 
df = pd.read_csv("/mnt/d/DETECT/OUTPUT/raw_export_for_r/raw_daily_data_all_subjects.csv", parse_dates=["date"])

#filter to subjects with known fall labels
df = df[df['label_fall'].notna()]
df = df.sort_values(['subid', 'date'])
                     
#build recurrent intervals
intervals = []

for subid, group in df.groupby('subid'):
    group = group.sort_values('date').reset_index(drop=True) #group by date
    start_day = group.loc[0, 'date']
    
    #get all dates where a fall occurred
    fall_indices = group.index[group['label_fall'] == 1].tolist()
    
    if len(fall_indices) == 0:
        #no fall for this person - skip or add censored interval
        continue
	
    prev_index = 0
    event_num = 0
    
    for fall_idx in fall_indices:
        fall_date = group.loc[fall_idx, 'date']
        tstart = (group.loc[prev_index, 'date'] - start_day).days
        tstop = (fall_date - start_day).days
        
        #build interval
        interval = {
			"id": subid,
			"tstart": tstart,
			"tstop": tstop,
			"status": 1,
			"enum": event_num,
			"sex": group.loc[fall_idx, "sex"] if "sex" in group else None,
			"age": fall_date.year - int(group.loc[fall_idx, "birthyr"]) if pd.notnull(group.loc[fall_idx, "birthyr"]) else None,
			"amyloid": group.loc[fall_idx, "amyloid"] if "amyloid" in group else None,
   
		}     
        
        intervals.append(interval)
        event_num += 1
        prev_index = fall_idx + 1 #next interval start after the current fall
        
    #add final interval (censored)
    if prev_index < len(group):
        tstart = (group.loc[prev_index, 'date'] - start_day).days
        tstop = (group.loc[len(group) - 1, 'date'] - start_day).days
        interval = {
            "id": subid,
            "tstart": tstart,
            "tstop": tstop,
            "status": 0,
            "enum": event_num,
            "sex": group.iloc[-1].get("sex", None),
            "age": group.iloc[-1]["date"].year - int(group.iloc[-1]["birthyr"]) if pd.notnull(group.iloc[-1]["birthyr"]) else None,
            "amyloid": group.iloc[-1].get("amyloid", None),
            "random": group.loc[0, "date"]
        }
        intervals.append(interval)
            
#create final dataframe
interval_df = pd.DataFrame(intervals)
interval_df = interval_df.sort_values(['id', 'tstart'])

#export to CSV for R
interval_df.to_csv("/mnt/d/DETECT/OUTPUT/raw_export_for_r/intervals.csv", index=False)
print("Exported intervals to CSV for R.")               