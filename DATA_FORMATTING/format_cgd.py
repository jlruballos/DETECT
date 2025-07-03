import pandas as pd

#load the full daily export 
#df = pd.read_csv("/mnt/d/DETECT/OUTPUT/raw_export_for_r/raw_daily_data_all_subjects.csv", parse_dates=["date"])

df = pd.read_csv("/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_mean.csv", parse_dates=["date"])


#filter to subjects with known fall labels
df = df[df['label_fall'].notna()] #can change to label_hospital if needed
df = df.sort_values(['subid', 'date'])
                     
#build recurrent intervals
intervals = []

for subid, group in df.groupby('subid'):
    group = group.sort_values('date').reset_index(drop=True) #group by date
    start_day = group.loc[0, 'date']
    
    #compute subject-level means for delta features
    feature_cols = ['steps', 'gait_speed',  'awakenings', 'bedexitcount', 'end_sleep_time', 
    'inbed_time', 'outbed_time', 'sleepscore', 
    'durationinsleep', 'durationawake', 'waso', 
    'hrvscore', 'start_sleep_time', 'time_to_sleep', 
    'time_in_bed_after_sleep', 'tossnturncount', 
      'maxhr', 
    'avghr', 'avgrr']
    
    #include additional features for interval summary
    demo_cols = ['birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat','moca_avg', 'cogstat']
    
    subject_means = group[feature_cols].mean() 
    
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
        
        #get the daily data for the interval including the fall
        #features from this section will be used to build the interval
        interval_data = group.loc[prev_index:fall_idx]
        
        feature_summary = {}
        
        for feat in feature_cols:
             #actual daily values in this interval
            vals = pd.to_numeric(interval_data[feat], errors='coerce')
            #interval mean
            interval_mean = vals.mean()
            #cumultive subject data up to but not including the fall
            cumulative_vals = pd.to_numeric(group.loc[prev_index:fall_idx - 1, feat], errors='coerce')
			#mean for this subject
            cumulative_mean = cumulative_vals.mean()
            
            # 7-day moving average before the fall
            lookback_vals = cumulative_vals[-7:] #last 7 days
            #rolling 7-day average computed over these values
            ma7_last = lookback_vals.mean() if not lookback_vals.empty else None
            #delta from feature mean minus the last 7-day average
            delta_to_ma7 = (vals.mean() - ma7_last) if ma7_last is not None else None

            
            #mean within the interval (prev fall to current fall)
            feature_summary[f"{feat}_interval_mean"] = interval_mean
            #7-day average leading up to the fall
            feature_summary[f"{feat}_ma7_last"] = ma7_last
            #deaviation from the subject's total mean minus the lst 7-day average
            feature_summary[f"{feat}_delta"] = delta_to_ma7
        
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
        
        #is there a hospital visit in this interval?
        has_hospital_visit = int((interval_data['label_hospital'] == 1).any())
        interval["hospital_visit"] = has_hospital_visit
        
        #add demographic features
        for col in demo_cols:
            interval[col] = group.loc[fall_idx, col] if col in group.columns else None
    
        interval.update(feature_summary)
        
        intervals.append(interval)
        event_num += 1
        prev_index = fall_idx + 1 #next interval start after the current fall
        
    #add final interval (censored)
    if prev_index < len(group):
        tstart = (group.loc[prev_index, 'date'] - start_day).days
        tstop = (group.loc[len(group) - 1, 'date'] - start_day).days
        interval_data = group.loc[prev_index:]
        
        fall_date = group.loc[len(group) - 1, 'date']

        feature_summary = {}
        for feat in feature_cols:
             #actual daily values in this interval
            vals = pd.to_numeric(interval_data[feat], errors='coerce')
            #interval mean
            interval_mean = vals.mean()
            #cumultive subject data up to but not including the fall
            cumulative_vals = pd.to_numeric(group.loc[:prev_index - 1, feat], errors='coerce')
			#mean for this subject
            cumulative_mean = cumulative_vals.mean()
            
            # 7-day moving average before the fall
            lookback_vals = cumulative_vals[-7:] #last 7 days
            #rolling 7-day average computed over these values
            ma7_last = lookback_vals.mean() if not lookback_vals.empty else None
            #delta from feature mean minus the last 7-day average
            delta_to_ma7 = (vals.mean() - ma7_last) if ma7_last is not None else None
            
            #mean within the interval (prev fall to current fall)
            feature_summary[f"{feat}_interval_mean"] = interval_mean
            #7-day average leading up to the fall
            feature_summary[f"{feat}_ma7_last"] = ma7_last
            #deaviation from the subject's total mean minus the lst 7-day average
            feature_summary[f"{feat}_delta"] = delta_to_ma7
            
        interval = {
            "id": subid,
            "tstart": tstart,
            "tstop": tstop,
            "status": 0,
            "enum": event_num,
            "sex": group.iloc[-1].get("sex", None),
            "age": group.iloc[-1]["date"].year - int(group.iloc[-1]["birthyr"]) if pd.notnull(group.iloc[-1]["birthyr"]) else None,
            "amyloid": group.iloc[-1].get("amyloid", None),
        }
        
         #is there a hospital visit in this interval?
        has_hospital_visit = int((interval_data['label_hospital'] == 1).any())
        interval["hospital_visit"] = has_hospital_visit
        
         #add demographic features
        for col in demo_cols:
            #interval[col] = group.loc[fall_idx, col] if col in group.columns else None
            interval[col] = group.iloc[-1][col] if col in group.columns else None
  
        interval.update(feature_summary)
        intervals.append(interval)
            
#create final dataframe
interval_df = pd.DataFrame(intervals)
interval_df = interval_df.sort_values(['id', 'tstart'])

#export to CSV for R
interval_df.to_csv("/mnt/d/DETECT/OUTPUT/raw_export_for_r/intervals.csv", index=False)
print("Exported intervals to CSV for R.")               