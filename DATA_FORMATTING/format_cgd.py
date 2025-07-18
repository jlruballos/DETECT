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
    'avghr', 'avgrr', 'Night_Bathroom_Visits', 'Night_Kitchen_Visits',
  ]
    
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
        
        # Compute interval length
        interval_length = (tstop - tstart) + 1

		# Total counts of hospital visits and accidents in this interval
        interval["hospital_visit_total"] = int((interval_data["label_hospital"] == 1).sum())
        interval["accident_total"] = int((interval_data["label_accident"] == 1).sum())
        interval["label_medication"] = int((interval_data["label_medication"] == 1).sum())
        interval["label_mood_lonely"] = int((interval_data["label_mood_lonely"] == 1).sum())
        interval["label_mood_blue"] = int((interval_data["label_mood_blue"] == 1).sum())

		# 7-day window (lookback) — before fall or last 7 days of interval
        if interval["status"] == 1:  # fall interval
            lookback_data = group.loc[max(prev_index, fall_idx - 7):fall_idx - 1]
        else:  # censored interval
            lookback_data = interval_data.tail(7)
        
        interval["hospital_visit_last7"] = int((lookback_data["label_hospital"] == 1).sum())
        interval["accident_last7"] = int((lookback_data["label_accident"] == 1).sum())
        interval["medication_last7"] = int((lookback_data["label_medication"] == 1).sum())
        interval["label_mood_lonely_last7"] = int((lookback_data["label_mood_lonely"] == 1).sum())
        interval["label_mood_blue_last7"] = int((lookback_data["label_mood_blue"] == 1).sum())

		# Delta = total - last7
        interval["hospital_visit_delta"] = interval["hospital_visit_total"] - interval["hospital_visit_last7"]
        interval["accident_delta"] = interval["accident_total"] - interval["accident_last7"]
        interval["medication_delta"] = interval["label_medication"] - interval["medication_last7"]
        interval["label_mood_lonely_delta"] = interval["label_mood_lonely"] - interval["label_mood_lonely_last7"]
        interval["label_mood_blue_delta"] = interval["label_mood_blue"] - interval["label_mood_blue_last7"]

		# Mean: proportion of days with hospital visits or accidents
        interval["hospital_visit_interval_mean"] = (
			interval["hospital_visit_total"] / interval_length if interval_length > 0 else 0
		)
        interval["accident_interval_mean"] = (
			interval["accident_total"] / interval_length if interval_length > 0 else 0
		)
        interval["medication_interval_mean"] = (
			interval["label_medication"] / interval_length if interval_length > 0 else 0
		)
        interval["label_mood_lonely_interval_mean"] = (
			interval["label_mood_lonely"] / interval_length if interval_length > 0 else 0
		)
        interval["label_mood_blue_interval_mean"] = (
			interval["label_mood_blue"] / interval_length if interval_length > 0 else 0
		)

        #add demographic features
        for col in demo_cols:
            interval[col] = group.loc[fall_idx, col] if col in group.columns else None
            
        # Ensure all output keys are always present
		# Ensure all output keys are always present
        interval["hospital_visit_total"] = interval.get("hospital_visit_total", 0)
        interval["hospital_visit_last7"] = interval.get("hospital_visit_last7", 0)
        interval["hospital_visit_delta"] = interval.get("hospital_visit_delta", 0)
        interval["hospital_visit_interval_mean"] = interval.get("hospital_visit_interval_mean", 0)
        interval["accident_total"] = interval.get("accident_total", 0)
        interval["accident_last7"] = interval.get("accident_last7", 0)
        interval["accident_delta"] = interval.get("accident_delta", 0)
        interval["accident_interval_mean"] = interval.get("accident_interval_mean", 0)
        interval["medication_total"] = interval.get("medication_total", 0)
        interval["medication_last7"] = interval.get("medication_last7", 0)
        interval["medication_delta"] = interval.get("medication_delta", 0)
        interval["medication_interval_mean"] = interval.get("medication_interval_mean", 0)
        interval["label_mood_lonely_total"] = interval.get("label_mood_lonely", 0)
        interval["label_mood_lonely_last7"] = interval.get("label_mood_lonely_last7", 0)
        interval["label_mood_lonely_delta"] = interval.get("label_mood_lonely_delta", 0)
        interval["label_mood_lonely_interval_mean"] = interval.get("label_mood_lonely_interval_mean", 0)
        interval["label_mood_blue_total"] = interval.get("label_mood_blue", 0)
        interval["label_mood_blue_last7"] = interval.get("label_mood_blue_last7", 0)
        interval["label_mood_blue_delta"] = interval.get("label_mood_blue_delta", 0)
        interval["label_mood_blue_interval_mean"] = interval.get("label_mood_blue_interval_mean", 0)
    
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
        
        # Compute interval length
        interval_length = (tstop - tstart) + 1

		# Total counts of hospital visits and accidents in this interval
        interval["hospital_visit_total"] = int((interval_data["label_hospital"] == 1).sum())
        interval["accident_total"] = int((interval_data["label_accident"] == 1).sum())
        interval["medication_total"] = int((interval_data["label_medication"] == 1).sum())
        interval["label_mood_lonely"] = int((interval_data["label_mood_lonely"] == 1).sum())
        interval["label_mood_blue"] = int((interval_data["label_mood_blue"] == 1).sum())

		# 7-day window (lookback) — before fall or last 7 days of interval
        if interval["status"] == 1:  # fall interval
            lookback_data = group.loc[max(prev_index, fall_idx - 7):fall_idx - 1]
        else:  # censored interval
            lookback_data = interval_data.tail(7)
        
        interval["hospital_visit_last7"] = int((lookback_data["label_hospital"] == 1).sum())
        interval["accident_last7"] = int((lookback_data["label_accident"] == 1).sum())
        interval["medication_last7"] = int((lookback_data["label_medication"] == 1).sum())
        interval["label_mood_lonely_last7"] = int((lookback_data["label_mood_lonely"] == 1).sum())
        interval["label_mood_blue_last7"] = int((lookback_data["label_mood_blue"] == 1).sum())

		# Delta = total - last7
        interval["hospital_visit_delta"] = interval["hospital_visit_total"] - interval["hospital_visit_last7"]
        interval["accident_delta"] = interval["accident_total"] - interval["accident_last7"]
        interval["medication_delta"] = interval["medication_total"] - interval["medication_last7"]
        interval["label_mood_lonely_delta"] = interval["label_mood_lonely"] - interval["label_mood_lonely_last7"]
        interval["label_mood_blue_delta"] = interval["label_mood_blue"] - interval["label_mood_blue_last7"]

		# Mean: proportion of days with hospital visits or accidents
        interval["hospital_visit_interval_mean"] = (
			interval["hospital_visit_total"] / interval_length if interval_length > 0 else 0
		)
        interval["accident_interval_mean"] = (
			interval["accident_total"] / interval_length if interval_length > 0 else 0
		)
        interval["medication_interval_mean"] = (
			interval["medication_total"] / interval_length if interval_length > 0 else 0
		)
        interval["label_mood_lonely_interval_mean"] = (
			interval["label_mood_lonely"] / interval_length if interval_length > 0 else 0
		)
        interval["label_mood_blue_interval_mean"] = (
			interval["label_mood_blue"] / interval_length if interval_length > 0 else 0
		)

         #add demographic features
        for col in demo_cols:
            #interval[col] = group.loc[fall_idx, col] if col in group.columns else None
            interval[col] = group.iloc[-1][col] if col in group.columns else None
        
		# Ensure all output keys are always present
        interval["hospital_visit_total"] = interval.get("hospital_visit_total", 0)
        interval["hospital_visit_last7"] = interval.get("hospital_visit_last7", 0)
        interval["hospital_visit_delta"] = interval.get("hospital_visit_delta", 0)
        interval["hospital_visit_interval_mean"] = interval.get("hospital_visit_interval_mean", 0)
        interval["accident_total"] = interval.get("accident_total", 0)
        interval["accident_last7"] = interval.get("accident_last7", 0)
        interval["accident_delta"] = interval.get("accident_delta", 0)
        interval["accident_interval_mean"] = interval.get("accident_interval_mean", 0)
        interval["medication_total"] = interval.get("medication_total", 0)
        interval["medication_last7"] = interval.get("medication_last7", 0)
        interval["medication_delta"] = interval.get("medication_delta", 0)
        interval["medication_interval_mean"] = interval.get("medication_interval_mean", 0)
        interval["label_mood_lonely_total"] = interval.get("label_mood_lonely", 0)
        interval["label_mood_lonely_last7"] = interval.get("label_mood_lonely_last7", 0)
        interval["label_mood_lonely_delta"] = interval.get("label_mood_lonely_delta", 0)
        interval["label_mood_lonely_interval_mean"] = interval.get("label_mood_lonely_interval_mean", 0)
        interval["label_mood_blue_total"] = interval.get("label_mood_blue", 0)
        interval["label_mood_blue_last7"] = interval.get("label_mood_blue_last7", 0)
        interval["label_mood_blue_delta"] = interval.get("label_mood_blue_delta", 0)
        interval["label_mood_blue_interval_mean"] = interval.get("label_mood_blue_interval_mean", 0)

        interval.update(feature_summary)
        intervals.append(interval)
            
#create final dataframe
interval_df = pd.DataFrame(intervals)
interval_df = interval_df.sort_values(['id', 'tstart'])

#export to CSV for R
interval_df.to_csv("/mnt/d/DETECT/OUTPUT/raw_export_for_r/intervals.csv", index=False)
print("Exported intervals to CSV for R.")               