import pandas as pd
import numpy as np
import os

# Create new output directory
output_dir = "/mnt/d/DETECT/OUTPUT/survival_intervals"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Load labeled daily data
df = pd.read_csv("/mnt/d/DETECT/OUTPUT/impute_data/labeled_daily_data_mean_imputed.csv", parse_dates=["date"])

label = 'label_accident' # Change to 'label_hospital' 'label_mood_lonely' 'label_mood_blue', 'label_accident', 'label_medication'
label_cols = ['label_hospital', 'label_accident', 'label_medication', 'label_mood_lonely', 'label_mood_blue', 'label_fall']

# KEY CHANGE 1: Remove the filter - keep ALL subjects
# NEW: Keep all subjects, fill NaN labels with 0
df[label] = df[label].fillna(0)
for col in label_cols:
    df[col] = df[col].fillna(0)

df = df.sort_values(['subid', 'date'])

# Define feature and demographic columns
feature_cols = [
    'steps', 'gait_speed', 'awakenings', 'bedexitcount', 'end_sleep_time',
    'inbed_time', 'outbed_time', 'sleepscore', 'durationinsleep', 'durationawake',
    'waso', 'hrvscore', 'start_sleep_time', 
    'time_in_bed_after_sleep', 'tossnturncount', 'maxhr', 'avghr', 'avgrr', 
    'Night_Bathroom_Visits', 'Night_Kitchen_Visits', 
]

demo_cols = ['birthyr', 'sex', 'hispanic', 'race_group', 'educ_group', 'livsitua', 'independ', 'residenc', 
             'alzdis', 'maristat','moca_category', 'cogstat', 'age_bucket']

intervals = []

for subid, group in df.groupby('subid'):
    group = group.sort_values('date').reset_index(drop=True)
    start_day = group.loc[0, 'date']
    
    # Indices where an event occurred
    event_indices = group.index[group[label] == 1].tolist()
    
    # KEY CHANGE 2: Handle both cases - with events and without events
    if len(event_indices) == 0:
        print(f"Subject {subid}: No events - creating single censored interval")
        # CREATE SINGLE LONG CENSORED INTERVAL
        tstart = 0
        tstop = (group.loc[len(group) - 1, 'date'] - start_day).days
        interval_data = group.copy()
        
        # ROBUST FEATURE CALCULATION FOR NO-EVENT PARTICIPANTS
        feature_summary = {}
        for feat in feature_cols:
            # ROBUST INTERVAL MEAN: Use all available values (ignore NaN)
            vals = pd.to_numeric(interval_data[feat], errors='coerce')
            interval_mean = vals.mean()  # pandas.mean() automatically ignores NaN
            
            # ROBUST MA7 LAST: Use last available values (not necessarily 7 days)
            available_vals = vals.dropna()  # Remove NaN values first
            if len(available_vals) >= 1:
                ma7_last = available_vals.tail(min(7, len(available_vals))).mean()
            else:
                ma7_last = np.nan
            
            # DELTA: For no-event subjects, no meaningful delta
            feature_summary[f"{feat}_interval_mean"] = interval_mean
            feature_summary[f"{feat}_ma7_last"] = ma7_last
            feature_summary[f"{feat}_delta"] = 0  # No delta since no event occurred
        
        interval = {
            "id": subid,
            "tstart": tstart,
            "tstop": tstop,
            "status": 0,  # Censored - no event occurred
            "enum": 0,    # First (and only) interval
            # Use demographics from FIRST DAY of interval (start of follow-up)
            "sex": group.iloc[0].get("sex", None),
            "age": group.iloc[0]["date"].year - int(group.iloc[0]["birthyr"]) if pd.notnull(group.iloc[0]["birthyr"]) else None,
            "amyloid": group.iloc[0].get("amyloid", None),
        }
        
        # Compute interval length
        interval_length = (tstop - tstart) + 1
        
        # Calculate totals for all other labels
        for col in label_cols:
            if col != label:
                interval[col + "_total"] = int((interval_data[col] == 1).sum())
                # For single long interval, last 7 days = last 7 days of data
                lookback_data = interval_data.tail(7)
                interval[col + "_last7"] = int((lookback_data[col] == 1).sum())
                interval[col + "_delta"] = interval[col + "_total"] - interval[col + "_last7"]
                interval[col + "_interval_mean"] = interval[col + "_total"] / interval_length if interval_length > 0 else 0
        
        # Add demographic data from FIRST DAY of interval
        for col in demo_cols:
            interval[col] = group.iloc[0][col] if col in group.columns else None
        
        interval.update(feature_summary)
        intervals.append(interval)
        
    else:
        print(f"Subject {subid}: {len(event_indices)} events - creating multiple intervals")
        # ROBUST LOGIC FOR SUBJECTS WITH EVENTS
        prev_index = 0
        event_num = 0
        
        for event_idx in event_indices:
            event_date = group.loc[event_idx, 'date']
            tstart = (group.loc[prev_index, 'date'] - start_day).days
            tstop = (event_date - start_day).days
            interval_data = group.loc[prev_index:event_idx]
            
            # ROBUST FEATURE CALCULATION FOR EVENT INTERVALS
            feature_summary = {}
            for feat in feature_cols:
                # ROBUST INTERVAL MEAN: Use all available values in interval
                interval_vals = pd.to_numeric(interval_data[feat], errors='coerce')
                interval_mean = interval_vals.mean()  # Ignores NaN automatically
                
                # ROBUST MA7 LAST: Use last available values before this interval  
                if prev_index > 0:
                    # Get all data before this interval starts
                    historical_data = group.loc[0:prev_index-1]
                    historical_vals = pd.to_numeric(historical_data[feat], errors='coerce').dropna()
                    
                    # Take last available values (up to 7, but could be fewer)
                    if len(historical_vals) >= 1:
                        ma7_last = historical_vals.tail(min(7, len(historical_vals))).mean()
                    else:
                        ma7_last = np.nan
                else:
                    # First interval - no historical data
                    ma7_last = interval_mean  # Use current interval as baseline
                
                # ROBUST DELTA: Difference between interval mean and historical baseline
                if pd.notna(ma7_last) and pd.notna(interval_mean):
                    delta_to_ma7 = interval_mean - ma7_last
                else:
                    delta_to_ma7 = np.nan

                feature_summary[f"{feat}_interval_mean"] = interval_mean
                feature_summary[f"{feat}_ma7_last"] = ma7_last
                feature_summary[f"{feat}_delta"] = delta_to_ma7

            interval = {
                "id": subid,
                "tstart": tstart,
                "tstop": tstop,
                "status": 1,  # Event occurred
                "enum": event_num,
                # Use demographics from FIRST DAY of this interval
                "sex": group.loc[prev_index, "sex"] if "sex" in group else None,
                "age": group.loc[prev_index, "date"].year - int(group.loc[prev_index, "birthyr"]) if pd.notnull(group.loc[prev_index, "birthyr"]) else None,
                "amyloid": group.loc[prev_index, "amyloid"] if "amyloid" in group else None,
            }
            
            # Compute interval length
            interval_length = (tstop - tstart) + 1
            
            # Calculate other label totals and lookbacks
            for col in label_cols:
                if col != label:
                    interval[col + "_total"] = int((interval_data[col] == 1).sum())

            lookback_data = group.loc[max(prev_index, event_idx - 7):event_idx - 1]
            for col in label_cols:
                if col != label:
                    interval[col + "_last7"] = int((lookback_data[col] == 1).sum())

            for col in label_cols:
                if col != label:
                    interval[col + "_delta"] = interval.get(col + "_total", 0) - interval.get(col + "_last7", 0)
                    interval[col + "_interval_mean"] = (
                        interval.get(col + "_total", 0) / interval_length if interval_length > 0 else 0
                    )

            # Add demographics from FIRST DAY of this interval
            for col in demo_cols:
                interval[col] = group.loc[prev_index, col] if col in group.columns else None

            interval.update(feature_summary)
            intervals.append(interval)
            prev_index = event_idx + 1
            event_num += 1

        # ROBUST FINAL CENSORED INTERVAL (if there are days after the last event)
        if prev_index < len(group):
            tstart = (group.loc[prev_index, 'date'] - start_day).days
            tstop = (group.loc[len(group) - 1, 'date'] - start_day).days
            interval_data = group.loc[prev_index:].copy()
            
            print(f"Final censored interval for subid {subid}: days {tstart} to {tstop}")
            
            # Force no events in final interval
            interval_data[label] = 0
            
            # ROBUST FEATURE CALCULATION FOR FINAL INTERVALS
            feature_summary = {}
            for feat in feature_cols:
                # ROBUST INTERVAL MEAN
                interval_vals = pd.to_numeric(interval_data[feat], errors='coerce')
                interval_mean = interval_vals.mean()
                
                # ROBUST MA7 LAST: Use data before this final interval
                if prev_index > 0:
                    historical_data = group.loc[0:prev_index-1]
                    historical_vals = pd.to_numeric(historical_data[feat], errors='coerce').dropna()
                    
                    if len(historical_vals) >= 1:
                        ma7_last = historical_vals.tail(min(7, len(historical_vals))).mean()
                    else:
                        ma7_last = np.nan
                else:
                    ma7_last = interval_mean
                
                # ROBUST DELTA
                if pd.notna(ma7_last) and pd.notna(interval_mean):
                    delta_to_ma7 = interval_mean - ma7_last
                else:
                    delta_to_ma7 = np.nan

                feature_summary[f"{feat}_interval_mean"] = interval_mean
                feature_summary[f"{feat}_ma7_last"] = ma7_last
                feature_summary[f"{feat}_delta"] = delta_to_ma7
            
            interval = {
                "id": subid,
                "tstart": tstart,
                "tstop": tstop,
                "status": 0,  # Censored
                "enum": event_num,
                # Use demographics from FIRST DAY of this final interval
                "sex": group.loc[prev_index, "sex"] if "sex" in group and prev_index < len(group) else group.iloc[-1].get("sex", None),
                "age": group.loc[prev_index, "date"].year - int(group.loc[prev_index, "birthyr"]) if prev_index < len(group) and pd.notnull(group.loc[prev_index, "birthyr"]) else (group.iloc[-1]["date"].year - int(group.iloc[-1]["birthyr"]) if pd.notnull(group.iloc[-1]["birthyr"]) else None),
                "amyloid": group.loc[prev_index, "amyloid"] if "amyloid" in group and prev_index < len(group) else group.iloc[-1].get("amyloid", None),
            }
            
            # Compute interval length
            interval_length = (tstop - tstart) + 1
            
            # Calculate totals for all other labels
            for col in label_cols:
                if col != label:
                    interval[col + "_total"] = int((interval_data[col] == 1).sum())

            # 7-day lookback for final interval
            lookback_data = interval_data.tail(7)
            for col in label_cols:
                if col != label:
                    interval[col + "_last7"] = int((lookback_data[col] == 1).sum())

            # Calculate deltas and interval means
            for col in label_cols:
                if col != label:
                    interval[col + "_delta"] = interval.get(col + "_total", 0) - interval.get(col + "_last7", 0)
                    interval[col + "_interval_mean"] = (
                        interval.get(col + "_total", 0) / interval_length if interval_length > 0 else 0
                    )
                    
            # Add demographics from FIRST DAY of this final interval
            for col in demo_cols:
                if prev_index < len(group):
                    interval[col] = group.loc[prev_index, col] if col in group.columns else None
                else:
                    interval[col] = group.iloc[-1][col] if col in group.columns else None

            interval.update(feature_summary)
            intervals.append(interval)

# Create dataframe and export
interval_df = pd.DataFrame(intervals)
interval_df = interval_df.sort_values(['id', 'tstart'])

# Enhanced diagnostic checks
print("\n=== DIAGNOSTIC CHECKS ===")
print(f"Total intervals created: {len(interval_df)}")
print(f"Unique participants: {interval_df['id'].nunique()}")
print(f"Status distribution:")
print(interval_df['status'].value_counts())

# Check for NaN values in key feature columns
feature_summary_cols = [col for col in interval_df.columns if any(suffix in col for suffix in ['_interval_mean', '_ma7_last', '_delta'])]
print(f"\nMissing data summary for feature columns:")
for col in feature_summary_cols[:10]:  # Show first 10 as example
    missing_count = interval_df[col].isna().sum()
    missing_pct = (missing_count / len(interval_df)) * 100
    print(f"  {col}: {missing_count} missing ({missing_pct:.1f}%)")

# Check how many participants have only 1 interval (no events)
single_interval_participants = interval_df.groupby('id').size()
no_event_participants = single_interval_participants[single_interval_participants == 1].index.tolist()
multi_event_participants = single_interval_participants[single_interval_participants > 1].index.tolist()

print(f"\nParticipants with NO events (1 interval): {len(no_event_participants)}")
print(f"Participants with events (multiple intervals): {len(multi_event_participants)}")

# Show examples
if no_event_participants:
    example_no_event = interval_df[interval_df['id'] == no_event_participants[0]]
    print(f"\nExample participant with NO events (ID {no_event_participants[0]}):")
    print(example_no_event[['id', 'tstart', 'tstop', 'status', 'enum']].to_string())

if multi_event_participants:
    example_with_events = interval_df[interval_df['id'] == multi_event_participants[0]]
    print(f"\nExample participant WITH events (ID {multi_event_participants[0]}):")
    print(example_with_events[['id', 'tstart', 'tstop', 'status', 'enum']].to_string())

# Verify no final intervals have status=1
problem_intervals = []
for subid in interval_df['id'].unique():
    subj_data = interval_df[interval_df['id'] == subid].sort_values('tstart')
    last_interval = subj_data.iloc[-1]
    if last_interval['status'] == 1:
        problem_intervals.append(subid)

if problem_intervals:
    print(f"\nWARNING: Participants with status=1 in final interval: {problem_intervals}")
else:
    print(f"\n✓ All participants have censored final intervals (status=0)")

# Save the robust interval data
interval_df.to_csv(f"{output_dir}/intervals_all_participants_{label}_robust.csv", index=False)
print(f"\nExported ROBUST intervals for ALL participants to: {output_dir}/intervals_all_participants_{label}_robust.csv")
print(f"This dataset uses robust feature calculations that handle missing data gracefully.")
print(f"Expected improvements:")
print(f"  - Interval Mean models should use ~{len(interval_df)} intervals")
print(f"  - Rolling Mean models should use ~{len(interval_df)} intervals ")