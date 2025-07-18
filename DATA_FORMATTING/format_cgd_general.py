import pandas as pd

# Load labeled daily data
df = pd.read_csv("/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_mean.csv", parse_dates=["date"])
#df = pd.read_csv("/mnt/d/DETECT/OUTPUT/raw_export_for_r/imputed_detect_data_pmm_per_participant.csv", parse_dates=["date"])

label = 'label_mood_blue' # Change to 'label_hospital' 'label_mood_lonely' 'label_mood_blue', 'label_accident', 'label_medication'
label_cols = ['label_hospital', 'label_accident', 'label_medication', 'label_mood_lonely', 'label_mood_blue', 'label_fall']

# Filter to subjects with known hospital visit labels
df = df[df[label].notna()]  # <- change here
df = df.sort_values(['subid', 'date'])

# Define feature and demographic columns
feature_cols = [
    'steps', 'gait_speed', 'awakenings', 'bedexitcount', 'end_sleep_time',
    'inbed_time', 'outbed_time', 'sleepscore', 'durationinsleep', 'durationawake',
    'waso', 'hrvscore', 'start_sleep_time', 'time_to_sleep',
    'time_in_bed_after_sleep', 'tossnturncount', 'maxhr', 'avghr', 'avgrr', 'Night_Bathroom_Visits', 'Night_Kitchen_Visits', 
]


demo_cols = ['birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat','moca_avg', 'cogstat']

intervals = []

for subid, group in df.groupby('subid'):
    group = group.sort_values('date').reset_index(drop=True)
    start_day = group.loc[0, 'date']
    
    # Indices where a hospital visit occurred
    hosp_indices = group.index[group[label] == 1].tolist()
    
    if len(hosp_indices) == 0:
        continue  # no hospital visit
    
    prev_index = 0
    event_num = 0
    
    for hosp_idx in hosp_indices:
        hosp_date = group.loc[hosp_idx, 'date']
        tstart = (group.loc[prev_index, 'date'] - start_day).days
        tstop = (hosp_date - start_day).days
        interval_data = group.loc[prev_index:hosp_idx]
        
        feature_summary = {}
        for feat in feature_cols:
            vals = pd.to_numeric(interval_data[feat], errors='coerce')
            cumulative_vals = pd.to_numeric(group.loc[prev_index:hosp_idx - 1, feat], errors='coerce')
            ma7_last = cumulative_vals[-7:].mean() if not cumulative_vals.empty else None
            interval_mean = vals.mean()
            # Calculate delta as the mean of the interval minus the last 7-day mean
            delta_to_ma7 = (interval_mean - ma7_last) if ma7_last is not None else None

            feature_summary[f"{feat}_interval_mean"] = interval_mean
            feature_summary[f"{feat}_ma7_last"] = ma7_last
            feature_summary[f"{feat}_delta"] = delta_to_ma7

        interval = {
            "id": subid,
            "tstart": tstart,
            "tstop": tstop,
            "status": 1,
            "enum": event_num,
            "sex": group.loc[hosp_idx, "sex"] if "sex" in group else None,
            "age": hosp_date.year - int(group.loc[hosp_idx, "birthyr"]) if pd.notnull(group.loc[hosp_idx, "birthyr"]) else None,
            "amyloid": group.loc[hosp_idx, "amyloid"] if "amyloid" in group else None,
        }
        
        # Compute interval length
        interval_length = (tstop - tstart) + 1
        
        for col in label_cols:
            if col != label:
                interval[col + "_total"] = int((interval_data[col] == 1).sum())

        lookback_data = group.loc[max(prev_index, hosp_idx - 7):hosp_idx - 1]
        for col in label_cols:
            if col != label:
                interval[col + "_last7"] = int((lookback_data[col] == 1).sum())

        for col in label_cols:
            if col != label:
                interval[col + "_delta"] = interval.get(col + "_total", 0) - interval.get(col + "_last7", 0)
                interval[col + "_interval_mean"] = (
                    interval.get(col + "_total", 0) / interval_length if interval_length > 0 else 0
                )

        for col in demo_cols:
            interval[col] = group.loc[hosp_idx, col] if col in group.columns else None

        interval.update(feature_summary)
        intervals.append(interval)
        prev_index = hosp_idx + 1
        event_num += 1

    # Add final censored interval
    if prev_index <= len(group):
        # tstart = (group.loc[prev_index, 'date'] - start_day).days
        # tstop = (group.loc[len(group) - 1, 'date'] - start_day).days
        # # Force final interval to be censored even if label == 1 appears
        # interval_data = group.loc[prev_index:].copy()
        if prev_index < len(group):
        # Normal case - there are days after the last event
            tstart = (group.loc[prev_index, 'date'] - start_day).days
            tstop = (group.loc[len(group) - 1, 'date'] - start_day).days
            interval_data = group.loc[prev_index:].copy()
        else:
			# Edge case - prev_index == len(group) (last event was on final day)
			# Create a 1-day censored interval starting the day off last event
            last_event_date = group.loc[prev_index - 1, 'date']
            tstart = (last_event_date - start_day).days
            tstop = tstart  # Same day interval (can be same as tstart)
			# Create empty interval_data with same structure as group
            interval_data = group.iloc[0:0].copy()  # Empty but with correct columns
        
        print(f"Final interval for subid {subid}:")
        print(f"  prev_index: {prev_index}, len(group): {len(group)}")
        print(f"  interval_data shape: {interval_data.shape}")
        print(f"  any events in final interval: {any(group.loc[prev_index:, label] == 1)}")
        print(f"  forcing status = 0")
        
        if any(group.loc[prev_index:, label] == 1):
            print(f"Warning: subid {subid} had event(s) in final interval — forcing status = 0")
        
        interval_data[label] = 0  # force all labels to 0
        
        feature_summary = {}
        for feat in feature_cols:
            vals = pd.to_numeric(interval_data[feat], errors='coerce')
            cumulative_vals = pd.to_numeric(group.loc[:prev_index - 1, feat], errors='coerce')
            ma7_last = cumulative_vals[-7:].mean() if not cumulative_vals.empty else None
            interval_mean = vals.mean()
            delta_to_ma7 = (interval_mean - ma7_last) if ma7_last is not None else None

            feature_summary[f"{feat}_interval_mean"] = interval_mean
            feature_summary[f"{feat}_ma7_last"] = ma7_last
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
        
        # Compute interval length
        interval_length = (tstop - tstart) + 1
        
        # Calculate totals for all other labels (excluding the target label)
        for col in label_cols:
            if col != label:
                interval[col + "_total"] = int((interval_data[col] == 1).sum())

        # 7-day lookback - use last 7 days of interval for censored intervals
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
                
        for col in demo_cols:
            interval[col] = group.iloc[-1][col] if col in group.columns else None

        interval.update(feature_summary)
        intervals.append(interval)


# Create dataframe and export
interval_df = pd.DataFrame(intervals)
interval_df = interval_df.sort_values(['id', 'tstart'])

# Diagnostic checks before export
print("\n=== DIAGNOSTIC CHECKS ===")
print(f"Total intervals created: {len(interval_df)}")
print(f"Unique participants: {interval_df['id'].nunique()}")
print(f"Status distribution:")
print(interval_df['status'].value_counts())

# Check for problematic participant 2394
if 2394 in interval_df['id'].values:
    subj_2394 = interval_df[interval_df['id'] == 2394]
    print(f"\nParticipant 2394 intervals:")
    print(subj_2394[['id', 'tstart', 'tstop', 'status', 'enum']].to_string())
    
    # Check if last interval has status=1
    last_interval = subj_2394.iloc[-1]
    print(f"\nLast interval for 2394: status={last_interval['status']}")

# Check for any intervals where status=1 and it's the last interval for that participant
problem_intervals = []
for subid in interval_df['id'].unique():
    subj_data = interval_df[interval_df['id'] == subid].sort_values('tstart')
    last_interval = subj_data.iloc[-1]
    if last_interval['status'] == 1:
        problem_intervals.append(subid)

if problem_intervals:
    print(f"\nParticipants with status=1 in final interval: {problem_intervals}")
else:
    print(f"\nNo participants have status=1 in final interval - Python code looks correct!")
    
interval_df.to_csv(f"/mnt/d/DETECT/OUTPUT/raw_export_for_r/intervals_{label}.csv", index=False)
print(f"Exported hospital-based intervals to CSV for R: {label}.")