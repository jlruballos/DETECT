#!/usr/bin/env python3

import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- CONFIG ----------
input_path = "/mnt/d/DETECT/OUTPUT/survey_processing/survey_cleaned.csv"
output_dir = "/mnt/d/DETECT/OUTPUT/survey_processing/event_histograms_aggregate"
os.makedirs(output_dir, exist_ok=True)

# ---------- Load Data ----------
df = pd.read_csv(input_path, parse_dates=[
    'fall1_date', 'mood_blue_date', 'mood_lonely_date',
    'MED1_DATE', 'MED2_DATE', 'MED3_DATE', 'MED4_DATE'
])
df['subid'] = df['subid'].astype(str)

# ---------- Define Events ----------
event_configs = {
    'fall':        ['fall1_date'],
    'hospital':    ['hospital_visit', 'hospital_visit_2'],
    'accident':    ['ACDT1_DATE', 'ACDT2_DATE', 'ACDT3_DATE'],
    'medication':  ['MED1_DATE', 'MED2_DATE', 'MED3_DATE', 'MED4_DATE'],
    'blue_mood':   ['mood_blue_date'],
    'lonely_mood': ['mood_lonely_date']
}

# ---------- Processing Function ----------
def generate_aggregate_histogram(df, date_cols, event_name):
    all_events = []

    for col in date_cols:
        if col in df.columns:
            temp = df[['subid', col]].dropna().copy()
            temp.columns = ['subid', 'event_date']
            temp['event_date'] = pd.to_datetime(temp['event_date'], errors='coerce')
            all_events.append(temp)

    if not all_events:
        return

    combined = pd.concat(all_events).dropna()
    combined['week'] = combined['event_date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Count events per (subid, week)
    weekly_counts = combined.groupby(['subid', 'week']).size().reset_index(name='event_count')

    # Build full grid of (subid, week)
    all_weeks = pd.date_range(combined['week'].min(), combined['week'].max(), freq='W-MON')
    all_subid_week = pd.MultiIndex.from_product([df['subid'].unique(), all_weeks], names=['subid', 'week'])
    merged = pd.DataFrame(index=all_subid_week).reset_index()
    merged = merged.merge(weekly_counts, on=['subid', 'week'], how='left').fillna(0)
    merged['event_count'] = merged['event_count'].astype(int)

    # Count how many participant-weeks had 0, 1, 2... events
    hist = merged['event_count'].value_counts().sort_index().reset_index()
    hist.columns = ['num_events', 'num_participant_weeks']

    # ---------- Plot ----------
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=hist, x='num_events', y='num_participant_weeks', color='steelblue')
    plt.title(f'Event Count per Participant-Week: {event_name.capitalize()}')
    plt.xlabel(f'{event_name.capitalize()} Events in a Week')
    plt.ylabel('Number of Participant-Weeks')

    for bar in ax.patches:
        count = int(bar.get_height())
        if count > 0:
            ax.annotate(str(count),
                        (bar.get_x() + bar.get_width() / 2, count),
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{event_name}_event_histogram.png"))
    plt.close()

# ---------- Run ----------
for event_name, date_cols in event_configs.items():
    generate_aggregate_histogram(df, date_cols, event_name)

# ---------- Summary Ratio Plot ----------
summary_rows = []

for event_name, date_cols in event_configs.items():
    all_events = []

    for col in date_cols:
        if col in df.columns:
            temp = df[['subid', col]].dropna().copy()
            temp.columns = ['subid', 'event_date']
            temp['event_date'] = pd.to_datetime(temp['event_date'], errors='coerce')
            all_events.append(temp)

    if not all_events:
        continue

    combined = pd.concat(all_events).dropna()
    combined['week'] = combined['event_date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly_counts = combined.groupby(['subid', 'week']).size().reset_index(name='event_count')

    all_weeks = pd.date_range(combined['week'].min(), combined['week'].max(), freq='W-MON')
    all_subid_week = pd.MultiIndex.from_product([df['subid'].unique(), all_weeks], names=['subid', 'week'])
    merged = pd.DataFrame(index=all_subid_week).reset_index()
    merged = merged.merge(weekly_counts, on=['subid', 'week'], how='left').fillna(0)
    merged['event_count'] = merged['event_count'].astype(int)

    num_with_event = (merged['event_count'] > 0).sum()
    num_no_event = (merged['event_count'] == 0).sum()

    ratio = num_with_event / num_no_event if num_no_event > 0 else float('inf')

    summary_rows.append({
        'event': event_name,
        'weeks_with_event': num_with_event,
        'weeks_without_event': num_no_event,
        'ratio': ratio
    })

summary_df = pd.DataFrame(summary_rows)

# ---------- Plot Ratio ----------
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=summary_df, x='event', y='ratio', color='darkorange')

for bar, label in zip(ax.patches, summary_df['ratio']):
    ax.annotate(f'{label:.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom')

plt.title("Ratio of Participant-Weeks With vs Without Events")
plt.ylabel("Ratio (Weeks with ≥1 Event / Weeks with 0 Events)")
plt.xlabel("Event Type")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "event_week_ratio_summary.png"))
plt.close()

# ---------- Unique Participants per Event ----------
unique_counts = []

for event_name, date_cols in event_configs.items():
    all_events = []

    for col in date_cols:
        if col in df.columns:
            temp = df[['subid', col]].dropna().copy()
            temp.columns = ['subid', 'event_date']
            all_events.append(temp)

    if not all_events:
        continue

    combined = pd.concat(all_events).dropna()
    unique_ids = combined['subid'].nunique()

    unique_counts.append({
        'event': event_name,
        'unique_participants': unique_ids
    })

unique_df = pd.DataFrame(unique_counts)

# ---------- Plot ----------
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=unique_df, x='event', y='unique_participants', color='seagreen')

for bar, label in zip(ax.patches, unique_df['unique_participants']):
    ax.annotate(f'{label}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom')

plt.title("Number of Unique Participants with ≥1 Event")
plt.ylabel("Unique Participants")
plt.xlabel("Event Type")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "unique_participants_per_event.png"))
plt.close()

# ---------- Proportion of Unique Participants per Event ----------
total_unique_subids = df['subid'].nunique()
prop_counts = []

for event_name, date_cols in event_configs.items():
    all_events = []

    for col in date_cols:
        if col in df.columns:
            temp = df[['subid', col]].dropna().copy()
            temp.columns = ['subid', 'event_date']
            all_events.append(temp)

    if not all_events:
        continue

    combined = pd.concat(all_events).dropna()
    unique_ids = combined['subid'].nunique()
    proportion = unique_ids / total_unique_subids

    prop_counts.append({
        'event': event_name,
        'percent_with_event': proportion * 100  # convert to %
    })

prop_df = pd.DataFrame(prop_counts)

# ---------- Plot ----------
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=prop_df, x='event', y='percent_with_event', color='orchid')

for bar, label in zip(ax.patches, prop_df['percent_with_event']):
    ax.annotate(f'{label:.1f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom')

plt.title("Percent of Participants with ≥1 Event")
plt.ylabel("Percent of Unique Participants (%)")
plt.xlabel("Event Type")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "percent_unique_participants_per_event.png"))
plt.close()
