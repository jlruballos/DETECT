#!/usr/bin/env python3
"""
EMFIT sleep scatter (marker-only) + survey event vlines.
- Day 0 = earliest sleep date per subject
- No moving averages, no connecting lines (gaps = missing days)
- Events: Fall (injury / no injury), Hospital, Medication change, Living change, Accident
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

program_name = "emfit_raw_events"
base_path = "/mnt/d/DETECT"
output_dir = os.path.join(base_path, "OUTPUT", program_name)
os.makedirs(output_dir, exist_ok=True)

FALLS_PATH = os.path.join(base_path, "OUTPUT", "survey_processing", "survey_cleaned.csv")
EMFIT_PATH = os.path.join(base_path, "OUTPUT", "emfit_processing", "emfit_sleep_1.csv")

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "figure.dpi": 300,
})

REMOVE_OUTLIERS = False
IQR_K = 1.5

def parse_date_safe(s: pd.Series):
    return pd.to_datetime(s, errors="coerce")

def remove_outliers_iqr(df: pd.DataFrame, column: str, k: float = IQR_K) -> pd.DataFrame:
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]

def build_day_map(sleep_df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    # Map all observed sleep dates to Day index
    s_dates = pd.to_datetime(sleep_df["date"]).dt.normalize().dropna().unique()
    all_dates = sorted(pd.to_datetime(np.unique(s_dates)))
    day_map = pd.DataFrame({"date": all_dates})
    day_map["Day"] = (day_map["date"] - start_date).dt.days
    return day_map

def add_event_vlines(ax, day_map_df, event_col, color, label_once):
    shown = False
    for d in event_col.dropna().dt.normalize().unique():
        match = day_map_df.loc[day_map_df["date"] == d, "Day"]
        if not match.empty:
            ax.axvline(float(match.iloc[0]), color=color, linestyle="-", linewidth=1.2,
                       label=(label_once if not shown else None))
            shown = True

def dedup_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen, new_h, new_l = set(), [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l); new_h.append(h); new_l.append(l)
    if new_h:
        ax.legend(new_h, new_l, loc="best", frameon=True)

print("Loading data …")
falls = pd.read_csv(FALLS_PATH, low_memory=False)
sleep = pd.read_csv(EMFIT_PATH, low_memory=False)

# Parse dates
sleep["date"] = parse_date_safe(sleep["date"]).dt.normalize()

# Pick the sleep metric column — change here if needed
METRIC_COL = "sleepscore"  # e.g., "sleep_score" / "total_sleep" / "sleep_efficiency"
if METRIC_COL not in sleep.columns:
    raise ValueError(f"Column '{METRIC_COL}' not found in EMFIT file. Available: {list(sleep.columns)}")

sleep_daily = sleep.rename(columns={METRIC_COL: "sleepscore"})[["subid", "date", "sleepscore"]]

# Optional outlier removal (usually OFF for missingness inspection)
if REMOVE_OUTLIERS:
    sleep_daily = (sleep_daily.groupby("subid", as_index=False, group_keys=False)
                   .apply(lambda df: remove_outliers_iqr(df, "sleepscore")))

# Events
falls = falls.copy()
falls["fall_date"]          = parse_date_safe(falls.get("fall1_date"))
falls["injury"]             = falls.get("FALL1_INJ")
falls["hospital_date"]      = parse_date_safe(falls.get("HCRU1_DATE"))
falls["med_change_date"]    = parse_date_safe(falls.get("MED1_DATE"))
falls["living_change_date"] = parse_date_safe(falls.get("SPACE_DATE"))
falls["accident_date"]      = parse_date_safe(falls.get("ACDT1_DATE"))

subjects = sorted(sleep_daily["subid"].dropna().unique())
print(f"Processing {len(subjects)} subjects …")

for subid in subjects:
    s_sub = sleep_daily[sleep_daily["subid"] == subid].copy()
    if s_sub.empty:
        continue

    start_date = s_sub["date"].min()
    if pd.isna(start_date):
        continue

    day_map = build_day_map(s_sub, start_date)
    s_sub = s_sub.merge(day_map, on="date", how="inner")

    ev_sub = falls[falls["subid"] == subid].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(s_sub["Day"], s_sub["sleepscore"], marker=".", linestyle="", alpha=0.7, label="Sleep")

    ax.set_ylabel("Sleep")
    ax.set_title(f"Subject {subid} — Sleep (Day 0 = {start_date.date()})")
    ax.grid(True, alpha=0.3)

    # Event lines (only appear if the event date exists in sleep dates)
    add_event_vlines(ax, day_map, ev_sub.loc[ev_sub["injury"] == 1.0, "fall_date"], color="orange", label_once="Fall (Injury)")
    add_event_vlines(ax, day_map, ev_sub.loc[(ev_sub["fall_date"].notna()) & (ev_sub["injury"] != 1.0), "fall_date"], color="grey",   label_once="Fall (No Injury)")
    add_event_vlines(ax, day_map, ev_sub["hospital_date"],      color="purple", label_once="Hospital Visit")
    add_event_vlines(ax, day_map, ev_sub["med_change_date"],    color="green",  label_once="Medication Change")
    add_event_vlines(ax, day_map, ev_sub["living_change_date"], color="brown",  label_once="Living Change")
    add_event_vlines(ax, day_map, ev_sub["accident_date"],      color="red",    label_once="Accident")

    dedup_legend(ax)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{subid}_sleep_scatter_events.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print(f"\n✅ Done. Plots saved to: {output_dir}")
