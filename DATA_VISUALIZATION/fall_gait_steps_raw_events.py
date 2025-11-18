#!/usr/bin/env python3
"""
Gait & Steps scatter plots with event markers (no moving averages, no connecting lines)
- Day 0 := earliest date across steps and gait for each subject (avoids the initial 7-day gap issue)
- Raw daily values only (marker-only scatter) so gaps reveal missing days
- Multiple event types: Fall (injury / no injury), Hospital visit, Medication change, Living situation change
- Thin, solid vertical lines for events

Expected inputs (from your DETECT pipeline):
- GAIT_PATH: COMBINED_NYCE_Area_Data_DETECT_GAIT_summary.csv (needs columns: homeid, start_time, gait_speed)
- CONTEXT_PATH: homeids_subids_NYCE.csv (maps unique home_id→sub_id one-to-one)
- FALLS_PATH: survey_cleaned.csv (needs columns: subid, fall1_date, FALL1_INJ, HCRU1_DATE, MED1_DATE, SPACE_DATE)
- STEPS_PATH: watch_steps_cleaned.csv (needs columns: subid, date, steps)

Output: one PNG per subject saved to OUTPUT/fall_gait_sp_plots_v2
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------- Config ---------------------------- #
program_name = "fall_gait_steps_raw_events"

# Paths
base_path = "/mnt/d/DETECT"
base_c_path = "/mnt/d/DETECT_33125/DETECT_Data_Pull_2024-12-16"
output_dir = os.path.join(base_path, "OUTPUT", program_name)
os.makedirs(output_dir, exist_ok=True)

GAIT_PATH = os.path.join(base_path, "OUTPUT", "GAIT", "COMBINED_NYCE_Area_Data_DETECT_GAIT_summary.csv")
CONTEXT_PATH = os.path.join(base_c_path, "_CONTEXT_FILES", "Study_Home-Subject_Dates_2024-12-16", "homeids_subids_NYCE.csv")
FALLS_PATH = os.path.join(base_path, "OUTPUT", "survey_processing", "survey_cleaned.csv")
STEPS_PATH = os.path.join(base_path, "OUTPUT", "watch_steps_processing", "watch_steps_cleaned.csv")

# Style
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 300,
})

# Options
REMOVE_OUTLIERS = True  # keep raw to better visualize missingness
IQR_K = 1.5

# ------------------------ Helper functions ---------------------- #

def remove_outliers_iqr(df: pd.DataFrame, column: str, k: float = IQR_K) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]


def parse_date_safe(s: pd.Series):
    """Parse to datetime64[ns] (naive), coerce errors to NaT, and downcast to date where needed."""
    return pd.to_datetime(s, errors="coerce")


def add_event_vlines(ax, day_map_df, event_col, color, label_once):
    """Draw thin, solid vlines for each non-null event date present in the subject.
    day_map_df needs columns: date, Day
    event_col is a Series of datetimes (datetime64[ns]) for this subject
    """
    shown = False
    for d in event_col.dropna().dt.normalize().unique():
        # map this date to a Day index if present in either steps or gait timeline
        # we try exact match; if not present, skip (no point plotting outside timeline)
        match = day_map_df.loc[day_map_df["date"] == d, "Day"]
        if not match.empty:
            ax.axvline(float(match.iloc[0]), color=color, linestyle="-", linewidth=1.2,
                       label=(label_once if not shown else None))
            shown = True


def build_day_map(steps_df, gait_df, start_date):
    """Construct a map of all observed dates → Day index (0..N) across BOTH streams.
    This ensures a shared x-axis so gaps represent true missing days in either stream.
    """
    s_dates = pd.to_datetime(steps_df["date"]).dt.normalize().dropna().unique()
    g_dates = pd.to_datetime(gait_df["date"]).dt.normalize().dropna().unique()
    all_dates = sorted(pd.to_datetime(np.unique(np.concatenate([s_dates, g_dates]))))
    day_map = pd.DataFrame({"date": all_dates})
    day_map["Day"] = (day_map["date"] - start_date).dt.days
    return day_map

# ---------------------------- Load ------------------------------ #
print("Loading data …")
gait = pd.read_csv(GAIT_PATH, low_memory=False)
map_df = pd.read_csv(CONTEXT_PATH, low_memory=False)
falls = pd.read_csv(FALLS_PATH, low_memory=False)
steps = pd.read_csv(STEPS_PATH, low_memory=False)

# Keep only one-to-one home_id→sub_id mappings
map_df = map_df.groupby("home_id").filter(lambda x: len(x) == 1)

# Join gait→subid
gait = gait.merge(map_df, left_on="homeid", right_on="home_id", how="inner").drop(columns=["home_id"]).rename(columns={"sub_id": "subid"})

# Basic parsing
gait["date"] = parse_date_safe(gait["start_time"]).dt.normalize()
steps["date"] = parse_date_safe(steps["date"]).dt.normalize()

# Daily aggregates (raw means)
gait_daily = gait.groupby(["subid", "date"], as_index=False)["gait_speed"].mean()
steps_daily = steps.rename(columns={"steps": "daily_steps"})[["subid", "date", "daily_steps"]]

# Optionally remove extreme values (usually keep raw for missingness views)
if REMOVE_OUTLIERS:
    gait_daily = gait_daily.groupby("subid", as_index=False, group_keys=False).apply(lambda df: remove_outliers_iqr(df, "gait_speed"))
    steps_daily = steps_daily.groupby("subid", as_index=False, group_keys=False).apply(lambda df: remove_outliers_iqr(df, "daily_steps"))

# Event parsing from surveys
falls = falls.copy()
falls["fall_date"] = parse_date_safe(falls.get("fall1_date"))
falls["injury"] = falls.get("FALL1_INJ")  # 1.0 == injury, else NaN/0
falls["hospital_date"] = parse_date_safe(falls.get("HCRU1_DATE"))
falls["med_change_date"] = parse_date_safe(falls.get("MED1_DATE"))
falls["living_change_date"] = parse_date_safe(falls.get("SPACE_DATE"))
falls["accident_date"] = parse_date_safe(falls.get("ACDT1_DATE"))

# Subjects with both streams
subjects = sorted(set(gait_daily["subid"]).intersection(steps_daily["subid"]))
print(f"Processing {len(subjects)} subjects …")

for subid in subjects:
    print(f"  → {subid}")
    g_sub = gait_daily[gait_daily["subid"] == subid].copy()
    s_sub = steps_daily[steps_daily["subid"] == subid].copy()

    if g_sub.empty or s_sub.empty:
        continue

    # Shared Day-0 = earliest date across BOTH streams
    start_date = min(g_sub["date"].min(), s_sub["date"].min())
    day_map = build_day_map(s_sub, g_sub, start_date)

    # Map each stream to shared Day index
    g_sub = g_sub.merge(day_map, on="date", how="inner")
    s_sub = s_sub.merge(day_map, on="date", how="inner")

    # Gather this subject's events
    ev_sub = falls[falls["subid"] == subid].copy()

    # -------------------------- Plot --------------------------- #
    # Figure A: STEPS (raw scatter, no lines)
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(s_sub["Day"], s_sub["daily_steps"], marker=".", linestyle="", alpha=0.7, label="Daily Steps")
    ax1.set_ylabel("Steps")
    ax1.set_title(f"Subject {subid} — Steps (Day 0 = {start_date.date()})")
    ax1.grid(True, alpha=0.3)

    # Event vlines on steps
    add_event_vlines(ax1, day_map, ev_sub.loc[ev_sub["injury"] == 1.0, "fall_date"], color="orange", label_once="Fall (Injury)")
    add_event_vlines(ax1, day_map, ev_sub.loc[(ev_sub["fall_date"].notna()) & (ev_sub["injury"] != 1.0), "fall_date"], color="grey", label_once="Fall (No Injury)")
    add_event_vlines(ax1, day_map, ev_sub["hospital_date"], color="purple", label_once="Hospital Visit")
    add_event_vlines(ax1, day_map, ev_sub["med_change_date"], color="green", label_once="Medication Change")
    add_event_vlines(ax1, day_map, ev_sub["living_change_date"], color="brown", label_once="Living Change")
    add_event_vlines(ax1, day_map, ev_sub["accident_date"], color="pink", label_once="Accident")

    # Dedup legend
    def dedup_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        new_h, new_l = [], []
        for h, l in zip(handles, labels):
            if l and l not in seen:
                seen[l] = True
                new_h.append(h)
                new_l.append(l)
        if new_h:
            ax.legend(new_h, new_l, loc="best", frameon=True)

    dedup_legend(ax1)
    plt.tight_layout()
    out_path_steps = os.path.join(output_dir, f"{subid}_steps_scatter_events.png")
    plt.savefig(out_path_steps, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {out_path_steps}")

    # Figure B: GAIT (raw scatter, no lines)
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(g_sub["Day"], g_sub["gait_speed"], marker=".", linestyle="", alpha=0.7, label="Daily Gait Speed")
    ax2.set_ylabel("Gait Speed (m/s)")
    ax2.set_xlabel("Day")
    ax2.set_title(f"Subject {subid} — Gait Speed (Day 0 = {start_date.date()})")
    ax2.grid(True, alpha=0.3)

    # Event vlines on gait
    add_event_vlines(ax2, day_map, ev_sub.loc[ev_sub["injury"] == 1.0, "fall_date"], color="orange", label_once="Fall (Injury)")
    add_event_vlines(ax2, day_map, ev_sub.loc[(ev_sub["fall_date"].notna()) & (ev_sub["injury"] != 1.0), "fall_date"], color="grey", label_once="Fall (No Injury)")
    add_event_vlines(ax2, day_map, ev_sub["hospital_date"], color="purple", label_once="Hospital Visit")
    add_event_vlines(ax2, day_map, ev_sub["med_change_date"], color="green", label_once="Medication Change")
    add_event_vlines(ax2, day_map, ev_sub["living_change_date"], color="brown", label_once="Living Change")
    add_event_vlines(ax2, day_map, ev_sub["accident_date"], color="pink", label_once="Accident")

    dedup_legend(ax2)
    plt.tight_layout()
    out_path_gait = os.path.join(output_dir, f"{subid}_gait_scatter_events.png")
    plt.savefig(out_path_gait, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path_gait}")


print(f"\n✅ Done. Plots saved to: {output_dir}")
