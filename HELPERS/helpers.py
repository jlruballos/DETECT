# helpers.py
"""
Helper functions for data preprocessing, feature engineering, and label generation.
Used across the DETECT feature engineering and LSTM sequence pipelines.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

def preprocess_steps(df):
    """Preprocess steps data by parsing dates and renaming columns."""
    df['date'] = pd.to_datetime(df['date']).dt.date
    #df = df.rename(columns={'steps': 'steps'})
    return df


def label_exact_day(current_date, event_dates):
    """Label if an event occurred exactly on the current date."""
    return int(current_date in event_dates)


def days_since_last_event(current_date, event_dates):
    """Compute days since the last event before or on current date."""
    past = [d for d in event_dates if d <= current_date]
    return (current_date - past[-1]).days if past else np.nan


def days_until_next_event(current_date, event_dates):
    """Compute days until the next event after the current date."""
    future = [d for d in event_dates if d > current_date]
    return (future[0] - current_date).days if future else np.nan


def get_label(daily_df, i, timesteps, label_shift=0):
    """
    Get the label for an LSTM window.
    label_shift=0 means predict immediately after window.
    label_shift=7 means predict 7 days after the window ends.
    currently the label is the fall label, but it can be changed to any other label.
    """
    index = i + timesteps + label_shift
    if index < len(daily_df):
        return daily_df.iloc[index]['label_fall']
    return np.nan

def impute_missing_data(df, columns=None, method='linear', multivariate=False):
	
    """
    Fill missing values in specified columns based on the selected method.
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of str (columns to fill), if None fills all numeric columns
    - method: str, one of ['linear', 'mean', 'median', 'ffill', 'bfill']
    
		- 'linear': Linear interpolation
		- 'nearest': Nearest neighbor interpolation
		- 'mean': Fill with mean value of the column
		- 'median': Fill with median value of the column
		- 'ffill': Forward fill (propagate last valid observation forward)
		- 'bfill': Backward fill (propagate next valid observation backward)
		- 'polynomial': Polynomial interpolation (order 2) can be used for non-linear data
		- 'knn': K-Nearest Neighbors imputation
  		- 'spline': Spline interpolation (order 2) can be used for non-linear data
		- 'random': Randomly fill missing values with existing values in the column
		- 'mice': Multiple Imputation by Chained Equations (MICE) for more complex imputation (uses Bayesian Ridge regression, 
  			can select different estimators)
		- 'pmm': Predictive Mean Matching (PMM) for more complex imputation (uses Bayesian Ridge regression,
			can select different estimators) o ruse multivariate imputation
			- multivariate imputation: is used when multiple columns are missing and we want to use the correlation between them to fill in the missing values.

    Returns:
    - DataFrame with missing values filled
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'linear':
            df[col] = df[col].interpolate(method='linear')
        elif method == 'nearest':
            df[col] = df[col].interpolate(method='nearest', limit_direction='both') 
        elif method == 'mean':
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
        elif method == 'median':
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
        elif method == 'ffill':
            df[col] = df[col].fillna(method='ffill')
        elif method == 'bfill':
            df[col]= df[col].fillna(method='bfill')
        elif method == 'polynomial':
            df[col] = df[col].interpolate(method='polynomial', order=2)
        elif method == 'knn':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=3)
            
            for col in columns:
                if df[col].dropna().empty:
                    print(f"[SKIP] KNN imputation skipped for '{col}' — column is all NaN.")
                    continue

				# Fit and transform single-column DataFrame
                imputed_col = imputer.fit_transform(df[[col]])

				# Safely assign result if lengths match
                if imputed_col.shape[0] == df.shape[0]:
                    df[col] = imputed_col[:, 0]
                else:
                    print(f"[ERROR] KNN imputer returned unexpected shape for '{col}'. Skipping.")
        elif method == 'spline':
            df[col] = df[col].interpolate(method='spline', order=2)
        elif method == 'random':
            observed_values = df[col].dropna().values
            if len(observed_values) > 0:
                random_values = np.random.choice(observed_values, size=df[col].isna().sum(), replace=True)
                df.loc[df[col].isna(), col] = random_values
        elif method == 'mice':
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            if multivariate:
                mice_imputer = IterativeImputer(max_iter=10, random_state=0)
                imputed_array = mice_imputer.fit_transform(df[columns])
                for idx, col in enumerate(columns):
                    df[col] = imputed_array[:, idx]
            else:
                for col in columns:
                    mice_imputer = IterativeImputer(max_iter=10, random_state=0)
                    imputed_array = mice_imputer.fit_transform(df[[col]])
                    df[col] = imputed_array[:, 0]
        elif method == 'pmm':
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            from sklearn.linear_model import BayesianRidge
            
            if multivariate:
                pmm_imputer = IterativeImputer(
					estimator=BayesianRidge(),
					max_iter=10,
					random_state=0,
					sample_posterior=True
				)
                imputed_array = pmm_imputer.fit_transform(df[columns])
                for idx, col in enumerate(columns):
                    df[col] = imputed_array[:, idx]
            else:
                for col in columns:
                    pmm_imputer = IterativeImputer(
						estimator=BayesianRidge(),
						max_iter=10,
						random_state=0,
						sample_posterior=True
					)
                    imputed_array = pmm_imputer.fit_transform(df[[col]])
                    df[col] = imputed_array[:, 0]
        else:
            raise ValueError(f"Unknown imputation method: {method}")
    
    return df

def plot_imp_diag_histo(original_df, imputed_df, columns, subid, output_dir, method_name):
	
    os.makedirs(output_dir, exist_ok=True)
    
    for col in columns:
        real_values = original_df[col].dropna()
        imputed_mask = original_df[col].isna()
        imputed_values = imputed_df.loc[imputed_mask, col]
        
        plt.figure(figsize=(8, 5))
        sns.histplot(
			real_values,
            color = 'blue',
			label='Observed',
			kde=True,
			stat = "density",
			alpha=0.5,
		)
        
        sns.histplot(
			imputed_values,
			color = 'red',
			label='Imputed',
			kde=True,
			stat = "density",
			alpha=0.5,
		)
        
        # sns.scatterplot(
        #     x=imputed_values, 
        #     y=[0.005]*len(imputed_values), 
        #     color='red', 
        #     label='Imputed', 
        #     marker='x', 
        #     s=100
        #     )
        
        plt.title(f"SubID: {subid} | Feature: {col} | Method: {method_name}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.5)
        
        subid_dir = os.path.join(output_dir, f"subid_{subid}")
        os.makedirs(subid_dir, exist_ok=True)
        plot_filename = f"subid_{subid}_{col}_histogram_{method_name}.png"
        plt.savefig(os.path.join(subid_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()
  
def track_missingness(original_df, imputed_df, columns, subid, imputation_method):
    """
    Tracks missing values before and after imputation for a participant.

    Parameters:
        original_df: DataFrame before imputation
        imputed_df: DataFrame after imputation
        columns: list of columns to track
        subid: participant ID
        imputation_method: name of the method used (string)

    Returns:
        A dictionary with missingness information
    """
    missing_before = original_df[columns].isna().sum()
    missing_after = imputed_df[columns].isna().sum()

    record = {
        'subid': subid,
        'imputation_method': imputation_method
    }
    for feature in columns:
        record[f'{feature}_missing_before'] = missing_before[feature]
        record[f'{feature}_missing_after'] = missing_after[feature]

    return record

def plot_imp_diag_timeseries(original_df, imputed_df, columns, subid, output_dir, method_name):
    """
    Plots a time series comparing observed and imputed values over time for diagnostics.

    Parameters:
        original_df: DataFrame before imputation (with NaNs)
        imputed_df: DataFrame after imputation (no NaNs)
        columns: list of columns to plot
        subid: participant ID
        output_dir: folder to save the plots
        method_name: name of imputation method
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for col in columns:
        if 'date' not in original_df.columns:
            print(f"Warning: no date column found for SubID {subid}. Skipping timeseries plot.")
            continue
        
        # Sort by date to be safe
        original_df = original_df.sort_values('date')
        imputed_df = imputed_df.sort_values('date')

        # Determine which points were imputed
        imputed_mask = original_df[col].isna()
        n_total = len(original_df)
        n_imputed = imputed_mask.sum()
        
        if n_total >0:
            imputed_percent = (n_imputed / n_total) * 100
        else:	
            imputed_percent = 0
        
        dates = imputed_df['date']

        plt.figure(figsize=(12, 5))

        # Plot full imputed data as a smooth line
        sns.lineplot(
            		 x=dates, 
                     y=imputed_df[col], 
                     label='All (Observed + Imputed)', 
                     color='grey', 
                     linewidth=1,
                     linestyle='--',
                     zorder = 1
                     )

        # Plot imputed points separately (red crosses)
        if imputed_mask.sum() > 0:
            plt.scatter(
                dates[imputed_mask],
                imputed_df.loc[imputed_mask, col],
                color='red',
                marker='o',
                s=1, 
                label=f'Imputed Points ({imputed_percent:.1f}%)',
                zorder=3
            )

        plt.title(f"SubID: {subid} | Feature: {col} | Method: {method_name}")
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.legend()
        plt.grid(True, alpha=0.5)
        
        subid_dir = os.path.join(output_dir, f"subid_{subid}")
        os.makedirs(subid_dir, exist_ok=True)
        plot_filename = f"subid_{subid}_{col}_timeseries_{method_name}.png"
        plt.savefig(os.path.join(subid_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()

def get_cv_splits(X, y, subid=None, method='groupkfold', n_splits=5):
    """
    Returns list of (train_idx, val_idx) splits based on selected CV method.

    Parameters:
    - X: input features (N samples)
    - y: target labels
    - subid: array of participant IDs (used for GroupKFold)
    - method: 'kfold', 'stratified', or 'groupkfold'
    - n_splits: number of folds

    Returns:
    - List of (train_idx, val_idx) index tuples
    """
    if method == 'kfold':
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(X, y))

    elif method == 'stratified':
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(X, y))

    elif method == 'groupkfold':
        if subid is None:
            raise ValueError("GroupKFold requires `subid` (group labels)")
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X, y, groups=subid))

    else:
        raise ValueError(f"Unsupported CV method: {method}")
    
# Function to remove outliers using IQR method
def remove_outliers(df, column_name):
    col = df[column_name]

    # Skip if column is all NaN or empty
    if col.dropna().empty:
        print(f"[SKIP] '{column_name}' is all NaN or empty — skipping outlier removal.")
        return df

    Q1 = col.quantile(0.05)
    Q3 = col.quantile(0.95)
    IQR = Q3 - Q1

    # Skip if IQR is 0 or NaN (no variation)
    if pd.isna(IQR) or IQR == 0:
        print(f"[SKIP] '{column_name}' has invalid IQR — skipping outlier removal.")
        return df

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter
    filtered_df = df[(col >= lower_bound) & (col <= upper_bound)]

    # Skip if fewer than 5 points would remain
    if filtered_df.shape[0] < 5:
        print(f"[WARN] Too few data points after outlier removal for '{column_name}' — skipping.")
        return df

    # Otherwise, print and return filtered data
    print(f"Outlier removal for {column_name}:")
    print(f"  Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    print(f"  Original size: {len(df)}, Size after outlier removal: {len(filtered_df)}")

    return filtered_df

#funtion to process emfit data
def proc_emfit_data(df):
    
		# Convert date columns to datetime
	df['date'] = pd.to_datetime(df['date'])

	# Process time-related columns
	df['start_sleep_time'] = pd.to_datetime(df['start_sleep']).dt.time
	df['end_sleep_time'] = pd.to_datetime(df['end_sleep']).dt.time
	df['inbed_time'] = pd.to_datetime(df['inbed']).dt.time
	df['outbed_time'] = pd.to_datetime(df['outbed']).dt.time

	# Convert times to hours (fractional)
	def time_to_hour(t):
		return t.hour + t.minute / 60 if pd.notnull(t) else np.nan

	df['inbed_time'] = df['inbed_time'].apply(time_to_hour)
	df['outbed_time'] = df['outbed_time'].apply(time_to_hour)
	df['start_sleep_time'] = df['start_sleep_time'].apply(time_to_hour)
	df['end_sleep_time'] = df['end_sleep_time'].apply(time_to_hour)

	# Add circular encoding for start and end sleep time
	for col in ['start_sleep_time', 'end_sleep_time']:
		df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / 24)
		df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / 24)

	# Calculate time-related features
	df['time_to_sleep'] = ((df['start_sleep_time'] - df['inbed_time']) * 60).mask(lambda x: x < 0)
	df['time_in_bed_after_sleep'] = ((df['outbed_time'] - df['end_sleep_time']) * 60).mask(lambda x: x < 0)
	df['total_time_in_bed'] = (pd.to_datetime(df['outbed'], errors='coerce') - pd.to_datetime(df['inbed'], errors='coerce')).dt.total_seconds() / 3600

	# Convert durations to appropriate units
	df['durationinsleep'] = (df['durationinsleep'] / 60) # to minutes
	df['durationawake'] = df['durationawake'] / 60  # to minutes
	df['duration'] = (df['duration'] / 60)   # to minutes
 
	df['durationinbed'] = (df['durationinbed'] / 60) #to minutes
	df['durationbedexit'] = df['durationbedexit'] / 60  # to minutes
	df['durationsleeponset'] = df['durationsleeponset'] / 60  # to minutes
	df['waso'] = df['waso'] / 60  # to minutes
 
	# -------- REMOVE NEGATIVE VALUES --------
	non_neg_features = [
		'time_to_sleep', 'time_in_bed_after_sleep', 'total_time_in_bed',
		'durationinsleep', 'durationawake', 'duration', 'durationinbed',
		'durationbedexit', 'durationsleeponset', 'waso'
	]

	# Mask negatives (or clip with clip(lower=0) if preferred)
	for col in non_neg_features:
		if col in df.columns:
			n_neg = (df[col] < 0).sum()
			if n_neg > 0:
				print(f"⚠️  {col}: {n_neg} negative values replaced with NaN")
			df[col] = df[col].mask(df[col] < 0)
 
	return df

def add_future_event_window_labels(df, subid_col='subid', date_col='date', event_cols=['label_fall'], horizons=[7]):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([subid_col, date_col]).reset_index(drop=True)

    for event in event_cols:
        for h in horizons:
            future_col = f"{event}_within{h}"
            df[future_col] = (
                df.groupby(subid_col)[event]
                .transform(lambda x: x.rolling(window=h, min_periods=1).max().shift(-h + 1).fillna(0))
                .astype(int)
            )
    return df