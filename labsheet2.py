import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
)

# ----------------------------------------------------
# Setup: Load Sample Dataset for Demonstration
# ----------------------------------------------------
df = pd.DataFrame({
    'Age': [25, 30, np.nan, 35, 200, 28, 40, 22, 38, np.nan],
    'Salary': [50000, 60000, 55000, 65000, 500000, 48000, 70000, np.nan, 62000, 58000],
    'Gender': ['Male', 'Female', 'Female', np.nan, 'Male', 'Female', 'Male', 'Female', np.nan, 'Male'],
    'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'Empty_Col': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan]
})

# Program 1: Load a dataset and identify missing values in each column
missing_counts = df.isnull().sum()

# Program 2: Display the percentage of missing values in every feature
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Program 3: Remove rows containing missing values from the dataset
df_no_missing_rows = df.dropna()

# Program 4: Remove columns having more than 50% missing values
df_no_missing_cols = df.dropna(thresh=len(df) * 0.5, axis=1)

# Program 5: Replace missing numerical values using the mean
df_mean_imp = df.copy()
df_mean_imp['Age'] = df_mean_imp['Age'].fillna(df_mean_imp['Age'].mean())

# Program 6: Replace missing numerical values using the median
df_median_imp = df.copy()
df_median_imp['Age'] = df_median_imp['Age'].fillna(df_median_imp['Age'].median())

# Program 7: Replace missing categorical values using the mode
df_mode_imp = df.copy()
df_mode_imp['Gender'] = df_mode_imp['Gender'].fillna(df_mode_imp['Gender'].mode()[0])

# Program 8: Fill missing values using forward fill (Forward Propagation)
df_ffill = df.ffill()

# Program 9: Fill missing values using backward fill (Backward Propagation)
df_bfill = df.bfill()

# Program 10: Compare the dataset before and after handling missing values
before_shape = df.shape
after_shape = df_ffill.shape

# Program 11: Detect outliers using the Interquartile Range (IQR) method
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR))]

# Program 12: Detect outliers using the Z-score method
z_scores = np.abs(stats.zscore(df['Age'].dropna()))
outliers_zscore = df.dropna(subset=['Age'])[z_scores > 2]

# Program 13: Visualize outliers using a Box Plot
plt.figure()
sns.boxplot(x=df['Age'])
plt.close()

# Program 14: Visualize outliers using a Scatter Plot
plt.figure()
plt.scatter(df.index, df['Age'])
plt.close()

# Program 15: Remove outliers using the IQR method
df_no_outliers = df[(df['Age'] >= (Q1 - 1.5 * IQR)) & (df['Age'] <= (Q3 + 1.5 * IQR))]

# Program 16: Replace outliers with the median value
df_median_outliers = df.copy()
median_age = df_median_outliers['Age'].median()
df_median_outliers.loc[(df_median_outliers['Age'] < (Q1 - 1.5 * IQR)) | (df_median_outliers['Age'] > (Q3 + 1.5 * IQR)), 'Age'] = median_age

# Program 17: Cap outliers using percentile-based capping
df_capped = df.copy()
lower_limit = df_capped['Age'].quantile(0.05)
upper_limit = df_capped['Age'].quantile(0.95)
df_capped['Age'] = np.clip(df_capped['Age'], lower_limit, upper_limit)

# Program 18: Compare the dataset before and after outlier treatment
desc_before = df['Age'].describe()
desc_after = df_capped['Age'].describe()

# Setup clean numerical dataset for scaling
df_num = df[['Age', 'Salary']].fillna(df[['Age', 'Salary']].median())

# Program 19: Apply Min-Max Normalization to numerical features
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df_num), columns=df_num.columns)

# Program 20: Apply Standardization (Z-score Scaling)
std_scaler = StandardScaler()
df_std = pd.DataFrame(std_scaler.fit_transform(df_num), columns=df_num.columns)

# Program 21: Apply Robust Scaling to handle outliers
robust_scaler = RobustScaler()
df_robust = pd.DataFrame(robust_scaler.fit_transform(df_num), columns=df_num.columns)

# Program 22: Apply Max Absolute Scaling
maxabs_scaler = MaxAbsScaler()
df_maxabs = pd.DataFrame(maxabs_scaler.fit_transform(df_num), columns=df_num.columns)

# Program 23: Compare original and normalized datasets
comp_original = df_num.describe()
comp_normalized = df_minmax.describe()

# Program 24: Visualize the effect of normalization using histograms
fig, axes = plt.subplots(1, 2)
df_num['Salary'].hist(ax=axes[0])
df_minmax['Salary'].hist(ax=axes[1])
plt.close()

# Program 25: Visualize the effect of scaling using box plots
plt.figure()
sns.boxplot(data=df_std)
plt.close()

# Program 26: Compare different scaling techniques on the same dataset
scaling_comparison = pd.DataFrame({
    'Original': df_num['Salary'],
    'MinMax': df_minmax['Salary'],
    'Standard': df_std['Salary'],
    'Robust': df_robust['Salary']
})

# Program 27: Encode categorical variables using Label Encoding
df_label = df.copy()
le = LabelEncoder()
df_label['Gender_Encoded'] = le.fit_transform(df_label['Gender'].astype(str))

# Program 28: Encode categorical variables using One-Hot Encoding
df_onehot = pd.get_dummies(df[['Gender']], drop_first=True)

# Program 29: Perform Binary Encoding on categorical data
# Alternative representation using pandas category codes in binary format
df_binary = df.copy()
df_binary['Gender_Binary'] = df_binary['Gender'].astype('category').cat.codes.map(lambda x: bin(x)[2:])

# Program 30: Create a new feature by combining two existing columns
df_combined = df_num.copy()
df_combined['Salary_Per_Age'] = df_combined['Salary'] / df_combined['Age']

# Program 31: Extract year, month, and day from a date column
df_dates = df.copy()
df_dates['Year'] = df_dates['Date'].dt.year
df_dates['Month'] = df_dates['Date'].dt.month
df_dates['Day'] = df_dates['Date'].dt.day

# Program 32: Create a new feature using mathematical transformations
df_math = df_num.copy()
df_math['Age_Squared'] = np.square(df_math['Age'])

# Program 33: Apply Log Transformation to skewed data
df_log = df_num.copy()
df_log['Salary_Log'] = np.log1p(df_log['Salary'])

# Program 34: Perform Feature Selection using correlation analysis
corr_matrix = df_num.corr()
selected_features = corr_matrix[abs(corr_matrix['Salary']) > 0.1].index

# Program 35: Create a final preprocessed dataset ready for Machine Learning
df_final = df.copy()
df_final = df_final.dropna(thresh=len(df_final) * 0.5, axis=1)
df_final['Age'] = df_final['Age'].fillna(df_final['Age'].median())
df_final['Salary'] = df_final['Salary'].fillna(df_final['Salary'].median())
df_final['Gender'] = df_final['Gender'].fillna(df_final['Gender'].mode()[0])
df_final = pd.get_dummies(df_final, columns=['Gender'], drop_first=True)
scaler = StandardScaler()
df_final[['Age', 'Salary']] = scaler.fit_transform(df_final[['Age', 'Salary']])
df_final = df_final.drop(columns=['Date'])
