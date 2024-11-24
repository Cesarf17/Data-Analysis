import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# Define value mappings (keeping the previous mappings)
race_mapping = {
    b'1': 'Asian, non-Hispanic',
    b'2': 'American Indian/Alaska Native, non-Hispanic',
    b'3': 'Black, non-Hispanic',
    b'4': 'Hispanic, any race',
    b'5': 'White, non-Hispanic',
    b'6': 'Native Hawaiian/Pacific Islander, non-Hispanic',
    b'7': 'Multiple Race, non-Hispanic'
}

school_mapping = {
    b'1': 'Publicly controlled',
    b'2': 'Privately controlled',
    b'L': 'Logical Skip',
    b'M': 'Public/Private status not available'
}

gender_mapping = {
    b'F': 'Female',
    b'M': 'Male'
}

major_mapping = {
    b'1': 'Computer/Math Sciences',
    b'2': 'Bio/Agri/Env Sciences',
    b'3': 'Physical Sciences',
    b'4': 'Social Sciences',
    b'5': 'Engineering',
    b'6': 'S&E-Related Fields',
    b'7': 'Non-S&E Fields',
    b'8': 'Logical Skip'
}

# Read and clean data
df = pd.read_sas('epcg21.sas7bdat')
##df.to_parquet('epcg21.parquet')
columns_to_use = ['RACETHM', 'SALARY', 'STRTYR', 'BAACYR', 'BAPBPR', 'GENDER', 'NBAMEMG']
df_clean = df[columns_to_use].copy()

# Clean extreme values
df_clean = df_clean[df_clean['SALARY'] > 0]  # Remove zero salaries
df_clean = df_clean[df_clean['STRTYR'] < 9000]  # Remove invalid years
df_clean = df_clean[df_clean['BAACYR'] < 9000]  # Remove invalid years

# Map categorical variables to descriptions
df_clean['RACETHM'] = df_clean['RACETHM'].map(race_mapping)
df_clean['BAPBPR'] = df_clean['BAPBPR'].map(school_mapping)
df_clean['GENDER'] = df_clean['GENDER'].map(gender_mapping)
df_clean['NBAMEMG'] = df_clean['NBAMEMG'].map(major_mapping)

# Create log of salary for better distribution
df_clean['LOG_SALARY'] = np.log(df_clean['SALARY'])

# Set up plotting style - corrected version
sns.set_theme()  # Using seaborn's default styling
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Salary Distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=df_clean, x='SALARY', bins=50)
plt.title('Distribution of Salary')
plt.xlabel('Salary ($)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.histplot(data=df_clean, x='LOG_SALARY', bins=50)
plt.title('Distribution of Log Salary')
plt.xlabel('Log Salary')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# 2. Salary by Major Field
plt.figure(figsize=(15, 6))
sns.boxplot(data=df_clean, x='NBAMEMG', y='SALARY')
plt.xticks(rotation=45)
plt.title('Salary Distribution by Major Field')
plt.show()

# 3. Salary by Gender and School Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[df_clean['BAPBPR'].isin(['Publicly controlled', 'Privately controlled'])], 
            x='BAPBPR', y='SALARY', hue='GENDER')
plt.title('Salary Distribution by School Type and Gender')
plt.show()

# 4. Initial Regression Analysis
# Prepare data for regression
reg_data = df_clean[df_clean['BAPBPR'].isin(['Publicly controlled', 'Privately controlled'])].copy()

# Create dummy variables and make sure they're numeric
reg_data = pd.get_dummies(reg_data, 
                         columns=['RACETHM', 'BAPBPR', 'GENDER', 'NBAMEMG'],
                         drop_first=True)

# Convert year columns to numeric if they aren't already
reg_data['STRTYR'] = pd.to_numeric(reg_data['STRTYR'], errors='coerce')
reg_data['BAACYR'] = pd.to_numeric(reg_data['BAACYR'], errors='coerce')

# Drop any remaining missing values
reg_data = reg_data.dropna()

y = reg_data['LOG_SALARY']
X = reg_data.drop(['SALARY', 'LOG_SALARY'], axis=1)

X = sm.add_constant(X)

# Ensure all data is numeric
X = X.astype(float)
y = y.astype(float)

# Fit model
model = sm.OLS(y, X).fit()

# Print summary
print("\nRegression Results:")
print(model.summary().tables[1])

# Calculate and print R-squared
print(f"\nR-squared: {model.rsquared:.4f}")


# Save processed data
df_clean.columns = ['Race_Ethnicity', 'Salary', 'Start_Year', 'BA_Year', 'School_Type', 'Gender', 'Major_Field', 'Log_Salary']
df_clean.to_csv('salary_analysis_data_processed.csv', index=False)

# Print key findings
print("\nKey Statistics:")
print(f"Average Salary by Major Field:")
print(df_clean.groupby('Major_Field')['Salary'].mean().sort_values(ascending=False))

print("\nAverage Salary by Gender:")
print(df_clean.groupby('Gender')['Salary'].mean())

print("\nAverage Salary by School Type:")
print(df_clean.groupby('School_Type')['Salary'].mean())