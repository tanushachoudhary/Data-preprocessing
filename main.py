# %%
!pip install pandas scikit-learn

# %%
import pandas as pd

# %%
# Load dataset from a CSV file
df = pd.read_csv("SocialMediaUsers.csv")

# %%
# View first few rows
print(df.head())

# %%
# Get shape (rows, columns)
print(df.shape)

# %%
# Summary statistics
print(df.describe())

# %%
# Data types
print(df.dtypes)

# %%
# Check for missing values
print(df.isnull().sum())

# %%
#Drop rows with any missing values
df = df.dropna()

# %%
# Binary encoding for Verified Account
df['Verified Account'] = df['Verified Account'].map({'Yes': 1, 'No': 0})

# %%
# Convert Date Joined to datetime and extract useful features
df['Date Joined'] = pd.to_datetime(df['Date Joined'])
df['Year Joined'] = df['Date Joined'].dt.year
df['Month Joined'] = df['Date Joined'].dt.month

# %%
# Drop the original Date Joined column
df = df.drop('Date Joined', axis=1)

# %%
# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Platform', 'Owner', 'Primary Usage', 'Country'])

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Daily Time Spent (min)', 'Year Joined', 'Month Joined']] = scaler.fit_transform(
    df[['Daily Time Spent (min)', 'Year Joined', 'Month Joined']]
)

# %%
print(df.head())
print(df.shape)


