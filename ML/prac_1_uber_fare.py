import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("uber.csv")
print(df.head())
print(df.isnull().sum())

df.drop(columns=['Unnamed: 0', 'key'], errors='ignore', inplace=True)
df.dropna(inplace=True)
df['pickup_datetime'] = pd.to_datetime(
    df['pickup_datetime'], errors='coerce', utc=True
)
df.dropna(subset=['pickup_datetime'], inplace=True)
df = df[
    (df['fare_amount'] > 0) &
    (df['passenger_count'].between(1, 6)) &
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_longitude'].between(-180, 180)) &
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_latitude'].between(-90, 90)) &
    (df['pickup_longitude'] != 0) &
    (df['dropoff_longitude'] != 0) &
    (df['pickup_latitude'] != 0) &
    (df['dropoff_latitude'] != 0)
].copy()
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
lat1 = np.radians(df['pickup_latitude'])
lat2 = np.radians(df['dropoff_latitude'])
lon1 = np.radians(df['pickup_longitude'])
lon2 = np.radians(df['dropoff_longitude'])
a = (np.sin((lat2-lat1)/2)**2 +
   np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)
df['distance'] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

sns.boxplot(x=df['fare_amount'])
plt.title("Fare Outliers")
plt.show()
Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[
  df['fare_amount'].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
].copy()
print("Dataset shape after preprocessing:", df.shape)

features = [
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'passenger_count', 'hour', 'day', 'month', 'distance'
]
plt.figure(figsize=(10, 6))
sns.heatmap(df[features + ['fare_amount']].corr(),
         annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

X = df[features]
y = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R2 Score': [
         r2_score(y_test, pred_lr),
         r2_score(y_test, pred_rf)
    ],
    'RMSE': [
         np.sqrt(mean_squared_error(y_test, pred_lr)),
         np.sqrt(mean_squared_error(y_test, pred_rf))
    ],
    'MAE': [
         mean_absolute_error(y_test, pred_lr),
         mean_absolute_error(y_test, pred_rf)
    ]
})
print("\nModel Comparison:")
print(results)

print("\nModel with higher R2 and lower RMSE/MAE performs better.")
