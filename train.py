
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

print("Loading dataset...")

df = pd.read_csv("dataset/train.csv")

# Prepare date and sales columns
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

df = df.dropna(subset=["date", "sales"])
df = df.sort_values("date")

# Aggregate sales by day
daily = df.groupby("date", as_index=False)["sales"].sum()
daily = daily.sort_values("date")

# Create past-sales features
daily["lag_1"] = daily["sales"].shift(1)
daily["lag_7"] = daily["sales"].shift(7)
daily["rolling_mean_7"] = daily["sales"].shift(1).rolling(7).mean()

daily = daily.dropna().reset_index(drop=True)

if len(daily) < 30:
    raise ValueError("Not enough daily data to train the model.")

X = daily[["lag_1", "lag_7", "rolling_mean_7"]]
y = daily["sales"]

# Keep the newest dates for testing
split = int(len(daily) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print("\nMODEL EVALUATION")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")

Path("artifacts").mkdir(exist_ok=True)

joblib.dump(model, "artifacts/sales_model.joblib")
joblib.dump(daily, "artifacts/daily_sales.joblib")

print("\nModel saved successfully!")
print("TRAINING COMPLETED SUCCESSFULLY!")