
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📈")

st.title("📈 Retail Sales Forecasting")
st.write("Machine Learning tho future daily sales ni forecast cheddham.")

model_path = Path("artifacts/sales_model.joblib")
data_path = Path("artifacts/daily_sales.joblib")

if not model_path.exists() or not data_path.exists():
    st.error("Model files dorakaledhu. Mundhu python train.py run cheyyi.")
    st.stop()

model = joblib.load(model_path)
daily = joblib.load(data_path)

daily = daily.sort_values("date").reset_index(drop=True)

st.subheader("Historical Sales")
st.line_chart(daily.set_index("date")["sales"])

st.subheader("Future Sales Prediction")

days = st.number_input(
    "Enni rojula sales predict cheyyali?",
    min_value=1,
    max_value=30,
    value=7
)

if st.button("Predict Future Sales"):
    history = daily["sales"].astype(float).tolist()
    future_dates = []
    future_sales = []

    last_date = pd.Timestamp(daily["date"].max())

    for i in range(1, int(days) + 1):
        lag_1 = history[-1]
        lag_7 = history[-7] if len(history) >= 7 else history[-1]
        rolling_mean_7 = sum(history[-7:]) / min(7, len(history))

        features = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7
        }])

        prediction = max(0.0, float(model.predict(features)[0]))
        history.append(prediction)

        future_dates.append(last_date + pd.Timedelta(days=i))
        future_sales.append(prediction)

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": future_sales
    })

    st.success("Forecast complete!")
    st.dataframe(forecast, use_container_width=True)

    st.metric(
        "Total predicted sales",
        f"{sum(future_sales):,.2f}"
    )

    st.line_chart(forecast.set_index("Date")["Predicted Sales"])

    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast CSV",
        data=csv,
        file_name="retail_sales_forecast.csv",
        mime="text/csv"
    )

st.caption("Educational forecasting project. Predictions are estimates, not guaranteed sales.")