"""
Nepal Trade Analysis -- Streamlit Dashboard
--------------------------------------------
Run this AFTER running the notebook (Section 14 exports the CSVs this app needs).

How to run:
    1. Put this file in the same folder as the `streamlit_data/` folder
       (created by the notebook's export cell).
    2. pip install streamlit plotly scikit-learn pandas numpy
    3. streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Nepal Trade Analysis", layout="wide")

DATA_DIR = "streamlit_data"


@st.cache_data
def load_data():
    nepal_trade = pd.read_csv(f"{DATA_DIR}/nepal_trade.csv")
    wb_trade = pd.read_csv(f"{DATA_DIR}/wb_trade_neighbors.csv")
    table2_clean = pd.read_csv(f"{DATA_DIR}/table2_clean.csv")
    table5_clean = pd.read_csv(f"{DATA_DIR}/table5_clean.csv")
    table6_clean = pd.read_csv(f"{DATA_DIR}/table6_clean.csv")
    ml_df = pd.read_csv(f"{DATA_DIR}/ml_df.csv")
    results_df = pd.read_csv(f"{DATA_DIR}/model_results.csv")
    return nepal_trade, wb_trade, table2_clean, table5_clean, table6_clean, ml_df, results_df


try:
    nepal_trade, wb_trade, table2_clean, table5_clean, table6_clean, ml_df, results_df = load_data()
except FileNotFoundError:
    st.error(
        "Couldn't find the data files. Run Section 14 of the notebook first -- "
        "it saves everything this app needs into a `streamlit_data/` folder "
        "that must sit next to this script."
    )
    st.stop()

feature_cols = [
    "year",
    "exports_lag1", "exports_lag2", "exports_lag3", "exports_roll3", "exports_growth_lag1",
    "imports_lag1", "imports_lag2", "imports_lag3", "imports_roll3", "imports_growth_lag1",
    "trade_balance_lag1",
]

# ----------------------------------------------------------------------
# Sidebar controls -- these are what make it genuinely "live"
# ----------------------------------------------------------------------
st.sidebar.title("Controls")

model_choice = st.sidebar.selectbox(
    "Forecasting model",
    ["Linear Regression", "Polynomial Regression", "Random Forest"],
    index=2,
)

horizon = st.sidebar.slider("Forecast years ahead", min_value=1, max_value=10, value=5)

if model_choice == "Random Forest":
    n_trees = st.sidebar.slider("Random Forest: number of trees", 50, 500, 300, step=50)
    max_depth = st.sidebar.slider("Random Forest: max depth", 2, 10, 4)
else:
    n_trees, max_depth = None, None

st.sidebar.markdown("---")
st.sidebar.caption(
    "Changing any control above **refits the model live** and redraws the "
    "forecast -- this isn't a pre-computed lookup."
)

# ----------------------------------------------------------------------
# Live model fitting + recursive forecasting
# ----------------------------------------------------------------------
def fit_model(model_name, X, y):
    if model_name == "Linear Regression":
        return LinearRegression().fit(X[feature_cols], y)
    elif model_name == "Polynomial Regression":
        return make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X[["year"]], y)
    elif model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42).fit(
            X[feature_cols], y
        )


def predict_one_year(model_name, model, year, history):
    if model_name == "Polynomial Regression":
        return model.predict(pd.DataFrame({"year": [year]}))[0]

    last3 = history.tail(3)
    row = {
        "year": year,
        "exports_lag1": history["exports_usd"].iloc[-1],
        "exports_lag2": history["exports_usd"].iloc[-2],
        "exports_lag3": history["exports_usd"].iloc[-3],
        "exports_roll3": last3["exports_usd"].mean(),
        "exports_growth_lag1": history["exports_usd"].iloc[-1] / history["exports_usd"].iloc[-2] - 1,
        "imports_lag1": history["imports_usd"].iloc[-1],
        "imports_lag2": history["imports_usd"].iloc[-2],
        "imports_lag3": history["imports_usd"].iloc[-3],
        "imports_roll3": last3["imports_usd"].mean(),
        "imports_growth_lag1": history["imports_usd"].iloc[-1] / history["imports_usd"].iloc[-2] - 1,
        "trade_balance_lag1": history["exports_usd"].iloc[-1] - history["imports_usd"].iloc[-1],
    }
    X_future = pd.DataFrame([row])[feature_cols]
    return model.predict(X_future)[0]


@st.cache_data(show_spinner=False)
def run_forecast(model_name, horizon, n_trees, max_depth):
    exp_model = fit_model(model_name, ml_df, ml_df["exports_usd"])
    imp_model = fit_model(model_name, ml_df, ml_df["imports_usd"])

    history = nepal_trade[["year", "exports_usd", "imports_usd"]].copy()
    last_year = history["year"].max()
    rows = []
    for step in range(1, horizon + 1):
        future_year = last_year + step
        pred_exp = predict_one_year(model_name, exp_model, future_year, history)
        pred_imp = predict_one_year(model_name, imp_model, future_year, history)
        rows.append({"year": future_year, "exports_usd": pred_exp, "imports_usd": pred_imp})
        history = pd.concat([history, pd.DataFrame([rows[-1]])], ignore_index=True)

    return pd.DataFrame(rows)


forecast_df = run_forecast(model_choice, horizon, n_trees, max_depth)

# ----------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------
st.title("Nepal Imports vs Exports -- Trade Analysis Dashboard")
st.caption("Data: World Bank (1965-2024) + Nepal Department of Customs (FY 2079/80)")

latest = nepal_trade.iloc[-1]
next_forecast = forecast_df.iloc[0]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Latest Exports", f"${latest['exports_usd']:,.0f}")
k2.metric("Latest Imports", f"${latest['imports_usd']:,.0f}")
k3.metric("Latest Trade Deficit", f"${latest['exports_usd'] - latest['imports_usd']:,.0f}")
k4.metric(f"Forecast Exports ({int(next_forecast['year'])})", f"${next_forecast['exports_usd']:,.0f}")
k5.metric(f"Forecast Imports ({int(next_forecast['year'])})", f"${next_forecast['imports_usd']:,.0f}")

st.markdown("---")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Historical Trend", "Trade Composition", "Trading Partners", "Neighbor Comparison", "Forecast"]
)

with tab1:
    fig1 = px.line(
        nepal_trade, x="year", y=["exports_usd", "imports_usd"],
        labels={"value": "US$", "year": "Year", "variable": "Type"},
        color_discrete_sequence=["#2E8B57", "#C0392B"],
        title="Nepal: Exports vs Imports (1965-2024)",
    )
    fig1.update_layout(hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.area(
        nepal_trade, x="year", y=nepal_trade["exports_usd"] - nepal_trade["imports_usd"],
        labels={"y": "Trade Balance (US$)", "year": "Year"},
        color_discrete_sequence=["#C0392B"],
        title="Trade Balance Over Time",
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    top_imports = table5_clean.sort_values("imports_npr_000", ascending=False).head(12)
    fig4 = px.bar(
        top_imports.sort_values("imports_npr_000"),
        x="imports_npr_000", y="description", orientation="h",
        color_discrete_sequence=["#C0392B"],
        title="Top 12 Import Commodities (FY 2079/80)",
    )
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    top_partners = table6_clean.sort_values("total_trade_npr_000", ascending=False).head(15) \
        if "total_trade_npr_000" in table6_clean.columns else table6_clean.head(15)
    fig5 = px.bar(
        top_partners.sort_values("total_trade_npr_000") if "total_trade_npr_000" in top_partners.columns else top_partners,
        x="total_trade_npr_000" if "total_trade_npr_000" in top_partners.columns else top_partners.columns[1],
        y="country", orientation="h",
        color_discrete_sequence=["#34495E"],
        title="Top 15 Trading Partners (FY 2079/80)",
    )
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    wb_sorted = wb_trade.sort_values(["country", "year"]).copy()
    wb_sorted["total_trade_usd"] = wb_sorted["exports_usd"] + wb_sorted["imports_usd"]
    wb_sorted["trade_growth_pct"] = wb_sorted.groupby("country")["total_trade_usd"].pct_change() * 100
    growth_plot_df = wb_sorted[wb_sorted["year"] >= 1990]

    fig6 = px.line(
        growth_plot_df, x="year", y="trade_growth_pct", color="country",
        labels={"trade_growth_pct": "YoY Total Trade Growth (%)"},
        title="South Asia: Year-over-Year Trade Growth Comparison",
    )
    fig6.add_hline(y=0, line_dash="dash", line_color="gray")
    fig6.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig6, use_container_width=True)

with tab5:
    st.subheader(f"{horizon}-Year Forecast -- {model_choice}")

    actual_plot = nepal_trade[["year", "exports_usd", "imports_usd"]].copy()
    actual_plot["type"] = "Actual"
    forecast_plot = forecast_df.copy()
    forecast_plot["type"] = "Forecast"
    combined = pd.concat([actual_plot, forecast_plot], ignore_index=True)

    fig7 = go.Figure()
    for col, name, color in [("exports_usd", "Exports", "#2E8B57"), ("imports_usd", "Imports", "#C0392B")]:
        a = combined[combined["type"] == "Actual"]
        f = combined[combined["type"] == "Forecast"]
        fig7.add_trace(go.Scatter(x=a["year"], y=a[col], mode="lines", name=f"{name} (actual)", line=dict(color=color)))
        fig7.add_trace(go.Scatter(x=f["year"], y=f[col], mode="lines+markers", name=f"{name} (forecast)",
                                   line=dict(color=color, dash="dash")))
    fig7.update_layout(title="Historical Actuals + Live Forecast", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig7, use_container_width=True)

    st.dataframe(forecast_df.style.format({"exports_usd": "${:,.0f}", "imports_usd": "${:,.0f}"}))

    st.markdown("#### Model comparison (from notebook evaluation)")
    st.dataframe(
        results_df.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R2": "{:.3f}"})
    )
