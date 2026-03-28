import traceback

typing_imports = """
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600)
def fetch_adj_close(ticker: str) -> pd.Series:
    """Fetch 10 years of daily adjusted close prices for the given ticker."""
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string.")

    ticker = ticker.strip().upper()
    if ticker == "":
        raise ValueError("Ticker must be a non-empty string.")

    horizon = "10y"
    try:
        data = yf.download(ticker, period=horizon, progress=False, auto_adjust=False)
    except Exception as err:
        raise RuntimeError(f"Data fetch failed for {ticker}: {err}")

    if data.empty or "Adj Close" not in data.columns:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    series = data["Adj Close"].dropna()
    if len(series) < 252 * 2:
        raise ValueError(f"Not enough historical data for ticker '{ticker}'.")

    return series


def compute_gbm_params(price_series: pd.Series, expected_annual_return: float) -> tuple[float, float, float]:
    """Compute daily returns, drift, and volatility for GBM."""
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    sigma = float(log_ret.std(ddof=1))
    historical_annual_volatility = sigma * np.sqrt(252)
    daily_drift = (expected_annual_return - 0.5 * historical_annual_volatility**2) / 252
    return float(price_series.iloc[-1]), daily_drift, sigma


def simulate_gbm(s0: float, drift: float, sigma: float, days: int, n_sims: int, enable_jumps: bool = False, lambda_j: float = 1.0, mu_j: float = -0.05, sigma_j: float = 0.10) -> np.ndarray:
    """Simulate GBM or Jump-Diffusion price paths: shape (n_sims, days + 1)."""
    np.random.seed(42)
    dt = 1.0
    increments = np.random.normal(loc=0.0, scale=1.0, size=(n_sims, days))
    exponent = drift * dt + sigma * np.sqrt(dt) * increments
    if enable_jumps:
        daily_lambda = lambda_j / 252
        num_jumps = np.random.poisson(daily_lambda, size=(n_sims, days))
        jump_sizes = np.random.normal(mu_j, sigma_j, size=(n_sims, days))
        total_jumps = num_jumps * jump_sizes
        exponent += total_jumps
    log_price = np.concatenate((np.zeros((n_sims, 1)), np.cumsum(exponent, axis=1)), axis=1)
    paths = s0 * np.exp(log_price)
    return paths


def create_plotly_plot(time_index: np.ndarray, sample_paths: np.ndarray, percentiles: dict, title: str) -> go.Figure:
    """Build the interactive Plotly figure for sample simulations and scenario bands."""
    fig = go.Figure()

    for idx in range(sample_paths.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=sample_paths[idx, :],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.25)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    color_map = {
        "p5": "firebrick",
        "p25": "#fb6a4a",
        "median": "royalblue",
        "p75": "#74c476",
        "p95": "darkgreen",
    }
    labels = {
        "p5": "5th percentile (Bear Case)",
        "p25": "25th Percentile (Lower Core)",
        "median": "50th percentile (Base Case)",
        "p75": "75th Percentile (Upper Core)",
        "p95": "95th percentile (Bull Case)",
    }

    for key, series in percentiles.items():
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=series,
                mode="lines",
                line=dict(color=color_map[key], width=3 if key in ["p5", "p95"] else 2),
                name=labels[key],
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Trading Day",
        yaxis_title="Price",
        template="plotly_white",
        font=dict(size=14),
        height=700,
        margin=dict(l=60, r=20, t=80, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def compute_terminal_metrics(final_prices: np.ndarray, s0: float, years: float) -> dict:
    median_price = float(np.median(final_prices))
    p95_price = float(np.percentile(final_prices, 95))
    p75_price = float(np.percentile(final_prices, 75))
    p25_price = float(np.percentile(final_prices, 25))
    p5_price = float(np.percentile(final_prices, 5))
    cagr = (median_price / s0) ** (1.0 / years) - 1.0
    return {
        "median_price": median_price,
        "p95_price": p95_price,
        "p75_price": p75_price,
        "p25_price": p25_price,
        "p5_price": p5_price,
        "cagr": cagr,
    }


def format_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def main() -> None:
    st.set_page_config(page_title="Institutional Monte Carlo Equity Forecaster", layout="centered")

    st.title("Institutional Monte Carlo Equity Forecasting")
    st.markdown(
        """Use Geometric Brownian Motion with 10,000 simulations to forecast equity price outcomes. """
    )

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker", "")
    with col2:
        years = st.selectbox("Forecast Horizon (years)", options=[1, 3, 5, 10, 20])

    use_capm = st.toggle("Auto-Calculate Return via CAPM", value=True)

    if use_capm:
        expected_annual_return = 0.08  # placeholder
    else:
        expected_annual_return = st.number_input("Expected Annual Return (Drift)", value=0.08, min_value=0.0, max_value=0.5, step=0.01, format="%.2f")

    enable_jumps = st.toggle("Enable Market Shocks (Jump-Diffusion)", value=False)

    if enable_jumps:
        shocks_per_decade = st.slider("Market Shocks per Decade", min_value=0, max_value=10, value=2)
        average_shock_size = st.slider("Average Shock Size", min_value=-50.0, max_value=0.0, value=-15.0, format="%.1f%%")
        lambda_j = shocks_per_decade / 10.0
        mu_j = average_shock_size / 100.0
    else:
        lambda_j = 1.0
        mu_j = -0.05

    col4, col5 = st.columns(2)
    with col4:
        adjust_inflation = st.toggle("Adjust for Inflation (Real Dollars)", value=False)
    with col5:
        inflation_rate = st.number_input("Expected Annual Inflation", value=0.025, step=0.005, format="%.3f")

    n_sims = 10000
    trading_days = int(252 * years)

    if ticker.strip():
        try:
            with st.spinner("Fetching historical data..."):
                historical_prices = fetch_adj_close(ticker)

            beta = 1.0
            if use_capm:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    beta = float(ticker_obj.info.get('beta', 1.0))
                except Exception:
                    beta = 1.0
                    st.warning("⚠️ Yahoo Finance rate limit reached. Defaulting to a Market Beta of 1.0 for CAPM.")

                risk_free_rate = 0.042
                market_risk_premium = 0.055
                capm_return = risk_free_rate + beta * market_risk_premium
                expected_annual_return = capm_return
                st.info(f"CAPM Expected Return: {capm_return:.2%}")

            s0, drift, sigma = compute_gbm_params(historical_prices, expected_annual_return)

            st.metric("Current Price", f"${s0:,.2f}")

            with st.spinner("Running Monte Carlo simulations (10,000 runs)..."):
                sim_paths = simulate_gbm(s0, drift, sigma, trading_days, n_sims, enable_jumps, lambda_j, mu_j)

            if adjust_inflation:
                time_in_years = np.arange(trading_days + 1) / 252.0
                discount_factor = (1 + inflation_rate) ** time_in_years
                sim_paths = sim_paths / discount_factor

            sample_indices = np.random.choice(n_sims, size=100, replace=False)
            sample_paths = sim_paths[sample_indices, :]

            p5 = np.percentile(sim_paths, 5, axis=0)
            p25 = np.percentile(sim_paths, 25, axis=0)
            med = np.percentile(sim_paths, 50, axis=0)
            p75 = np.percentile(sim_paths, 75, axis=0)
            p95 = np.percentile(sim_paths, 95, axis=0)

            time_index = np.arange(0, trading_days + 1)
            title = "Monte Carlo GBM Forecast (Real Dollars)" if adjust_inflation else "Monte Carlo GBM Forecast (Nominal Dollars)"
            fig = create_plotly_plot(time_index, sample_paths, {"p95": p95, "p75": p75, "median": med, "p25": p25, "p5": p5}, title)

            st.plotly_chart(fig, use_container_width=True)

            final_prices = sim_paths[:, -1]
            metrics = compute_terminal_metrics(final_prices, s0, float(years))

            # Row 1: 95th Percentile (Bull), 75th Percentile (Upper Core), Expected Final Price (Median)
            col1, col2, col3 = st.columns(3)
            col1.metric("95th Percentile (Bull)", f"${metrics['p95_price']:,.2f}")
            col2.metric("75th Percentile (Upper Core)", f"${metrics['p75_price']:,.2f}")
            col3.metric("Expected Final Price (Median)", f"${metrics['median_price']:,.2f}")

            st.divider()

            # Row 2: 25th Percentile (Lower Core), 5th Percentile (Bear), Expected CAGR
            col4, col5, col6 = st.columns(3)
            col4.metric("25th Percentile (Lower Core)", f"${metrics['p25_price']:,.2f}")
            col5.metric("5th Percentile (Bear)", f"${metrics['p5_price']:,.2f}")
            col6.metric("Expected CAGR", format_pct(metrics['cagr']))

        except Exception as error:
            st.error(f"Error: {error}")
            st.error("Please check the ticker symbol and try again.")
            st.text(traceback.format_exc())
            return
    else:
        st.info("👈 Please enter a valid stock ticker (e.g., AAPL, VOO, NVDA) to run the institutional stress-test.")


if __name__ == "__main__":
    main()
