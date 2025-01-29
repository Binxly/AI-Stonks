#    _    ___      ____  _____ ___  _   _ _  __ ____
#   / \  |_ _|    / ___||_   _/ _ \| \ | | |/ // ___|
#  / _ \  | |     \___ \  | || | | |  \| | ' /\___ \
# / ___ \ | |      ___) | | || |_| | |\  | . \ ___) |
#/_/   \_\___|    |____/  |_| \___/|_| \_|_|\_\____/
#
# An AI-Powered Technical Analysis Stock Dashboard in Python with Streamlit and Ollama
#
# Reference code credits to substack user: deepcharts
# Post: "AI-Powered Technical Analysis Stock Dashboard in Python with Streamlit and Ollama"
# URL: https://deepcharts.substack.com/p/build-an-ai-powered-technical-analysis

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("The ChatGippity Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., NVDA):", "NVDA")
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp.now() - pd.DateOffset(years=1))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.now())

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    st.session_state["stock_data"] = yf.download(ticker, start=start_date, end=end_date)
    st.success("Stock data loaded successfully!")

# Check if data is available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    # Plot candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick",
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350'
        )
    ])

    # Sidebar: Select technical indicators
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "MACD", "RSI"],
        default=["20-Day SMA"]
    )

    def calculate_rsi(data, window=14):
        """Calculate RSI for the given data."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(data):
        """Calculate MACD and signal line."""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    # Helper function to add indicators to the chart
    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
        elif indicator == "RSI":
            rsi = calculate_rsi(data)
            fig.add_trace(go.Scatter(x=data.index, y=rsi, mode='lines', 
                                   name='RSI', yaxis='y2'))
            
            fig.add_trace(go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[70, 70],
                mode='lines',
                line=dict(dash='dash', color='red', width=1),
                name='Overbought',
                yaxis='y2'
            ))
            
            fig.add_trace(go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[30, 30],
                mode='lines',
                line=dict(dash='dash', color='green', width=1),
                name='Oversold',
                yaxis='y2'
            ))
        elif indicator == "MACD":
            macd, signal = calculate_macd(data)
            fig.add_trace(go.Scatter(x=data.index, y=macd, mode='lines', 
                                   name='MACD', yaxis='y3'))
            fig.add_trace(go.Scatter(x=data.index, y=signal, mode='lines', 
                                   name='Signal', yaxis='y3'))

    # Add selected indicators to the chart
    for indicator in indicators:
        add_indicator(indicator)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis=dict(gridcolor='#31333F'),
        xaxis=dict(gridcolor='#31333F')
    )

    # Update layout based on selected indicators
    layout_update = {
        'height': 800,
        'yaxis': dict(domain=[0.6, 1]),
        'title': f"{ticker} Performance Chart ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"
    }

    if "RSI" in indicators and "MACD" in indicators:
        layout_update.update({
            'height': 1000,
            'yaxis2': dict(title="RSI", domain=[0.35, 0.55]),
            'yaxis3': dict(title="MACD", domain=[0.1, 0.3])
        })
    elif "RSI" in indicators:
        layout_update.update({
            'yaxis2': dict(title="RSI", domain=[0.1, 0.3])
        })
    elif "MACD" in indicators:
        layout_update.update({
            'yaxis3': dict(title="MACD", domain=[0.1, 0.3])
        })

    fig.update_layout(**layout_update)

    st.plotly_chart(fig)

    # Analyze chart with LLaMA 3.2 Vision
    st.subheader("AI-Powered Analysis")
    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing the chart, please wait..."):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.write_image(tmpfile.name)
                tmpfile_path = tmpfile.name

            with open(tmpfile_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            tech_data = ""
            if "RSI" in indicators:
                rsi = calculate_rsi(data)
                recent_rsi = rsi.tail(10)
                rsi_data = "\nRecent RSI values:"
                for date, value in recent_rsi.items():
                    rsi_data += f"\n{date.strftime('%Y-%m-%d')}: {value:.2f}"
                tech_data += rsi_data

            if "MACD" in indicators:
                macd, signal = calculate_macd(data)
                recent_data = pd.DataFrame({
                    'MACD': macd.tail(10),
                    'Signal': signal.tail(10)
                })
                macd_data = "\n\nRecent MACD values:"
                for date, row in recent_data.iterrows():
                    macd_data += f"\n{date.strftime('%Y-%m-%d')}: MACD={row['MACD']:.2f}, Signal={row['Signal']:.2f}"
                tech_data += macd_data

            messages = [{
                'role': 'user',
                'content': f"""You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                            Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                            Base your recommendation only on the candlestick chart and the displayed technical indicators.
                            First, provide the recommendation, then, provide your detailed reasoning.
                            
                            Additional Technical Data:{tech_data}
                """,
                'images': [image_data]
            }]
            response = ollama.chat(model='llama3.2-vision', messages=messages)

            st.write("**AI Analysis Results:**")
            st.write(response["message"]["content"])

            os.remove(tmpfile_path)
