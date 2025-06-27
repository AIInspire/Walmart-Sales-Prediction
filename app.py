import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(page_title="Weekly Sales Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🛒 Weekly Sales Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for input configuration
st.sidebar.header("📊 Prediction Settings")
plot_weeks_back = st.sidebar.slider("Weeks of Historical Data to Show", min_value=1, max_value=12, value=4)
predict_weeks_ahead = st.sidebar.slider("Weeks to Predict Ahead", min_value=1, max_value=8, value=2)
show_confidence_interval = st.sidebar.checkbox("Show Confidence Interval", value=True)

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Input Parameters")
    
    # Group inputs logically
    st.subheader("📈 Economic Indicators")
    inputs = {}
    inputs['Temperature'] = st.number_input("Temperature (°F)", value=50.0, help="Average temperature for the week")
    inputs['Fuel_Price'] = st.number_input("Fuel Price ($)", value=2.5, help="Average fuel price per gallon")
    inputs['CPI'] = st.number_input("Consumer Price Index", value=211.0, help="Economic indicator")
    inputs['Unemployment'] = st.number_input("Unemployment Rate (%)", value=8.0, help="Regional unemployment rate")
    
    st.subheader("📅 Time Information")
    inputs['week'] = st.number_input("Week Number", value=12, min_value=1, max_value=52)
    inputs['Month'] = st.number_input("Month", value=3, min_value=1, max_value=12)
    inputs['Year'] = st.number_input("Year", value=2010, min_value=2010, max_value=2025)
    
    st.subheader("📊 Historical Sales Data")
    inputs['Weekly_Sales_Lag_1'] = st.number_input("Sales 1 Week Ago ($)", value=45000.0, help="Sales from previous week")
    inputs['Weekly_Sales_Lag_2'] = st.number_input("Sales 2 Weeks Ago ($)", value=43000.0, help="Sales from 2 weeks ago")
    inputs['Weekly_Sales_Lag_3'] = st.number_input("Sales 3 Weeks Ago ($)", value=41000.0, help="Sales from 3 weeks ago")
    
    st.subheader("📈 Trend Indicators")
    inputs['Weekly_Sales_Rolling_Mean'] = st.number_input("Rolling Mean Sales ($)", value=42000.0, help="Average of recent weeks")
    inputs['Weekly_Sales_Rolling_Std'] = st.number_input("Rolling Std Sales ($)", value=2000.0, help="Standard deviation of recent weeks")
    inputs['Weekly_Sales_Cumulative_Sum'] = st.number_input("Cumulative Sales ($)", value=500000.0, help="Year-to-date sales")

with col2:
    st.header("📊 Sales Prediction & Visualization")
    
    if st.button("🔍 Predict Weekly Sales", type="primary"):
        try:
            # Make prediction request
            response = requests.post("http://127.0.0.1:8000/predict", json=inputs)
            
            if response.status_code == 200:
                result = response.json()
                predicted_sales = result['Predicted_Weekly_Sales']
                
                # Display prediction with styling
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric("📈 Predicted Sales", f"${predicted_sales:,.0f}")
                
                with col_metric2:
                    # Calculate change from last week
                    change = predicted_sales - inputs['Weekly_Sales_Lag_1']
                    st.metric("📊 Change from Last Week", f"${change:,.0f}", f"{change/inputs['Weekly_Sales_Lag_1']*100:.1f}%")
                
                with col_metric3:
                    # Calculate performance vs rolling mean
                    vs_mean = predicted_sales - inputs['Weekly_Sales_Rolling_Mean']
                    st.metric("📋 vs Recent Average", f"${vs_mean:,.0f}", f"{vs_mean/inputs['Weekly_Sales_Rolling_Mean']*100:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create historical data for plotting
                # Calculate current date based on year, month, and week
                year_start = datetime(inputs['Year'], 1, 1)
                # Find the first Monday of the year (ISO week standard)
                days_to_monday = (7 - year_start.weekday()) % 7
                first_monday = year_start + timedelta(days=days_to_monday)
                # Calculate current date based on week number
                current_date = first_monday + timedelta(weeks=inputs['week']-1)
                
                # Generate historical dates
                historical_dates = []
                historical_sales = []
                
                for i in range(plot_weeks_back, 0, -1):
                    date = current_date - timedelta(weeks=i)
                    historical_dates.append(date)
                    
                    # Use lag values for recent weeks, simulate older data
                    if i == 1:
                        historical_sales.append(inputs['Weekly_Sales_Lag_1'])
                    elif i == 2:
                        historical_sales.append(inputs['Weekly_Sales_Lag_2'])
                    elif i == 3:
                        historical_sales.append(inputs['Weekly_Sales_Lag_3'])
                    else:
                        # Simulate historical data based on trend
                        base_value = inputs['Weekly_Sales_Rolling_Mean']
                        noise = np.random.normal(0, inputs['Weekly_Sales_Rolling_Std'] * 0.5)
                        trend = (inputs['Weekly_Sales_Lag_1'] - inputs['Weekly_Sales_Lag_3']) / 2 * (i-1)
                        historical_sales.append(base_value - trend + noise)
                
                # Generate future dates and predictions
                future_dates = []
                future_predictions = []
                future_lower = []
                future_upper = []
                
                for i in range(predict_weeks_ahead):
                    future_date = current_date + timedelta(weeks=i+1)
                    future_dates.append(future_date)
                    
                    # For first prediction, use API result
                    if i == 0:
                        prediction = predicted_sales
                    else:
                        # Simulate future predictions with slight trend continuation
                        trend = (predicted_sales - inputs['Weekly_Sales_Lag_1']) * 0.8  # Damped trend
                        prediction = predicted_sales + trend * i
                    
                    future_predictions.append(prediction)
                    
                    # Add confidence intervals
                    std_error = inputs['Weekly_Sales_Rolling_Std'] * (1 + i * 0.1)  # Increasing uncertainty
                    future_lower.append(prediction - 1.96 * std_error)
                    future_upper.append(prediction + 1.96 * std_error)
                
                # Create the plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_sales,
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>'
                ))
                
                # Current week (connecting point)
                fig.add_trace(go.Scatter(
                    x=[current_date],
                    y=[inputs['Weekly_Sales_Lag_1']],
                    mode='markers',
                    name='Current Week',
                    marker=dict(size=12, color='#2ca02c', symbol='diamond'),
                    hovertemplate='<b>Current Week</b><br><b>Sales:</b> $%{y:,.0f}<extra></extra>'
                ))
                
                # Predicted data
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    marker=dict(size=10, symbol='star'),
                    hovertemplate='<b>Predicted Date:</b> %{x}<br><b>Predicted Sales:</b> $%{y:,.0f}<extra></extra>'
                ))
                
                # Confidence interval
                if show_confidence_interval:
                    fig.add_trace(go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=future_upper + future_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval',
                        hoverinfo='skip'
                    ))
                
                # # Add vertical line to separate historical from predictions
                # fig.add_vline(
                #     x=current_date.strftime('%Y-%m-%d'),
                #     line_dash="dot",
                #     line_color="gray",
                #     annotation_text="Current Week",
                #     annotation_position="top"
                # )
                
                # Update layout
                fig.update_layout(
                    title=f'Sales Forecast: {plot_weeks_back} Weeks History + {predict_weeks_ahead} Weeks Prediction',
                    xaxis_title='Date',
                    yaxis_title='Weekly Sales ($)',
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Format y-axis as currency
                fig.update_yaxes(tickformat='$,.0f')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional insights
                st.subheader("📋 Prediction Insights")
                
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    st.write("**Trend Analysis:**")
                    if predicted_sales > inputs['Weekly_Sales_Lag_1']:
                        st.success("📈 Sales are trending upward")
                    elif predicted_sales < inputs['Weekly_Sales_Lag_1']:
                        st.warning("📉 Sales are trending downward")
                    else:
                        st.info("➡️ Sales are stable")
                
                with col_insight2:
                    st.write("**Performance vs Average:**")
                    if predicted_sales > inputs['Weekly_Sales_Rolling_Mean']:
                        st.success("🎯 Above recent average performance")
                    else:
                        st.warning("⚠️ Below recent average performance")
                
                # Data table
                with st.expander("📊 View Detailed Data"):
                    # Combine all data
                    all_dates = historical_dates + [current_date] + future_dates
                    all_sales = historical_sales + [inputs['Weekly_Sales_Lag_1']] + future_predictions
                    all_types = ['Historical'] * len(historical_dates) + ['Current'] + ['Predicted'] * len(future_dates)
                    
                    df_display = pd.DataFrame({
                        'Date': all_dates,
                        'Sales': all_sales,
                        'Type': all_types
                    })
                    df_display['Sales'] = df_display['Sales'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(df_display, use_container_width=True)
                
            else:
                st.error("❌ Failed to get prediction. Please check your API connection and try again.")
                st.write(f"Status Code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to the prediction API. Make sure your FastAPI server is running on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    else:
        # Show placeholder chart when no prediction is made
        st.info("👆 Click 'Predict Weekly Sales' to see the forecast visualization")
        
        # Create a sample chart to show what it will look like
        sample_dates = pd.date_range(start='2024-01-01', periods=10, freq='W')
        sample_sales = np.random.normal(45000, 5000, 10)
        
        fig_sample = go.Figure()
        fig_sample.add_trace(go.Scatter(
            x=sample_dates[:6],
            y=sample_sales[:6],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4')
        ))
        fig_sample.add_trace(go.Scatter(
            x=sample_dates[5:],
            y=sample_sales[5:],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        fig_sample.update_layout(
            title='Sample Sales Forecast (Your prediction will appear here)',
            xaxis_title='Date',
            yaxis_title='Weekly Sales ($)',
            height=400
        )
        fig_sample.update_yaxes(tickformat='$,.0f')
        
        st.plotly_chart(fig_sample, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** Adjust the historical weeks and prediction period in the sidebar to customize your view.")