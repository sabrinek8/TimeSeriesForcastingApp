import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from predicthq import Client  # Import PredictHQ client

st.title('Streaming Forecast App')

# File selection for multiple parquet files
uploaded_files = st.file_uploader("Upload Parquet files", type=["parquet"], accept_multiple_files=True)

if uploaded_files:
    # List to store individual dataframes
    df_list = []

    # Read each uploaded parquet file and append to df_list
    for file in uploaded_files:
        df = pd.read_parquet(file)
        # Convert 'ts' column to datetime if it's in Unix timestamp format
        if pd.api.types.is_numeric_dtype(df['ts']):
            df['ts'] = pd.to_datetime(df['ts'], unit='s')
        df_list.append(df)

    # Concatenate all dataframes into one
    data = pd.concat(df_list, ignore_index=True)

    # Sort by 'ts' column to ensure chronological order
    data.sort_values(by='ts', inplace=True)
    
    st.subheader('Combined and Sorted Data')

    # Display combined and sorted data
    st.write(data)

    # Plot raw data
    def plot_raw_data():
        for col in data.columns:
            if col != 'ts' and pd.api.types.is_numeric_dtype(data[col]):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['ts'], y=data[col], name=col))
                fig.layout.update(title_text=f'Time Series data for {col}', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig)

    plot_raw_data()

    st.subheader('Forecast data')

    # Slider for the number of years for prediction
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    combined_forecast_df = pd.DataFrame()  # DataFrame to store combined forecasts

    # Iterate through numerical columns (assuming 'ts' and numerical columns)
    for col in data.columns:
        if col != 'ts' and pd.api.types.is_numeric_dtype(data[col]):
            df_train = data[['ts', col]].rename(columns={"ts": "ds", col: "y"})

            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            # Filter to get only the new forecasted values
            forecast_new = forecast[forecast['ds'] > data['ts'].max()]

            # Collect forecasted values
            forecast_col = forecast_new[['ds', 'yhat']].rename(columns={'ds': 'ts', 'yhat': f'Forecast_{col}'})
            if combined_forecast_df.empty:
                combined_forecast_df = forecast_col
            else:
                combined_forecast_df = pd.merge(combined_forecast_df, forecast_col, on='ts')

            st.write(f'Forecast plot for {col}')
            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig)

            st.write(f"Forecast components for {col}")
            fig_components = m.plot_components(forecast)
            st.write(fig_components)

    # Save the combined forecast dataframe to a CSV file
    combined_forecast_df.to_csv('combined_forecast.csv', index=False)
    st.write("Forecasts saved to CSV file")

    st.subheader('PredictHQ Events')

    # Initialize PredictHQ client with your access token
    phq = Client(access_token="**********************************")

    # Get the first and last dates from the combined forecast CSV
    if not combined_forecast_df.empty:
        gte_date = combined_forecast_df['ts'].iloc[0].strftime('%Y-%m-%d')
        lte_date = combined_forecast_df['ts'].iloc[-1].strftime('%Y-%m-%d')
        categories = ["sports","public-holidays"]
        # Example query: Fetching sports events within the date range of combined forecast
        events = phq.events.search(category="public-holidays", rank={'gte': 90, 'lt': 100}, active={
            'lte':f'{lte_date}',
            'gte':f'{gte_date}',
            'tz': 'UTC'
        })
        sports = phq.events.search(category="sports", rank={'gte': 90, 'lt': 100}, active={
            'lte':f'{lte_date}',
            'gte':f'{gte_date}',
            'tz': 'UTC'
        })

         # Collect events in a DataFrame
        events_list = []
        for event in events:
            events_list.append({
                "Title": event.title,
                "Category": event.category,
                "Rank": event.rank,
                "Start Date": event.start.strftime('%Y-%m-%d'),
                "End Date": event.end.strftime('%Y-%m-%d')
            })
        for event in sports:
            events_list.append({
                "Title": event.title,
                "Category": event.category,
                "Rank": event.rank,
                "Start Date": event.start.strftime('%Y-%m-%d'),
                "End Date": event.end.strftime('%Y-%m-%d')
            })

        events_df = pd.DataFrame(events_list)

        # Display events DataFrame
        st.subheader('Events DataFrame')
        st.write(events_df)

        # Apply 10% increase to forecast values when dates match
        for event_date in events_df['Start Date']:
            combined_forecast_df.loc[combined_forecast_df['ts'] == event_date, combined_forecast_df.columns != 'ts'] *= 1.10

        st.subheader('Adjusted Forecast DataFrame')
        st.write(combined_forecast_df)

        # Save the adjusted combined forecast dataframe to a new CSV file
        combined_forecast_df.to_csv('adjusted_combined_forecast.csv', index=False)
        st.write("Adjusted forecasts saved to CSV file")

        # Plotting the adjusted combined forecast
        fig = go.Figure()

        for column in combined_forecast_df.columns:
            if column != 'ts':
                 fig.add_trace(go.Scatter(x=combined_forecast_df['ts'], y=combined_forecast_df[column], mode='lines+markers', name=column))

                 fig.update_layout(
                     title='Adjusted Combined Forecast',
                     xaxis_title='Date',
                     yaxis_title='Forecast Values',
                     hovermode='x'
                 )

        st.plotly_chart(fig)
