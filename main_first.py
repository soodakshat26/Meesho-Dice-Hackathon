import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy.stats import norm

# Function to preprocess input data according to the original model training process
def preprocess_input_data(input_data):
    if 'Date' in input_data.columns:
        input_data['Date'] = pd.to_datetime(input_data['Date'])

    # Extract new features from 'Date' if this was done in training
    input_data['DayOfMonth'] = input_data['Date'].dt.day
    input_data['WeekOfYear'] = input_data['Date'].dt.isocalendar().week

    # One-hot encoding or label encoding of categorical variables
    categorical_columns = ['DayOfWeek', 'Month', 'Season', 'Event', 'ProductID', 
                           'ProductCategory', 'CustomerGender', 'CustomerLocation', 
                           'CustomerIncome', 'CustomerSegment']

    # Apply One-Hot Encoding as per the model training process
    input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

    # Fill any missing columns that might be missing after encoding (with 0 or appropriate values)
    model_columns = ['DayOfWeek', 'Month', 'Season', 'Year', 'Price', 'DiscountRate', 'InventoryLevel', 'ProductRating', 
                     'ReviewCount', 'Revenue', 'MarketingSpend', 'CompetitorPrice', 'InflationRate', 
                     'ConsumerConfidenceIndex', 'Temperature', 'Rainfall', 'CustomerAge', 'CustomerGender', 
                     'CustomerLocation', 'DayOfMonth', 'WeekOfYear', 'IsWeekend', 'IsEventDay', 'SalesVolume_Lag1', 
                     'SalesVolume_Lag7', 'SalesVolume_Lag14', 'SalesVolume_RollingMean_7', 'SalesVolume_RollingStd_7', 
                     'SalesVolume_RollingMean_14', 'SalesVolume_RollingStd_14', 'ProductCategory_Beauty Products', 
                     'ProductCategory_Electronics', 'ProductCategory_Fashion', 'ProductCategory_Footwear', 
                     'ProductCategory_Grocery', 'ProductCategory_Home Decor', 'ProductCategory_Jewelry', 
                     'ProductCategory_Kitchen Appliances', 'ProductCategory_Sports Equipment', 'ProductCategory_Toys', 
                     'ProductCategory_Traditional Wear', 'CustomerSegment_Frequent Buyer', 'CustomerSegment_Occasional Buyer', 
                     'CustomerSegment_Occasional Buyer ', 'Event_FSMS', 'Event_MBS', 'Event_MDS', 'CustomerIncome_Encoded', 
                     'Price_DiscountRate', 'Marketing_CompetitorPrice', 'InflationRate_Lag1', 'ConsumerConfidenceIndex_Lag1']

    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill missing columns with zeros or appropriate default values

    # Ensure the order of columns matches the training data
    input_data = input_data[model_columns]

    return input_data

# Load Data
@st.cache_data
def load_data():
    sales_data = pd.read_csv('sales.csv')
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    return sales_data

# Load the XGBoost Model
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Predict using XGBoost model
def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    return predictions

# Calculate Safety Stock
def calculate_safety_stock(predictions, service_level=0.95):
    std_forecast_error = np.std(predictions)
    z_score = norm.ppf(service_level)
    safety_stock = z_score * std_forecast_error
    return safety_stock

# Calculate key metrics
def calculate_metrics(data):
    # Ensure there are no divisions by zero
    data['Forecasted Sales Volume'] = data['Forecasted Sales Volume'].replace(0, np.nan)

    # Average Daily Sales (ADS)
    ads = data.groupby(['ProductID'])['Forecasted Sales Volume'].mean().reset_index()
    ads.columns = ['ProductID', 'Average Daily Sales (ADS)']

    # Revenue Per Unit (RPU)
    data['Revenue Per Unit (RPU)'] = data['Revenue'] / data['Forecasted Sales Volume']

    # Event Sales Multiplier calculation
    event_sales = data[data['Event Indicator'] != 'BAU'].groupby('Event Indicator')['Forecasted Sales Volume'].sum()
    bau_sales = data[data['Event Indicator'] == 'BAU']['Forecasted Sales Volume'].mean()

    # Handle case where bau_sales might be NaN due to no BAU data
    if np.isnan(bau_sales):
        event_multiplier = pd.DataFrame({'Event Indicator': event_sales.index, 'Event Sales Multiplier': [np.nan] * len(event_sales)})
    else:
        event_multiplier = (event_sales / bau_sales).reset_index()
        event_multiplier.columns = ['Event Indicator', 'Event Sales Multiplier']

    return ads, event_multiplier

# Load sales data and model
sales_data = load_data()
xgb_model = load_model()

# Streamlit App Layout
st.title('Inventory Projection and Analysis')

st.sidebar.header('Navigation')
section = st.sidebar.selectbox('Choose a section', ['Data Overview', 'Model Predictions', 'Visualizations'])

# Section: Data Overview
if section == 'Data Overview':
    st.header('Data Overview')
    st.write('### Sales Data')
    st.dataframe(sales_data)

# Section: Model Predictions
if section == 'Model Predictions':
    st.header('Model Predictions')
    st.write('### Upload Your Data for Predictions')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write('### Input Data')
        st.dataframe(input_data)

        # Preprocess input data to ensure correct data types and features
        input_data_prepared = preprocess_input_data(input_data)

        # Make predictions
        predictions = make_predictions(xgb_model, input_data_prepared)
        input_data['Forecasted Sales Volume'] = predictions

        # Calculate Safety Stock and add to DataFrame
        safety_stock = calculate_safety_stock(predictions)
        input_data['Safety Stock'] = safety_stock

        # Define the event days explicitly for each event
        fsms_days = pd.to_datetime(['2023-12-02'])
        mbs_days = pd.to_datetime(['2023-12-03'])
        mds_days = pd.to_datetime(['2023-12-24'])

        # Create 'Event Indicator' column
        input_data['Event Indicator'] = 'BAU'  # Default to 'BAU'
        input_data.loc[input_data['Date'].isin(fsms_days), 'Event Indicator'] = 'FSMS'
        input_data.loc[input_data['Date'].isin(mbs_days), 'Event Indicator'] = 'MBS'
        input_data.loc[input_data['Date'].isin(mds_days), 'Event Indicator'] = 'MDS'

        # Calculate key metrics
        ads, event_multiplier = calculate_metrics(input_data)

        st.write('### Predictions')
        st.dataframe(input_data)

        # Display Key Metrics
        st.write('### Key Metrics')
        st.write('#### Average Daily Sales (ADS)')
        st.dataframe(ads)
        st.write('#### Event Sales Multiplier')
        st.dataframe(event_multiplier)

        # Save the predictions to CSV
        input_data.to_csv('predictions.csv', index=False)
        st.write('Predictions saved to predictions.csv.')

# Section: Visualizations
if section == 'Visualizations':
    st.header('Visualizations')
    st.write('### Inventory Projection Charts')

    # Load prediction data if available
    try:
        predictions_df = pd.read_csv('predictions.csv')
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])

        # Recalculate key metrics to ensure availability
        ads, event_multiplier = calculate_metrics(predictions_df)

    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        predictions_df = pd.DataFrame()  # Create an empty dataframe as fallback

    if not predictions_df.empty:
        # 1. Inventory Level Over Time by Event Type
        st.write('#### Inventory Level Over Time by Event Type')
        fig, ax = plt.subplots()
        for event_type, group in predictions_df.groupby('Event Indicator'):
            ax.plot(group['Date'], group['Forecasted Sales Volume'], label=event_type)
        ax.set_title('Forecasted Inventory Levels Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Inventory Levels')
        ax.legend()
        st.pyplot(fig)

        # 2. Bar Plot of Safety Stock by Product Category
        st.write('#### Safety Stock by Product Category')
        safety_stock_category = predictions_df.groupby('ProductCategory')['Safety Stock'].sum().reset_index()
        fig, ax = plt.subplots()
        ax.bar(safety_stock_category['ProductCategory'], safety_stock_category['Safety Stock'])
        ax.set_xticklabels(safety_stock_category['ProductCategory'], rotation=45)
        ax.set_title('Total Safety Stock by Product Category')
        ax.set_xlabel('Product Category')
        ax.set_ylabel('Safety Stock')
        st.pyplot(fig)

        # 3. Pie Chart of Revenue Contribution by Event
        st.write('#### Revenue Contribution by Event')
        revenue_by_event = predictions_df.groupby('Event Indicator')['Revenue'].sum()
        fig, ax = plt.subplots()
        ax.pie(revenue_by_event, labels=revenue_by_event.index, autopct='%1.1f%%', startangle=140)
        ax.set_title('Revenue Contribution by Event')
        st.pyplot(fig)

        # 4. Histogram of Forecasted Sales Volume
        st.write('#### Distribution of Forecasted Sales Volume')
        fig, ax = plt.subplots()
        ax.hist(predictions_df['Forecasted Sales Volume'], bins=20, edgecolor='black')
        ax.set_title('Distribution of Forecasted Sales Volume')
        ax.set_xlabel('Forecasted Sales Volume')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # 5. Event Sales Multiplier Comparison
        st.write('#### Event Sales Multiplier Comparison')
        fig, ax = plt.subplots()
        ax.bar(event_multiplier['Event Indicator'], event_multiplier['Event Sales Multiplier'], color='skyblue')
        ax.set_title('Event Sales Multiplier for Different Events')
        ax.set_xlabel('Event Type')
        ax.set_ylabel('Sales Multiplier')
        st.pyplot(fig)

        # 6. Average Daily Sales by Product
        st.write('#### Average Daily Sales by Product')
        fig, ax = plt.subplots()
        ax.bar(ads['ProductID'], ads['Average Daily Sales (ADS)'], color='orange')
        ax.set_title('Average Daily Sales by Product')
        ax.set_xlabel('Product ID')
        ax.set_ylabel('Average Daily Sales (ADS)')
        ax.set_xticklabels(ads['ProductID'], rotation=45)
        st.pyplot(fig)

        st.write('### Inventory Projection Table')
        st.dataframe(predictions_df)
