import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
import plotly.express as px  # Import Plotly Express for visualizations
import plotly.graph_objects as go
import pickle

# Set Streamlit app configuration
st.set_page_config(page_title="Meesho Inventory and Revenue Optimization", layout="wide")

# Sidebar - Upload Dataset
st.sidebar.header('Upload your CSV file')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Main Panel
st.title('Inventory and Revenue Optimization Dashboard')

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Convert 'Date' to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Extract features from 'Date'
    data['Day'] = data['Date'].dt.day
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    # Create Lag Features
    data['SalesVolume_Lag1'] = data['SalesVolume'].shift(1).fillna(0)
    data['Revenue_Lag1'] = data['Revenue'].shift(1).fillna(0)

    # Keep a copy of the 'ProductID' column before one-hot encoding
    product_id_column = data['ProductID'].copy()

    # Drop 'Date' column if not needed
    data.drop(['Date'], axis=1, inplace=True)

    # Convert 'CustomerIncome' to numeric
    data['CustomerIncome'] = data['CustomerIncome'].replace({'Low': 1, 'Medium': 2, 'High': 3}).astype(int)

    # Convert categorical columns to category type
    categorical_columns = [
        'DayOfWeek', 'Season', 'Event', 'ProductID', 'ProductCategory',
        'CustomerGender', 'CustomerLocation', 'CustomerSegment'
    ]

    for col in categorical_columns:
        data[col] = data[col].astype('category')

    # One-Hot Encoding for Categorical Variables
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Log Transformation of skewed features
    data['Revenue'] = np.log1p(data['Revenue'])
    data['MarketingSpend'] = np.log1p(data['MarketingSpend'])
    data['SalesVolume'] = np.log1p(data['SalesVolume'])

    # Define features and target variables
    X = data.drop(['SalesVolume', 'Revenue'], axis=1)
    y_sales = data['SalesVolume']
    y_revenue = data['Revenue']

    # Save the full feature names before feature selection
    full_feature_names = X.columns.tolist()

    # Feature Selection using Feature Importance from XGBoost
    feature_selector = SelectFromModel(xgb.XGBRegressor(n_estimators=100, random_state=42), threshold='median')
    feature_selector.fit(X, y_sales)
    X_selected = feature_selector.transform(X)

    # Get selected feature names based on the feature selector
    selected_features = np.array(full_feature_names)[feature_selector.get_support()].tolist()

    # Convert selected features back to a DataFrame with the original feature names
    X_selected = pd.DataFrame(X_selected, columns=selected_features)

    # Split data for training and testing
    X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(X_selected, y_sales, test_size=0.2, random_state=42)
    X_train_revenue, X_test_revenue, y_train_revenue, y_test_revenue = train_test_split(X_selected, y_revenue, test_size=0.2, random_state=42)

    # Save the selected feature names to be used during prediction
    with open('selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)

    # Define XGBoost model with reduced parameters
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8)

    # Hyperparameter tuning for Sales Volume model
    random_search_sales = RandomizedSearchCV(estimator=xgb_model, param_distributions={
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8]
    }, scoring='neg_mean_squared_error', cv=TimeSeriesSplit(n_splits=3), verbose=1, n_jobs=-1, n_iter=10)
    random_search_sales.fit(X_train_sales, y_train_sales)

    # Use the best estimator for Sales Volume
    best_xgb_sales_model = random_search_sales.best_estimator_
    best_xgb_sales_model.fit(X_train_sales, y_train_sales)

    # Hyperparameter tuning for Revenue model
    random_search_revenue = RandomizedSearchCV(estimator=xgb_model, param_distributions={
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8]
    }, scoring='neg_mean_squared_error', cv=TimeSeriesSplit(n_splits=3), verbose=1, n_jobs=-1, n_iter=10)
    random_search_revenue.fit(X_train_revenue, y_train_revenue)

    # Use the best estimator for Revenue
    best_xgb_revenue_model = random_search_revenue.best_estimator_
    best_xgb_revenue_model.fit(X_train_revenue, y_train_revenue)

    # Evaluate models
    sales_predictions = best_xgb_sales_model.predict(X_test_sales)
    revenue_predictions = best_xgb_revenue_model.predict(X_test_revenue)
    print(f"Sales Volume MSE: {mean_squared_error(y_test_sales, sales_predictions)}")
    print(f"Revenue MSE: {mean_squared_error(y_test_revenue, revenue_predictions)}")

    # Save the improved models as pickle files
    with open('optimized_sales_volume_model.pkl', 'wb') as f:
        pickle.dump(best_xgb_sales_model, f)

    with open('optimized_revenue_model.pkl', 'wb') as f:
        pickle.dump(best_xgb_revenue_model, f)

    # Load the models back
    with open('optimized_sales_volume_model.pkl', 'rb') as f:
        sales_model = pickle.load(f)

    with open('optimized_revenue_model.pkl', 'rb') as f:
        revenue_model = pickle.load(f)

    # Display Overview Metrics
    try:
        st.write("## Dashboard Overview")
        st.metric(label="Overall Revenue", value=f"${np.expm1(data['Revenue']).sum():,.2f}")
        st.metric(label="Average Daily Sales (ADS)", value=f"{np.expm1(data['SalesVolume']).mean():.2f} units")
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")

    # Event-Specific Highlights
    try:
        event_columns = [col for col in data.columns if 'Event_' in col]
        for event_col in event_columns:
            event_name = event_col.split('_')[1]
            event_data = data[data[event_col] == 1]
            st.write(f"#### {event_name} Event Highlights")
            st.metric(label=f"Projected Revenue for {event_name}", value=f"${np.expm1(event_data['Revenue']).sum():,.2f}")
            st.metric(label=f"Inventory Needs for {event_name}", value=f"{event_data['InventoryLevel'].sum()} units")
            st.write("---")
    except Exception as e:
        st.error(f"Error displaying event highlights: {e}")

    # Revenue Optimization Page
    try:
        st.header('Revenue Optimization Page')
        st.write('Use the Discount Strategy Simulator to project revenue.')
        discount_rate = st.slider('Select Discount Rate (%)', min_value=0, max_value=100, step=5)
        st.write(f'You selected a discount rate of {discount_rate}%')

        # Apply discount rate for predictions
        X_selected['DiscountRate'] = discount_rate

        # Predict sales and revenue
        predicted_sales = sales_model.predict(X_selected)
        predicted_revenue = revenue_model.predict(X_selected)

        st.write("Projected Sales Volume:", np.sum(np.expm1(predicted_sales)))
        st.write("Projected Revenue:", f"${np.sum(np.expm1(predicted_revenue)):,.2f}")

        # Plot historical relationship between discounts and sales/revenue
        fig1 = px.scatter(data, x='DiscountRate', y='SalesVolume', trendline='ols', title='Historical Discount vs Sales Volume')
        st.plotly_chart(fig1)
        fig2 = px.scatter(data, x='DiscountRate', y='Revenue', trendline='ols', title='Historical Discount vs Revenue')
        st.plotly_chart(fig2)
    except Exception as e:
        st.error(f"Error generating discount vs sales/revenue plots: {e}")

    # Sales Event Strategy Page
    try:
        st.header('Sales Event Strategy Page')
        st.write('Adjust inventory and pricing strategies based on projected demand for sale events.')

        # Dynamic Pricing Recommendations
        st.write("### Dynamic Pricing Recommendations")
        price_recommendations = pd.DataFrame({
            'ProductID': product_id_column,
            'Current Price': data['Price'],
            'Competitor Price': data['CompetitorPrice'],
            'Recommended Price': data['Price'] * (1 - discount_rate / 100),
            'Price Elasticity': np.random.uniform(0.5, 2.0, size=len(data)),  # Mock data for demonstration
            'Projected Revenue': np.expm1(predicted_revenue)
        })
        st.dataframe(price_recommendations)
    except Exception as e:
        st.error(f"Error generating pricing recommendations: {e}")

    # Inventory Management Insights
    try:
        st.write("### Inventory Management Insights")
        inventory_needs = pd.DataFrame({
            'ProductID': data.index,  # Adjust to use the index or correct column name for ProductID
            'Current Inventory': data['InventoryLevel'],
            'Projected Inventory Needs': np.ceil(predicted_sales * 1.1)  # Example multiplier
        })

        # Calculate 'Status' column
        inventory_needs['Status'] = np.where(
            inventory_needs['Current Inventory'] < inventory_needs['Projected Inventory Needs'], 
            'Critical', 'Sufficient'
        )

        # Apply correct styling function
        def highlight_status(row):
            return ['background-color: red' if val == 'Critical' else 'background-color: green' for val in row]

        # Ensure the styler returns a properly shaped list
        styler = inventory_needs.style.apply(lambda x: highlight_status(x), axis=1)

        st.dataframe(styler)
    except Exception as e:
        st.error(f"Error generating inventory management insights: {e}")

    # Interactive Visualizations
    try:
        st.header('Interactive Visualizations')

        # Revenue vs. Discount Charts
        fig3 = px.line(data, x='DiscountRate', y='Revenue', title='Revenue vs. Discount')
        st.plotly_chart(fig3)

        # Sales Volume Forecast Charts
        fig4 = px.bar(data, x=product_id_column, y='SalesVolume', title='Sales Volume Forecast by Product')
        st.plotly_chart(fig4)

        # Inventory Projection Heatmaps
        fig5 = go.Figure(data=go.Heatmap(z=inventory_needs['Projected Inventory Needs'], x=inventory_needs['ProductID'], y=inventory_needs['Projected Inventory Needs']))
        st.plotly_chart(fig5)

        # Event Sales Multiplier Comparisons
        event_sales_multiplier = pd.DataFrame({'Event': ['FSMS', 'MBS', 'MDS'], 'Sales Multiplier': [1.5, 2.0, 1.8]})
        fig6 = px.bar(event_sales_multiplier, x='Event', y='Sales Multiplier', title='Event Sales Multiplier Comparison')
        st.plotly_chart(fig6)
    except Exception as e:
        st.error(f"Error generating visualizations: {e}")

    # What-If Analysis Tool
    try:
        st.header('What-If Analysis Tool')
        st.write('Simulate various scenarios by adjusting different variables.')

        # Inputs for What-If Analysis
        st.write("### Adjust Parameters")
        new_discount_rate = st.slider('Discount Percentage', min_value=0, max_value=100, step=5)
        new_competitor_price = st.slider('Competitor Price', min_value=0.0, max_value=1000.0, step=10.0)
        new_inventory_level = st.slider('Inventory Level', min_value=0, max_value=1000, step=10)

        # Simulate new predictions
        X_what_if = X_selected.copy()
        X_what_if['DiscountRate'] = new_discount_rate
        X_what_if['CompetitorPrice'] = new_competitor_price
        X_what_if['InventoryLevel'] = new_inventory_level

        new_sales_pred = sales_model.predict(X_what_if)
        new_revenue_pred = revenue_model.predict(X_what_if)

        st.write("New Projected Sales Volume:", np.sum(np.expm1(new_sales_pred)))
        st.write("New Projected Revenue:", f"${np.sum(np.expm1(new_revenue_pred)):,.2f}")
    except Exception as e:
        st.error(f"Error in What-If Analysis Tool: {e}")

    # Summary Reports
    try:
        st.header('Summary Reports')
        st.write('Download the detailed summary reports for strategic decision-making.')

        # Buttons to download reports
        st.download_button(label='Download Revenue Projections (CSV)', data=price_recommendations.to_csv(index=False), file_name='revenue_projections.csv', mime='text/csv')
        st.download_button(label='Download Inventory Plans (CSV)', data=inventory_needs.to_csv(index=False), file_name='inventory_plans.csv', mime='text/csv')

        st.write("*Note*: These reports contain detailed insights for each event and BAU to help with decision-making.")
    except Exception as e:
        st.error(f"Error generating summary reports: {e}")