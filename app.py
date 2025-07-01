import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from google.oauth2.service_account import Credentials
import gspread

# Page configuration
st.set_page_config(
    page_title="Sales Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Data preprocessing function
def preprocess_data(df):
    """Preprocess the sales data"""
    # Convert date columns
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure numeric columns
    numeric_columns = ['Revenue', 'Units Sold', 'Customer Satisfaction']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add derived columns
    if 'Revenue' in df.columns and 'Units Sold' in df.columns:
        df['Revenue per Unit'] = df['Revenue'] / df['Units Sold']
    
    # Extract time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Day of Week'] = df['Date'].dt.day_name()
    
    return df

# Cached data loading function with backup
@st.cache_data(ttl=3600)
def load_sales_data():
    try:
        # Your existing Google Sheets code...
        credentials_dict = st.secrets["google_credentials"]
        
        # Create credentials
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        
        # Authorize and get the sheet
        gc = gspread.authorize(credentials)
        sheet = gc.open_by_key(st.secrets["sheet_id"]).sheet1
        
        # Get all values
        data = sheet.get_all_values()
        
        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Preprocess the data
        df = preprocess_data(df)
        
        return df, "Connected to Google Sheets"
        
    except Exception as e:
        # BACKUP: Load from CSV when API quota exceeded
        st.warning(f"⚠️ Google Sheets API quota exceeded. Loading from backup data.")
        try:
            df = pd.read_csv("backup_sales_data.csv")
            df = preprocess_data(df)
            return df, "Using backup data due to API limits"
        except FileNotFoundError:
            st.error("❌ Backup file not found. Please add backup_sales_data.csv to your project.")
            return None, "No backup data available"

# Generate sample data for demonstration
@st.cache_data
def generate_sample_data():
    """Generate sample sales data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    data = {
        'Date': dates,
        'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], len(dates)),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'Sales Rep': np.random.choice(['John', 'Jane', 'Bob', 'Alice', 'Charlie'], len(dates)),
        'Revenue': np.random.normal(1000, 300, len(dates)).clip(100, 2000),
        'Units Sold': np.random.poisson(50, len(dates)),
        'Customer Satisfaction': np.random.uniform(3.5, 5.0, len(dates))
    }
    
    df = pd.DataFrame(data)
    return preprocess_data(df)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Sales Performance Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading sales data...'):
        result = load_sales_data()
        
        if result[0] is None:
            # Use sample data if no data available
            st.info("📝 Using sample data for demonstration")
            df = generate_sample_data()
            data_source = "Sample Data"
        else:
            df, data_source = result
    
    # Display data source
    st.sidebar.markdown(f"**Data Source:** {data_source}")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )
    
    # Product filter
    products = st.sidebar.multiselect(
        "Select Products",
        options=df['Product'].unique(),
        default=df['Product'].unique()
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )
    
    # Apply filters
    mask = (
        (df['Date'].dt.date >= date_range[0]) & 
        (df['Date'].dt.date <= date_range[1]) &
        (df['Product'].isin(products)) &
        (df['Region'].isin(regions))
    )
    filtered_df = df[mask]
    
    # Key Metrics
    st.header("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_df['Revenue'].sum()
        st.metric(
            "Total Revenue",
            f"${total_revenue:,.2f}",
            delta=f"{((total_revenue / df['Revenue'].sum()) - 1) * 100:.1f}% vs all time"
        )
    
    with col2:
        total_units = filtered_df['Units Sold'].sum()
        st.metric(
            "Units Sold",
            f"{total_units:,}",
            delta=f"{((total_units / df['Units Sold'].sum()) - 1) * 100:.1f}% vs all time"
        )
    
    with col3:
        avg_satisfaction = filtered_df['Customer Satisfaction'].mean()
        st.metric(
            "Avg. Satisfaction",
            f"{avg_satisfaction:.2f} / 5.0",
            delta=f"{(avg_satisfaction - df['Customer Satisfaction'].mean()):.2f} vs average"
        )
    
    with col4:
        avg_revenue_per_unit = filtered_df['Revenue per Unit'].mean()
        st.metric(
            "Avg. Revenue/Unit",
            f"${avg_revenue_per_unit:.2f}",
            delta=f"{((avg_revenue_per_unit / df['Revenue per Unit'].mean()) - 1) * 100:.1f}% vs average"
        )
    
    # Charts
    st.header("📊 Sales Analytics")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Revenue over time
        st.subheader("Revenue Trend")
        daily_revenue = filtered_df.groupby('Date')['Revenue'].sum().reset_index()
        fig_revenue = px.line(
            daily_revenue, 
            x='Date', 
            y='Revenue',
            title='Daily Revenue',
            labels={'Revenue': 'Revenue ($)'}
        )
        fig_revenue.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Product performance
        st.subheader("Product Performance")
        product_revenue = filtered_df.groupby('Product')['Revenue'].sum().reset_index()
        fig_product = px.bar(
            product_revenue,
            x='Product',
            y='Revenue',
            title='Revenue by Product',
            color='Product',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_product, use_container_width=True)
    
    with chart_col2:
        # Regional distribution
        st.subheader("Regional Distribution")
        region_revenue = filtered_df.groupby('Region')['Revenue'].sum().reset_index()
        fig_region = px.pie(
            region_revenue,
            values='Revenue',
            names='Region',
            title='Revenue by Region',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_region, use_container_width=True)
        
        # Sales rep performance
        st.subheader("Sales Rep Performance")
        rep_performance = filtered_df.groupby('Sales Rep').agg({
            'Revenue': 'sum',
            'Units Sold': 'sum',
            'Customer Satisfaction': 'mean'
        }).reset_index()
        
        fig_rep = go.Figure()
        fig_rep.add_trace(go.Bar(
            x=rep_performance['Sales Rep'],
            y=rep_performance['Revenue'],
            name='Revenue',
            yaxis='y',
            marker_color='lightblue'
        ))
        fig_rep.add_trace(go.Scatter(
            x=rep_performance['Sales Rep'],
            y=rep_performance['Customer Satisfaction'],
            name='Satisfaction',
            yaxis='y2',
            mode='lines+markers',
            marker_color='orange',
            line_width=3
        ))
        
        fig_rep.update_layout(
            title='Sales Rep Performance',
            yaxis=dict(title='Revenue ($)', side='left'),
            yaxis2=dict(title='Satisfaction (1-5)', side='right', overlaying='y', range=[0, 5]),
            hovermode='x unified'
        )
        st.plotly_chart(fig_rep, use_container_width=True)
    
    # Monthly trends
    st.header("📅 Monthly Analysis")
    monthly_df = filtered_df.groupby(['Year', 'Month']).agg({
        'Revenue': 'sum',
        'Units Sold': 'sum',
        'Customer Satisfaction': 'mean'
    }).reset_index()
    monthly_df['Month-Year'] = pd.to_datetime(monthly_df[['Year', 'Month']].assign(day=1))
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_df['Month-Year'],
        y=monthly_df['Revenue'],
        name='Revenue',
        marker_color='skyblue'
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly_df['Month-Year'],
        y=monthly_df['Units Sold'] * 20,  # Scale for visibility
        name='Units Sold (x20)',
        mode='lines+markers',
        marker_color='green',
        yaxis='y2'
    ))
    
    fig_monthly.update_layout(
        title='Monthly Revenue and Units Sold',
        xaxis_title='Month',
        yaxis=dict(title='Revenue ($)', side='left'),
        yaxis2=dict(title='Units Sold (scaled)', side='right', overlaying='y'),
        hovermode='x unified'
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Data table
    st.header("📋 Detailed Data")
    
    # Add download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f'sales_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )
    
    # Display data table with pagination
    st.dataframe(
        filtered_df.sort_values('Date', ascending=False).head(100),
        use_container_width=True,
        hide_index=True
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"*Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}* | "
        f"*Showing {len(filtered_df):,} of {len(df):,} records*"
    )

if __name__ == "__main__":
    main()
