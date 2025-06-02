import gspread
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from google.oauth2.service_account import Credentials
import os
from datetime import datetime, timedelta
import json

# ADD THIS ENTIRE FUNCTION HERE (after imports, before anything else)
from datetime import datetime, timedelta

def get_date_contexts():
    """Generate all date contexts automatically based on current date"""
    
# Get current date
    today = datetime.now()
    
# Helper function to check if a date is a workday (Sunday=0 to Thursday=4)
    def is_workday(date):
        return date.weekday() in [6, 0, 1, 2, 3]  # Sunday(6), Monday(0), Tuesday(1), Wednesday(2), Thursday(3)
    
# Calculate yesterday
    yesterday = today - timedelta(days=1)
    
# Calculate last workday (most recent workday before today)
    last_workday = today - timedelta(days=1)
    while not is_workday(last_workday):
        last_workday -= timedelta(days=1)
    
# Calculate last week (same day previous week)
    last_week = today - timedelta(weeks=1)
    
# Calculate last month (same day previous month)
    if today.month == 1:
        last_month = today.replace(year=today.year-1, month=12)
    else:
        try:
            last_month = today.replace(month=today.month-1)
        except ValueError:
            last_month = today.replace(month=today.month-1, day=28)
    
# Calculate this week start (last Sunday)
    days_since_sunday = (today.weekday() + 1) % 7
    week_start = today - timedelta(days=days_since_sunday)
    
# Calculate this month start
    month_start = today.replace(day=1)
    
# Format all dates
    date_formats = {
        'today': today.strftime("%Y-%m-%d"),
        'today_formatted': today.strftime("%B %d, %Y"),
        'yesterday': yesterday.strftime("%Y-%m-%d"),
        'yesterday_formatted': yesterday.strftime("%B %d, %Y"),
        'last_workday': last_workday.strftime("%Y-%m-%d"),
        'last_workday_formatted': last_workday.strftime("%B %d, %Y"),
        'last_week': last_week.strftime("%Y-%m-%d"),
        'last_week_formatted': last_week.strftime("%B %d, %Y"),
        'last_month': last_month.strftime("%Y-%m-%d"),
        'last_month_formatted': last_month.strftime("%B %d, %Y"),
        'week_start': week_start.strftime("%Y-%m-%d"),
        'week_start_formatted': week_start.strftime("%B %d, %Y"),
        'month_start': month_start.strftime("%Y-%m-%d"),
        'month_start_formatted': month_start.strftime("%B %d, %Y"),
    }
    
    return date_formats
    
# Set page config
st.set_page_config(
    page_title="RTE Sales Data Analyzer",
    page_icon="📊",
    layout="wide"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sales_data():
    """Load sales data from Google Sheets with caching"""
    try:
        # Load credentials from Streamlit secrets
        credentials_dict = st.secrets["google_credentials"]
        
        # Create credentials object
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
        
        # Connect to Google Sheets
        client = gspread.authorize(credentials)
        
        # Open the spreadsheet
        spreadsheet_url = "https://docs.google.com/spreadsheets/d/1CArpdv0DSd1Ng-bWocnf3c7bV4_44OjsK9ocHxvMSwE/edit"
        spreadsheet = client.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet("sales orderline")
        
        # Get all data starting from row 2 (headers are in row 1)
        data = worksheet.get_all_values()
        
        # Use the first row as headers and the rest as data
        headers = data[0]  # Row 1 contains headers
        rows = data[1:]    # Row 2 onwards contains data
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        # Preprocess the data
        df = preprocess_data(df)
        
        return df, None
    except Exception as e:
        return None, str(e)

def preprocess_data(df):
    """Clean and preprocess the sales data"""
    # Convert date columns to datetime
    date_columns = [
        'SALES_LINE_CREATION_DATE', 
        'SALES_ORDER_CREATION_DATE',
        'SALES_LINE_REQUESTED_SHIPPING_DATE', 
        'SALES_ORDER_REQUESTED_SHIPPING_DATE',
        'LAST_UPDATED_AT'
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['LINE_AMOUNT', 'LINE_AMOUNT_AFTER_DISCOUNT', 'SALES_QUANTITY', 'DISCOUNT_PERCENTAGE']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
    
    # Fill NaN values appropriately
    if 'IS_SISTER_COMPANY' in df.columns:
        df['IS_SISTER_COMPANY'] = df['IS_SISTER_COMPANY'].fillna('NO')
    if 'REGION' in df.columns:
        df['REGION'] = df['REGION'].fillna('Unknown')
    
    return df

def get_data_summary(df):
    """Get a summary of the dataset for Claude context"""
    summary = {
        'total_rows': len(df),
        'date_range': {
            'start': 'Unknown',
            'end': 'Unknown'
        },
        'unique_customers': 0,
        'unique_regions': [],
        'sales_order_statuses': [],
        'total_sales_amount': 0,
        'customer_groups': []
    }
    
    # Safely get date range
    if 'SALES_LINE_CREATION_DATE' in df.columns:
        date_min = df['SALES_LINE_CREATION_DATE'].min()
        date_max = df['SALES_LINE_CREATION_DATE'].max()
        if pd.notna(date_min):
            summary['date_range']['start'] = date_min.strftime('%Y-%m-%d')
        if pd.notna(date_max):
            summary['date_range']['end'] = date_max.strftime('%Y-%m-%d')
    
    # Safely get other metrics
    if 'CUSTOMER_NAME' in df.columns:
        summary['unique_customers'] = df['CUSTOMER_NAME'].nunique()
    
    if 'REGION' in df.columns:
        summary['unique_regions'] = df['REGION'].unique().tolist()
    
    if 'SALES_ORDER_STATUS' in df.columns:
        summary['sales_order_statuses'] = df['SALES_ORDER_STATUS'].unique().tolist()
    
    if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary['total_sales_amount'] = df['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
    
    if 'CUSTOMER_GROUP' in df.columns:
        summary['customer_groups'] = df['CUSTOMER_GROUP'].unique().tolist()[:10]
    
    return summary

def analyze_data_with_claude(df, user_question):
    """Analyze data using Claude API"""
    try:
        # Initialize Claude API
        client = Anthropic(api_key=st.secrets["anthropic_api_key"])
        
        # Get data summary
        summary = get_data_summary(df)
        
        # Get a sample of recent data (last 10 rows)
        recent_data = df.tail(10).to_string(index=False)
        
        # Create system prompt with comprehensive context
        system_prompt = f"""You are a data analyst for RTE/Raqtan, a company that uses Microsoft Dynamics 365 for sales data.

DATA CONTEXT:
- Total records: {summary['total_rows']}
- Date range: {summary['date_range']['start']} to {summary['date_range']['end']}
- Unique customers: {summary['unique_customers']}
- Total sales amount: {summary['total_sales_amount']:,.2f} SAR
- Regions: {', '.join(summary['unique_regions'])}
- Sales order statuses: {', '.join(summary['sales_order_statuses'])}

COLUMN DESCRIPTIONS:
- SALES_ID: Unique identifier for each sales line
- CUSTOMER_NAME: Name of the customer
- CUSTOMER_ACCOUNT: Customer account number
- ITEM_ID: Product/service identifier
- SALES_ORDER_STATUS: Status of the entire order (invoiced, Open, Delivered, cancelled)
- SALES_LINE_STATUS: Status of the individual line item
- LINE_AMOUNT: Net amount (unit price × quantity)
- LINE_AMOUNT_AFTER_DISCOUNT: Final amount after discounts (USE THIS for financial calculations)
- SALES_QUANTITY: Quantity sold
- DISCOUNT_PERCENTAGE: Discount applied
- SALES_LINE_CREATION_DATE: When the line item was created
- SALES_ORDER_CREATION_DATE: When the entire order was created
- CUSTOMER_GROUP: Sales person name (starts with region prefix: J-, R-, K-)
- REGION: Geographic region (Jeddah, Khobar, Riyadh, Dubai, etc.)
- IS_SISTER_COMPANY: YES = internal job, NO = external sale
- Currency: Saudi Riyals (SAR)

BUSINESS RULES:
- Use LINE_AMOUNT_AFTER_DISCOUNT for all financial calculations
- IS_SISTER_COMPANY = YES means internal transactions
- Sales order data is for the whole order, sales line data is item-level
- This table is at the sales line level
- Customer groups starting with J- are from Jeddah, R- from Riyadh, K- from Khobar
- Spherey is an after-sales service company under Raqtan

ANALYSIS INSTRUCTIONS:
1. Always show your calculation logic
2. Provide specific numbers and percentages
3. Include relevant context and insights
4. If analyzing time periods, be specific about dates
5. Consider business context in your analysis
6. Format large numbers with commas for readability

When answering questions about "yesterday" or "last quarter", calculate based on the most recent data available."""

# Generate current date context
dates = get_date_contexts()

date_context = f"""
CURRENT DATE CONTEXT (Auto-generated):
- Today: {dates['today_formatted']} ({dates['today']})
- Yesterday: {dates['yesterday_formatted']} ({dates['yesterday']})
- Last workday: {dates['last_workday_formatted']} ({dates['last_workday']})
- Last week (same day): {dates['last_week_formatted']} ({dates['last_week']})
- Last month (same day): {dates['last_month_formatted']} ({dates['last_month']})
- This week started: {dates['week_start_formatted']} ({dates['week_start']})
- This month started: {dates['month_start_formatted']} ({dates['month_start']})

WORKWEEK DEFINITION:
- Workdays: Sunday, Monday, Tuesday, Wednesday, Thursday
- Weekends: Friday, Saturday

DATE INTERPRETATION RULES:
- "Yesterday" = {dates['yesterday']}
- "Today" = {dates['today']}
- "Last workday" = {dates['last_workday']} (most recent workday)
- "This week" = from {dates['week_start']} to today
- "Last week" = use {dates['last_week']} as reference
- "This month" = from {dates['month_start']} to today
- "Last month" = use {dates['last_month']} as reference

RESPONSE GUIDELINES:
- Provide direct, conversational business answers
- Never show SQL queries or technical details
- Use the exact dates provided above for filtering data
- Focus on actionable business insights
"""
        # Create user prompt with question and sample data
        user_prompt = f"""
[date_context]

QUESTION: {user_question}

RECENT DATA SAMPLE (last 10 records):
{recent_data}

Please analyze the full dataset to answer this question. Show your reasoning and provide specific insights.
"""

        # Get response from Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.content[0].text, None
    
    except Exception as e:
        return None, str(e)

def main():
    """Main Streamlit application"""
    st.title("🏢 RTE/Raqtan Sales Data Analyzer")
    st.markdown("Ask questions about your sales data and get AI-powered insights from Claude.")
    
    # Sidebar for data info
    with st.sidebar:
        st.header("📊 Data Information")
        
        # Load data
        with st.spinner("Loading data..."):
            df, error = load_sales_data()
        
        if error:
            st.error(f"Error loading data: {error}")
            st.stop()
        
        if df is not None:
            st.success("✅ Data loaded successfully")
            st.metric("Total Records", len(df))
            
            # Safe metrics calculation
            if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                total_sales = df['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                st.metric("Total Sales Amount", f"{total_sales:,.0f} SAR")
            
            if 'CUSTOMER_NAME' in df.columns:
                unique_customers = df['CUSTOMER_NAME'].nunique()
                st.metric("Unique Customers", unique_customers)
            
            # Show data freshness
            if 'LAST_UPDATED_AT' in df.columns:
                last_update = df['LAST_UPDATED_AT'].max()
                if pd.notna(last_update):
                    st.info(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M')}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Question")
        
        # Sample questions
        sample_questions = [
            "Who did the most sales yesterday?",
            "Which customer did less business with us in the last quarter?",
            "What is the total sales amount for each region this month?",
            "Show me the top 5 customers by sales volume",
            "Which sales person (customer group) has the highest sales?",
            "How many orders are still open?",
            "What is the average discount percentage given?",
            "Compare internal vs external sales this month"
        ]
        
        selected_question = st.selectbox(
            "Choose a sample question or type your own:",
            ["Type your own question..."] + sample_questions
        )
        
        if selected_question != "Type your own question...":
            user_question = st.text_area("Your question:", value=selected_question, height=100)
        else:
            user_question = st.text_area("Your question:", height=100, 
                                       placeholder="e.g., Who are the top 3 customers by sales amount this month?")
    
    with col2:
        st.subheader("📋 Quick Stats")
        if df is not None:
            # Recent activity (safe calculation)
            if 'SALES_LINE_CREATION_DATE' in df.columns:
                recent_sales = df[df['SALES_LINE_CREATION_DATE'] >= datetime.now() - timedelta(days=7)]
                st.metric("Sales This Week", len(recent_sales))
            
            # Top region (safe calculation)
            if 'REGION' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                top_region_data = df.groupby('REGION')['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                if not top_region_data.empty:
                    top_region = top_region_data.idxmax()
                    st.metric("Top Region", top_region)
            
            # Internal vs External (safe calculation)
            if 'IS_SISTER_COMPANY' in df.columns:
                internal_pct = (df['IS_SISTER_COMPANY'] == 'YES').mean() * 100
                st.metric("Internal Sales %", f"{internal_pct:.1f}%")
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not user_question.strip():
            st.error("Please enter a question to analyze.")
        else:
            with st.spinner("Analyzing data with Claude AI..."):
                answer, error = analyze_data_with_claude(df, user_question)
                
                if error:
                    st.error(f"Error during analysis: {error}")
                else:
                    st.subheader("📈 Analysis Results")
                    st.markdown(answer)
                    
                    # Show timestamp
                    st.caption(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and Claude AI for RTE/Raqtan sales data analysis*")

if __name__ == "__main__":
    main()
