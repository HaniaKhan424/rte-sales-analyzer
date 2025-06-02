import gspread
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from google.oauth2.service_account import Credentials
import os
from datetime import datetime, timedelta
import json

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

def get_comprehensive_data_summary(df, user_question):
    """Generate comprehensive summary of the entire dataset"""
    summary = []
    
    # Get date contexts
    dates = get_date_contexts()
    today = datetime.strptime(dates['today'], '%Y-%m-%d').date()
    yesterday = datetime.strptime(dates['yesterday'], '%Y-%m-%d').date()
    week_start = datetime.strptime(dates['week_start'], '%Y-%m-%d').date()
    month_start = datetime.strptime(dates['month_start'], '%Y-%m-%d').date()
    
    # Basic dataset info
    summary.append(f"COMPLETE DATASET OVERVIEW ({len(df):,} total records):")
    if 'SALES_LINE_CREATION_DATE' in df.columns:
        date_min = df['SALES_LINE_CREATION_DATE'].min()
        date_max = df['SALES_LINE_CREATION_DATE'].max()
        if pd.notna(date_min) and pd.notna(date_max):
            summary.append(f"- Date range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
            summary.append(f"- Data spans: {(date_max - date_min).days} days")
    
    if 'CUSTOMER_NAME' in df.columns:
        summary.append(f"- Unique customers: {df['CUSTOMER_NAME'].nunique():,}")
    
    if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        total_sales = df['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
        summary.append(f"- Total sales amount: {total_sales:,.2f} SAR")
    
    # Time-based analysis using actual data
    summary.append(f"\nTIME-BASED ANALYSIS (from complete dataset):")
    
    # Yesterday's data
    if 'SALES_LINE_CREATION_DATE' in df.columns:
        yesterday_data = df[df['SALES_LINE_CREATION_DATE'].dt.date == yesterday]
        if len(yesterday_data) > 0:
            yesterday_sales = yesterday_data['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
            summary.append(f"- Yesterday ({yesterday}): {len(yesterday_data)} transactions, {yesterday_sales:,.2f} SAR")
            
            # Yesterday's top performers
            if len(yesterday_data) > 0 and 'CUSTOMER_GROUP' in yesterday_data.columns:
                yesterday_reps = yesterday_data.groupby('CUSTOMER_GROUP')['LINE_AMOUNT_AFTER_DISCOUNT'].sum().sort_values(ascending=False)
                summary.append("  Top performers yesterday:")
                for rep, amount in yesterday_reps.head(5).items():
                    summary.append(f"    • {rep}: {amount:,.2f} SAR")
        else:
            # Find most recent date if yesterday has no data
            recent_dates = df['SALES_LINE_CREATION_DATE'].dropna().dt.date.unique()
            if len(recent_dates) > 0:
                most_recent = max(recent_dates)
                recent_data = df[df['SALES_LINE_CREATION_DATE'].dt.date == most_recent]
                recent_sales = recent_data['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                summary.append(f"- No data for yesterday ({yesterday})")
                summary.append(f"- Most recent data ({most_recent}): {len(recent_data)} transactions, {recent_sales:,.2f} SAR")
    
    # This week's data
    this_week_data = df[df['SALES_LINE_CREATION_DATE'].dt.date >= week_start]
    if len(this_week_data) > 0:
        week_sales = this_week_data['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
        summary.append(f"- This week (from {week_start}): {len(this_week_data)} transactions, {week_sales:,.2f} SAR")
    
    # This month's data
    this_month_data = df[df['SALES_LINE_CREATION_DATE'].dt.date >= month_start]
    if len(this_month_data) > 0:
        month_sales = this_month_data['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
        summary.append(f"- This month (from {month_start}): {len(this_month_data)} transactions, {month_sales:,.2f} SAR")
    
    # Sales by region (complete dataset)
    if 'REGION' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary.append(f"\nSALES BY REGION (complete dataset):")
        region_sales = df.groupby('REGION')['LINE_AMOUNT_AFTER_DISCOUNT'].agg(['count', 'sum']).round(2)
        region_sales = region_sales.sort_values('sum', ascending=False)
        for region, data in region_sales.iterrows():
            summary.append(f"- {region}: {data['count']:,} transactions, {data['sum']:,.2f} SAR")
    
    # Top customers (complete dataset)
    if 'CUSTOMER_NAME' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary.append(f"\nTOP 10 CUSTOMERS (complete dataset):")
        top_customers = df.groupby('CUSTOMER_NAME')['LINE_AMOUNT_AFTER_DISCOUNT'].sum().sort_values(ascending=False).head(10)
        for i, (customer, amount) in enumerate(top_customers.items(), 1):
            summary.append(f"- {i}. {customer}: {amount:,.2f} SAR")
    
    # Sales representatives performance (complete dataset)
    if 'CUSTOMER_GROUP' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary.append(f"\nSALES REPRESENTATIVES PERFORMANCE (complete dataset):")
        rep_sales = df.groupby('CUSTOMER_GROUP')['LINE_AMOUNT_AFTER_DISCOUNT'].agg(['count', 'sum']).round(2)
        rep_sales = rep_sales.sort_values('sum', ascending=False).head(10)
        for rep, data in rep_sales.iterrows():
            summary.append(f"- {rep}: {data['count']:,} transactions, {data['sum']:,.2f} SAR")
    
    # Order status analysis
    if 'SALES_ORDER_STATUS' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary.append(f"\nORDER STATUS ANALYSIS (complete dataset):")
        status_analysis = df.groupby('SALES_ORDER_STATUS')['LINE_AMOUNT_AFTER_DISCOUNT'].agg(['count', 'sum']).round(2)
        for status, data in status_analysis.iterrows():
            summary.append(f"- {status}: {data['count']:,} orders, {data['sum']:,.2f} SAR")
    
    # Internal vs External
    if 'IS_SISTER_COMPANY' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary.append(f"\nINTERNAL vs EXTERNAL SALES (complete dataset):")
        internal_external = df.groupby('IS_SISTER_COMPANY')['LINE_AMOUNT_AFTER_DISCOUNT'].agg(['count', 'sum']).round(2)
        for category, data in internal_external.iterrows():
            category_name = "Internal" if category == "YES" else "External"
            summary.append(f"- {category_name}: {data['count']:,} transactions, {data['sum']:,.2f} SAR")
    
    return "\n".join(summary)

def get_targeted_data_sample(df, user_question):
    """Get targeted sample based on the specific question"""
    question_lower = user_question.lower()
    
    # Time-based questions
    if any(word in question_lower for word in ['yesterday', 'today']):
        dates = get_date_contexts()
        target_date = datetime.strptime(dates['yesterday'], '%Y-%m-%d').date()
        
        if 'SALES_LINE_CREATION_DATE' in df.columns:
            filtered_df = df[df['SALES_LINE_CREATION_DATE'].dt.date == target_date]
            if len(filtered_df) > 0:
                return f"YESTERDAY'S COMPLETE DATA ({target_date}) - {len(filtered_df)} records:\n" + filtered_df.to_string(index=False)
            else:
                # Get most recent date
                recent_date = df['SALES_LINE_CREATION_DATE'].dt.date.max()
                recent_df = df[df['SALES_LINE_CREATION_DATE'].dt.date == recent_date]
                return f"No data for yesterday ({target_date}). Most recent data ({recent_date}) - {len(recent_df)} records:\n" + recent_df.to_string(index=False)
    
    # Customer-based questions
    elif any(word in question_lower for word in ['customer', 'client']):
        if 'CUSTOMER_NAME' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
            top_customers = df.groupby('CUSTOMER_NAME')['LINE_AMOUNT_AFTER_DISCOUNT'].sum().sort_values(ascending=False).head(5)
            sample_data = df[df['CUSTOMER_NAME'].isin(top_customers.index)].head(20)
            return f"SAMPLE FROM TOP CUSTOMERS:\n" + sample_data.to_string(index=False)
    
    # Sales rep questions
    elif any(word in question_lower for word in ['sales', 'rep', 'representative', 'group']):
        if 'SALES_LINE_CREATION_DATE' in df.columns:
            recent_sales_data = df.sort_values('SALES_LINE_CREATION_DATE', ascending=False).head(30)
            return f"RECENT SALES DATA:\n" + recent_sales_data.to_string(index=False)
    
    # Default: most recent data
    if 'SALES_LINE_CREATION_DATE' in df.columns:
        recent_data = df.sort_values('SALES_LINE_CREATION_DATE', ascending=False).head(20)
        return f"MOST RECENT DATA:\n" + recent_data.to_string(index=False)
    else:
        return df.head(20).to_string(index=False)

def analyze_data_with_claude(df, user_question):
    """Analyze data using Claude API with full dataset access"""
    try:
        # Initialize Claude API
        client = Anthropic(api_key=st.secrets["anthropic_api_key"])
        
        # Get comprehensive data summary from entire dataset
        comprehensive_summary = get_comprehensive_data_summary(df, user_question)
        
        # Get targeted sample based on question
        targeted_sample = get_targeted_data_sample(df, user_question)
        
        # Generate current date context
        dates = get_date_contexts()
        
        # Get actual date range from dataset
        date_range_info = ""
        if 'SALES_LINE_CREATION_DATE' in df.columns:
            latest_date = df['SALES_LINE_CREATION_DATE'].max()
            earliest_date = df['SALES_LINE_CREATION_DATE'].min()
            
            date_range_info = f"""
ACTUAL DATA DATE RANGE:
- Earliest date in dataset: {earliest_date.strftime('%Y-%m-%d') if pd.notna(earliest_date) else 'Unknown'}
- Latest date in dataset: {latest_date.strftime('%Y-%m-%d') if pd.notna(latest_date) else 'Unknown'}
- Total date span: {(latest_date - earliest_date).days if pd.notna(latest_date) and pd.notna(earliest_date) else 'Unknown'} days
"""
        
        date_context = f"""
CURRENT DATE CONTEXT (Auto-generated):
- Today: {dates['today_formatted']} ({dates['today']})
- Yesterday: {dates['yesterday_formatted']} ({dates['yesterday']})
- Last workday: {dates['last_workday_formatted']} ({dates['last_workday']})
- This week started: {dates['week_start_formatted']} ({dates['week_start']})
- This month started: {dates['month_start_formatted']} ({dates['month_start']})

{date_range_info}

DATE INTERPRETATION RULES:
- "Yesterday" = {dates['yesterday']}
- "Today" = {dates['today']}
- "This week" = from {dates['week_start']} to today
- "This month" = from {dates['month_start']} to today

IMPORTANT: All analysis above is based on the COMPLETE dataset of {len(df):,} records.
If requested dates aren't available, I've provided the most recent data available.
"""
        
        # Create system prompt
        system_prompt = f"""You are a data analyst for RTE/Raqtan with FULL ACCESS to the complete sales dataset.

You have access to ALL {len(df):,} records in the dataset. The comprehensive summary below contains aggregated information from the ENTIRE dataset, not just a sample.

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
- Customer groups starting with J- are from Jeddah, R- from Riyadh, K- from Khobar
- Spherey is an after-sales service company under Raqtan

ANALYSIS INSTRUCTIONS:
1. Base your analysis on the COMPLETE dataset summary provided
2. Show specific calculations and numbers from the full dataset
3. Use actual date filtering based on current date context
4. Provide actionable business insights
5. If requested date isn't available, use most recent data and explain clearly
6. Format large numbers with commas for readability
7. Be conversational and business-focused in your response"""

        # Create user prompt
        user_prompt = f"""
{date_context}

COMPLETE DATASET SUMMARY:
{comprehensive_summary}

QUESTION: {user_question}

TARGETED DATA SAMPLE (for reference):
{targeted_sample}

Please analyze based on the complete dataset summary above. The sample is just for reference - your analysis should use the full dataset metrics provided in the summary.
"""

        # Get response from Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            max_tokens=2500,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.content[0].text, None
    
    except Exception as e:
        return None, str(e)

def get_data_summary(df):
    """Get a basic summary of the dataset for sidebar display"""
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
            if 'SALES_LINE_CREATION_DATE' in df.columns:
                latest_date = df['SALES_LINE_CREATION_DATE'].max()
                if pd.notna(latest_date):
                    st.info(f"Latest data: {latest_date.strftime('%Y-%m-%d')}")
    
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
    if st.button("🔍 Analyze Full Dataset", type="primary", use_container_width=True):
        if not user_question.strip():
            st.error("Please enter a question to analyze.")
        else:
            with st.spinner("Analyzing complete dataset with Claude AI..."):
                answer, error = analyze_data_with_claude(df, user_question)
                
                if error:
                    st.error(f"Error during analysis: {error}")
                else:
                    st.subheader("📈 Complete Dataset Analysis")
                    st.markdown(answer)
                    
                    # Show analysis details
                    st.caption(f"Analysis based on {len(df):,} total records | Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and Claude AI for RTE/Raqtan sales data analysis*")

if __name__ == "__main__":
    main()
