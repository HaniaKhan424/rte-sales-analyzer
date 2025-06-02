import gspread
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from google.oauth2.service_account import Credentials
import os
from datetime import datetime, timedelta
import json

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
        
        # Get all data
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
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
    df['IS_SISTER_COMPANY'] = df['IS_SISTER_COMPANY'].fillna('NO')
    df['REGION'] = df['REGION'].fillna('Unknown')
    
    return df

def get_data_summary(df):
    """Get a summary of the dataset for Claude context"""
    summary = {
        'total_rows': len(df),
        'date_range': {
            'start': df['SALES_LINE_CREATION_DATE'].min().strftime('%Y-%m-%d') if df['SALES_LINE_CREATION_DATE'].min() else 'Unknown',
            'end': df['SALES_LINE_CREATION_DATE'].max().strftime('%Y-%m-%d') if df['SALES_LINE_CREATION_DATE'].max() else 'Unknown'
        },
        'unique_customers': df['CUSTOMER_NAME'].nunique(),
        'unique_regions': df['REGION'].unique().tolist(),
        'sales_order_statuses': df['SALES_ORDER_STATUS'].unique().tolist(),
        'total_sales_amount': df['LINE_AMOUNT_AFTER_DISCOUNT'].sum(),
        'customer_groups': df['CUSTOMER_GROUP'].unique().tolist()[:10]  # Limit to first 10
    }
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

        # Create user prompt with question and sample data
        user_prompt = f"""
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
            st.metric("Total Sales Amount", f"{df['LINE_AMOUNT_AFTER_DISCOUNT'].sum():,.0f} SAR")
            st.metric("Unique Customers", df['CUSTOMER_NAME'].nunique())
            
            # Show data freshness
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
            # Recent activity
            recent_sales = df[df['SALES_LINE_CREATION_DATE'] >= datetime.now() - timedelta(days=7)]
            st.metric("Sales This Week", len(recent_sales))
            
            # Top region
            top_region = df.groupby('REGION')['LINE_AMOUNT_AFTER_DISCOUNT'].sum().idxmax()
            st.metric("Top Region", top_region)
            
            # Internal vs External
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
