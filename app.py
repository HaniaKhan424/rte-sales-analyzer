import gspread
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from google.oauth2.service_account import Credentials
import os
from datetime import datetime, timedelta
import json

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Get the password from secrets
        correct_password = st.secrets.get("app_password", "default")
        
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "🔐 Enter Password to Access RTE Sales Analyzer", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.info("Please enter the password to access the sales data analyzer.")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "🔐 Enter Password to Access RTE Sales Analyzer", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😞 Password incorrect. Please try again.")
        return False
    else:
        # Password correct
        return True

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

def detect_date_column(df):
    """Detect the main date column in the dataset"""
    possible_date_columns = [
        'SALES_LINE_CREATION_DATE',
        'SALES_ORDER_CREATION_DATE', 
        'CREATION_DATE',
        'DATE',
        'SALES_DATE',
        'ORDER_DATE',
        'CREATED_DATE'
    ]
    
    # Check which date columns exist
    existing_date_columns = []
    for col in possible_date_columns:
        if col in df.columns:
            existing_date_columns.append(col)
    
    # Return the first available date column, or None if none exist
    return existing_date_columns[0] if existing_date_columns else None

def show_dataset_info(df):
    """Display dataset information for debugging"""
    st.subheader("🔍 Dataset Debug Information")
    
    # Show total columns and any duplicates
    st.write(f"**Total Columns:** {len(df.columns)}")
    
    # Check for any remaining issues
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        st.warning(f"**Still duplicated columns:** {duplicate_cols}")
    
    # Show column names (first 20 to avoid clutter)
    st.write("**Available Columns (first 20):**")
    cols_to_show = df.columns.tolist()[:20]
    st.write(cols_to_show)
    if len(df.columns) > 20:
        st.write(f"... and {len(df.columns) - 20} more columns")
    
    # Show date columns specifically
    date_columns = [col for col in df.columns if any(word in col.upper() for word in ['DATE', 'TIME', 'CREATED'])]
    if date_columns:
        st.write("**Date-related Columns:**")
        st.write(date_columns)
    
    # Show first few rows (only first 10 columns to avoid display issues)
    st.write("**First 5 rows (first 10 columns):**")
    try:
        display_df = df.iloc[:5, :10]  # First 5 rows, first 10 columns
        st.dataframe(display_df)
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")
        # Try showing just column info
        st.write("**Column names and types:**")
        col_info = pd.DataFrame({
            'Column': df.columns[:10],
            'Type': [str(df[col].dtype) for col in df.columns[:10]]
        })
        st.dataframe(col_info)
    
    # Show data types for first 10 columns
    st.write("**Data Types (first 10 columns):**")
    types_df = pd.DataFrame({
        'Column': df.columns[:10],
        'Type': [str(df[col].dtype) for col in df.columns[:10]]
    })
    st.dataframe(types_df)

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
        headers = data[1]  # Row 2 contains headers
        rows = data[2:]    # Row 3 onwards contains data
        
        # Create DataFrame and fix duplicate column names
        df = pd.DataFrame(rows, columns=headers)
        
        # Fix duplicate column names by adding suffixes
        df.columns = pd.Index(df.columns).to_series().groupby(level=0).cumcount().astype(str).replace('0','') + df.columns
        df.columns = df.columns.str.replace(r'^(\d+)', r'_\1', regex=True)  # Add underscore before numbers
        
        # Clean up column names (remove extra spaces, etc.)
        df.columns = df.columns.str.strip()
        
        # Preprocess the data
        df = preprocess_data(df)
        
        return df, None
    except Exception as e:
        return None, str(e)

def preprocess_data(df):
    """Clean and preprocess the sales data with safety checks"""
    # Suppress warnings during preprocessing
    import warnings
    warnings.filterwarnings('ignore')
    
    # Convert date columns to datetime with safety checks
    date_columns = [
        'SALES_LINE_CREATION_DATE', 
        'SALES_ORDER_CREATION_DATE',
        'SALES_LINE_REQUESTED_SHIPPING_DATE', 
        'SALES_ORDER_REQUESTED_SHIPPING_DATE',
        'LAST_UPDATED_AT'
    ]
    
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col].astype(str), errors='coerce')
            except Exception:
                pass  # Silently skip problematic columns during initial load
    
    # Convert numeric columns with safety checks
    numeric_columns = ['LINE_AMOUNT', 'LINE_AMOUNT_AFTER_DISCOUNT', 'SALES_QUANTITY', 'DISCOUNT_PERCENTAGE']
    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
            except Exception:
                pass  # Silently skip problematic columns during initial load
    
    # Fill NaN values appropriately
    if 'IS_SISTER_COMPANY' in df.columns:
        df['IS_SISTER_COMPANY'] = df['IS_SISTER_COMPANY'].fillna('NO')
    if 'REGION' in df.columns:
        df['REGION'] = df['REGION'].fillna('Unknown')
    
    return df

def get_comprehensive_data_summary(df, user_question):
    """Generate comprehensive summary of the entire dataset with safety checks"""
    summary = []
    
    # Detect the main date column
    date_column = detect_date_column(df)
    
    # Get date contexts
    dates = get_date_contexts()
    today = datetime.strptime(dates['today'], '%Y-%m-%d').date()
    yesterday = datetime.strptime(dates['yesterday'], '%Y-%m-%d').date()
    week_start = datetime.strptime(dates['week_start'], '%Y-%m-%d').date()
    month_start = datetime.strptime(dates['month_start'], '%Y-%m-%d').date()
    
    # Basic dataset info
    summary.append(f"COMPLETE DATASET OVERVIEW ({len(df):,} total records):")
    
    # Date analysis (only if date column exists)
    if date_column and date_column in df.columns:
        try:
            date_min = df[date_column].min()
            date_max = df[date_column].max()
            if pd.notna(date_min) and pd.notna(date_max):
                summary.append(f"- Date range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
                summary.append(f"- Data spans: {(date_max - date_min).days} days")
                summary.append(f"- Using date column: {date_column}")
        except Exception as e:
            summary.append(f"- Date analysis unavailable (Error: {str(e)})")
    else:
        summary.append("- No date column found for time-based analysis")
        summary.append(f"- Available columns: {', '.join(df.columns.tolist()[:10])}...")
    
    # Customer analysis
    if 'CUSTOMER_NAME' in df.columns:
        summary.append(f"- Unique customers: {df['CUSTOMER_NAME'].nunique():,}")
    
    # Sales amount analysis
    if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        total_sales = df['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
        summary.append(f"- Total sales amount: {total_sales:,.2f} SAR")
    elif 'LINE_AMOUNT' in df.columns:
        total_sales = df['LINE_AMOUNT'].sum()
        summary.append(f"- Total sales amount (before discount): {total_sales:,.2f} SAR")
    
    # Time-based analysis (only if date column exists)
    if date_column and date_column in df.columns:
        summary.append(f"\nTIME-BASED ANALYSIS (using {date_column}):")
        
        try:
            # Yesterday's data
            yesterday_data = df[df[date_column].dt.date == yesterday]
            if len(yesterday_data) > 0:
                if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                    yesterday_sales = yesterday_data['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                    summary.append(f"- Yesterday ({yesterday}): {len(yesterday_data)} transactions, {yesterday_sales:,.2f} SAR")
                else:
                    summary.append(f"- Yesterday ({yesterday}): {len(yesterday_data)} transactions")
                
                # Yesterday's top performers
                if 'CUSTOMER_GROUP' in yesterday_data.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                    yesterday_reps = yesterday_data.groupby('CUSTOMER_GROUP')['LINE_AMOUNT_AFTER_DISCOUNT'].sum().sort_values(ascending=False)
                    summary.append("  Top performers yesterday:")
                    for rep, amount in yesterday_reps.head(3).items():
                        summary.append(f"    • {rep}: {amount:,.2f} SAR")
            else:
                # Find most recent date if yesterday has no data
                recent_dates = df[date_column].dropna().dt.date.unique()
                if len(recent_dates) > 0:
                    most_recent = max(recent_dates)
                    recent_data = df[df[date_column].dt.date == most_recent]
                    if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                        recent_sales = recent_data['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                        summary.append(f"- No data for yesterday ({yesterday})")
                        summary.append(f"- Most recent data ({most_recent}): {len(recent_data)} transactions, {recent_sales:,.2f} SAR")
                    else:
                        summary.append(f"- Most recent data ({most_recent}): {len(recent_data)} transactions")
            
            # This week's data
            this_week_data = df[df[date_column].dt.date >= week_start]
            if len(this_week_data) > 0:
                if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                    week_sales = this_week_data['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                    summary.append(f"- This week (from {week_start}): {len(this_week_data)} transactions, {week_sales:,.2f} SAR")
                else:
                    summary.append(f"- This week (from {week_start}): {len(this_week_data)} transactions")
            
            # This month's data
            this_month_data = df[df[date_column].dt.date >= month_start]
            if len(this_month_data) > 0:
                if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                    month_sales = this_month_data['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                    summary.append(f"- This month (from {month_start}): {len(this_month_data)} transactions, {month_sales:,.2f} SAR")
                else:
                    summary.append(f"- This month (from {month_start}): {len(this_month_data)} transactions")
                    
        except Exception as e:
            summary.append(f"- Time-based analysis error: {str(e)}")
    
    # Sales by region (if available)
    if 'REGION' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary.append(f"\nSALES BY REGION (complete dataset):")
        try:
            region_sales = df.groupby('REGION')['LINE_AMOUNT_AFTER_DISCOUNT'].agg(['count', 'sum']).round(2)
            region_sales = region_sales.sort_values('sum', ascending=False)
            for region, data in region_sales.head(10).iterrows():
                summary.append(f"- {region}: {data['count']:,} transactions, {data['sum']:,.2f} SAR")
        except Exception as e:
            summary.append(f"- Region analysis error: {str(e)}")
    
    # Top customers (if available)
    if 'CUSTOMER_NAME' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary.append(f"\nTOP 10 CUSTOMERS (complete dataset):")
        try:
            top_customers = df.groupby('CUSTOMER_NAME')['LINE_AMOUNT_AFTER_DISCOUNT'].sum().sort_values(ascending=False).head(10)
            for i, (customer, amount) in enumerate(top_customers.items(), 1):
                summary.append(f"- {i}. {customer}: {amount:,.2f} SAR")
        except Exception as e:
            summary.append(f"- Customer analysis error: {str(e)}")
    
    # Sales representatives performance (if available)
    if 'CUSTOMER_GROUP' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
        summary.append(f"\nSALES REPRESENTATIVES PERFORMANCE (complete dataset):")
        try:
            rep_sales = df.groupby('CUSTOMER_GROUP')['LINE_AMOUNT_AFTER_DISCOUNT'].agg(['count', 'sum']).round(2)
            rep_sales = rep_sales.sort_values('sum', ascending=False).head(10)
            for rep, data in rep_sales.iterrows():
                summary.append(f"- {rep}: {data['count']:,} transactions, {data['sum']:,.2f} SAR")
        except Exception as e:
            summary.append(f"- Sales rep analysis error: {str(e)}")
    
    return "\n".join(summary)

def get_targeted_data_sample(df, user_question):
    """Get targeted sample based on the specific question with safety checks"""
    question_lower = user_question.lower()
    
    # Detect the main date column
    date_column = detect_date_column(df)
    
    # Time-based questions
    if any(word in question_lower for word in ['yesterday', 'today']) and date_column:
        dates = get_date_contexts()
        target_date = datetime.strptime(dates['yesterday'], '%Y-%m-%d').date()
        
        try:
            filtered_df = df[df[date_column].dt.date == target_date]
            if len(filtered_df) > 0:
                return f"YESTERDAY'S COMPLETE DATA ({target_date}) using {date_column} - {len(filtered_df)} records:\n" + filtered_df.head(20).to_string(index=False)
            else:
                # Get most recent date
                recent_date = df[date_column].dt.date.max()
                recent_df = df[df[date_column].dt.date == recent_date]
                return f"No data for yesterday ({target_date}). Most recent data ({recent_date}) using {date_column} - {len(recent_df)} records:\n" + recent_df.head(20).to_string(index=False)
        except Exception as e:
            return f"Error filtering by date: {str(e)}\nShowing recent data instead:\n" + df.head(20).to_string(index=False)
    
    # Customer-based questions
    elif any(word in question_lower for word in ['customer', 'client']):
        if 'CUSTOMER_NAME' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
            try:
                top_customers = df.groupby('CUSTOMER_NAME')['LINE_AMOUNT_AFTER_DISCOUNT'].sum().sort_values(ascending=False).head(5)
                sample_data = df[df['CUSTOMER_NAME'].isin(top_customers.index)].head(20)
                return f"SAMPLE FROM TOP CUSTOMERS:\n" + sample_data.to_string(index=False)
            except Exception as e:
                return f"Error analyzing customers: {str(e)}\nShowing sample data:\n" + df.head(20).to_string(index=False)
    
    # Default: most recent data or first 20 rows
    if date_column and date_column in df.columns:
        try:
            recent_data = df.sort_values(date_column, ascending=False).head(20)
            return f"MOST RECENT DATA (sorted by {date_column}):\n" + recent_data.to_string(index=False)
        except Exception as e:
            return f"Error sorting by date: {str(e)}\nShowing first 20 rows:\n" + df.head(20).to_string(index=False)
    else:
        return f"SAMPLE DATA (first 20 rows):\n" + df.head(20).to_string(index=False)

def analyze_data_with_claude(df, user_question):
    """Analyze data using Claude API with full dataset access and safety checks"""
    try:
        # Initialize Claude API
        client = Anthropic(api_key=st.secrets["anthropic_api_key"])
        
        # Get comprehensive data summary from entire dataset
        comprehensive_summary = get_comprehensive_data_summary(df, user_question)
        
        # Get targeted sample based on question
        targeted_sample = get_targeted_data_sample(df, user_question)
        
        # Generate current date context
        dates = get_date_contexts()
        
        # Detect available date column
        date_column = detect_date_column(df)
        
        # Get actual date range from dataset (if date column exists)
        date_range_info = ""
        if date_column and date_column in df.columns:
            try:
                latest_date = df[date_column].max()
                earliest_date = df[date_column].min()
                
                if pd.notna(latest_date) and pd.notna(earliest_date):
                    date_range_info = f"""
ACTUAL DATA DATE RANGE (using {date_column}):
- Earliest date in dataset: {earliest_date.strftime('%Y-%m-%d')}
- Latest date in dataset: {latest_date.strftime('%Y-%m-%d')}
- Total date span: {(latest_date - earliest_date).days} days
"""
                else:
                    date_range_info = f"\nDATE COLUMN ISSUES: Found {date_column} but contains invalid dates"
            except Exception as e:
                date_range_info = f"\nDATE ANALYSIS ERROR: {str(e)}"
        else:
            date_range_info = f"\nNO DATE COLUMN FOUND. Available columns: {', '.join(df.columns.tolist()[:10])}..."
        
        date_context = f"""
CURRENT DATE CONTEXT:
- Today: {dates['today_formatted']} ({dates['today']})
- Yesterday: {dates['yesterday_formatted']} ({dates['yesterday']})
- This week started: {dates['week_start_formatted']} ({dates['week_start']})
- This month started: {dates['month_start_formatted']} ({dates['month_start']})

{date_range_info}

IMPORTANT: All analysis above is based on the COMPLETE dataset of {len(df):,} records.
If date-based filtering isn't available, analysis will be based on the full dataset without time filtering.
"""
        
        # Create system prompt
        system_prompt = f"""You are a data analyst for RTE/Raqtan with FULL ACCESS to the complete sales dataset.

You have access to ALL {len(df):,} records in the dataset. The comprehensive summary below contains aggregated information from the ENTIRE dataset, not just a sample.

AVAILABLE COLUMNS: {', '.join(df.columns.tolist())}

COLUMN DESCRIPTIONS (if available):
- SALES_ID: Unique identifier for each sales line
- CUSTOMER_NAME: Name of the customer
- CUSTOMER_ACCOUNT: Customer account number
- ITEM_ID: Product/service identifier
- SALES_ORDER_STATUS: Status of orders
- LINE_AMOUNT_AFTER_DISCOUNT: Final amount after discounts (USE THIS for financial calculations)
- SALES_QUANTITY: Quantity sold
- CUSTOMER_GROUP: Sales person name (may start with region prefix: J-, R-, K-)
- REGION: Geographic region
- IS_SISTER_COMPANY: YES = internal job, NO = external sale
- Currency: Saudi Riyals (SAR)

ANALYSIS INSTRUCTIONS:
1. Base your analysis on the COMPLETE dataset summary provided
2. Show specific calculations and numbers from the full dataset
3. If date filtering isn't available, provide analysis based on complete dataset
4. Provide actionable business insights
5. Format large numbers with commas for readability
6. Be conversational and business-focused in your response
7. If certain columns aren't available, work with what's available and mention limitations"""

        # Create user prompt
        user_prompt = f"""
{date_context}

COMPLETE DATASET SUMMARY:
{comprehensive_summary}

QUESTION: {user_question}

TARGETED DATA SAMPLE (for reference):
{targeted_sample}

Please analyze based on the complete dataset summary above. Work with the available columns and data, and mention any limitations if certain expected columns are missing.
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

def main():
    """Main Streamlit application"""
    
    # Check password first - this will control access to the entire app
    if not check_password():
        return  # Don't show anything else if password is incorrect
    
    # Only show the main app if password is correct
    st.title("🏢 RTE/Raqtan Sales Data Analyzer")
    st.markdown("Ask questions about your sales data and get AI-powered insights from Claude.")
    
    # Add logout button in sidebar
    with st.sidebar:
        if st.button("🚪 Logout"):
            # Clear the password session state to log out
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
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
            
            # Show debug info
            if st.checkbox("Show Debug Info"):
                show_dataset_info(df)
            
            # Safe metrics calculation
            if 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                total_sales = df['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                st.metric("Total Sales Amount", f"{total_sales:,.0f} SAR")
            elif 'LINE_AMOUNT' in df.columns:
                total_sales = df['LINE_AMOUNT'].sum()
                st.metric("Total Sales Amount", f"{total_sales:,.0f} SAR")
            
            if 'CUSTOMER_NAME' in df.columns:
                unique_customers = df['CUSTOMER_NAME'].nunique()
                st.metric("Unique Customers", unique_customers)
            
            # Show data freshness
            date_column = detect_date_column(df)
            if date_column and date_column in df.columns:
                try:
                    latest_date = df[date_column].max()
                    if pd.notna(latest_date):
                        st.info(f"Latest data: {latest_date.strftime('%Y-%m-%d')} (from {date_column})")
                except:
                    st.warning("Date column found but contains invalid data")
            else:
                st.warning("No date column detected")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Question")
        
        # Sample questions
        sample_questions = [
            "Who did the most sales yesterday?",
            "Which customer did less business with us in the last quarter?",
            "What is the total sales amount for each region?",
            "Show me the top 5 customers by sales volume",
            "Which sales person (customer group) has the highest sales?",
            "How many orders are still open?",
            "What is the average discount percentage given?",
            "Compare internal vs external sales"
        ]
        
        selected_question = st.selectbox(
            "Choose a sample question or type your own:",
            ["Type your own question..."] + sample_questions
        )
        
        if selected_question != "Type your own question...":
            user_question = st.text_area("Your question:", value=selected_question, height=100)
        else:
            user_question = st.text_area("Your question:", height=100, 
                                       placeholder="e.g., Who are the top 3 customers by sales amount?")
    
    with col2:
        st.subheader("📋 Quick Stats")
        if df is not None:
            # Recent activity (safe calculation)
            date_column = detect_date_column(df)
            if date_column and date_column in df.columns:
                try:
                    recent_sales = df[df[date_column] >= datetime.now() - timedelta(days=7)]
                    st.metric("Sales This Week", len(recent_sales))
                except:
                    st.metric("Sales This Week", "N/A")
            
            # Top region (safe calculation)
            if 'REGION' in df.columns and 'LINE_AMOUNT_AFTER_DISCOUNT' in df.columns:
                try:
                    top_region_data = df.groupby('REGION')['LINE_AMOUNT_AFTER_DISCOUNT'].sum()
                    if not top_region_data.empty:
                        top_region = top_region_data.idxmax()
                        st.metric("Top Region", top_region)
                except:
                    st.metric("Top Region", "N/A")
            
            # Internal vs External (safe calculation)
            if 'IS_SISTER_COMPANY' in df.columns:
                try:
                    internal_pct = (df['IS_SISTER_COMPANY'] == 'YES').mean() * 100
                    st.metric("Internal Sales %", f"{internal_pct:.1f}%")
                except:
                    st.metric("Internal Sales %", "N/A")
    
    # Analyze button
    if st.button("🔍 Analyze Full Dataset", type="primary", use_container_width=True):
        if not user_question.strip():
            st.error("Please enter a question to analyze.")
        else:
            with st.spinner("Analyzing complete dataset with Claude AI..."):
                answer, error = analyze_data_with_claude(df, user_question)
                
                if error:
                    st.error(f"Error during analysis: {error}")
                    
                    # Show helpful debugging information
                    st.subheader("🔧 Debug Information")
                    st.write("**Available columns:**")
                    st.write(df.columns.tolist())
                    
                    date_column = detect_date_column(df)
                    if date_column:
                        st.write(f"**Detected date column:** {date_column}")
                    else:
                        st.write("**No date column detected**")
                        
                    st.write("**First few rows:**")
                    st.dataframe(df.head())
                else:
                    st.subheader("📈 Complete Dataset Analysis")
                    st.markdown(answer)
                    
                    # Show analysis details
                    date_column = detect_date_column(df)
                    date_info = f" | Using date column: {date_column}" if date_column else " | No date column available"
                    st.caption(f"Analysis based on {len(df):,} total records{date_info} | Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and Claude AI for RTE/Raqtan sales data analysis*")

if __name__ == "__main__":
    main()
