import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
import random

# --- COLOR PALETTE ---
ACCENT_GREEN = '#4CAF50'
ACCENT_ORANGE = '#FF7F50'
ACCENT_BLUE = '#1E90FF'
HIGHLIGHT_COLOR = '#E8F5E9'  # Light green background

# --- 0. CONFIGURATION AND INITIALIZATION ---
st.set_page_config(
    layout="wide",
    page_title="Crop Price Forecaster",
    page_icon="💰"
)

# 1. CSS INJECTION TO REMOVE TOP SPACE AND DEFINE HEADER STYLE
st.markdown(
    """
    <style>
        /* 🎯 AGGRESSIVE CSS TO REMOVE TOP SPACE */

        /* Targets the entire app view block to remove default margin/padding */
        [data-testid="stAppViewBlock"] {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        /* Targets the main content container and applies normal page padding/margin */
        .block-container {
            padding-top: 0rem !important; 
            padding-bottom: 5rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }

        /* Hides the default Streamlit header/menu bar if it appears */
        div.stApp > header {
            display: none;
        }

        /* Class for the fixed-height header image (2cm height, full width) */
        .header-image {
            width: 100%;
            height: 2cm;
            object-fit: cover; /* Ensures the image covers the area without distortion */
            margin: 0;
            padding: 0;
            display: block; 
        }
        /* Navigation bar styling */
        .navbar-container {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 2px solid #f0f0f0;
            margin-bottom: 20px;
        }
        .stButton>button {
            width: 100%;
            font-weight: bold;
            border-radius: 8px;
        }
        /* Styles for the stacked image container */
        .stacked-images-container {
            display: flex;
            flex-direction: column;
            gap: 15px; /* Spacing between the stacked images */
        }

        /* Ensure H2 elements (used for st.header) are styled for better fit */
        h2 {
            white-space: nowrap; /* Prevents wrapping if possible */
            overflow: hidden;    /* Hides overflow if strictly necessary */
            text-overflow: ellipsis; /* Adds dots if overflow occurs */
            font-size: 1.8rem; /* Slightly reduced size for better fit */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for page control
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'results' not in st.session_state:
    st.session_state.results = {}


# --- 1. LOAD ASSETS (Model and Features) ---
# NOTE: The model and column loading logic is retained for functionality but removed from user-facing text.
@st.cache_resource
def load_assets():
    """Loads the model and feature columns list with error handling."""
    global rf_model, ALL_COLUMNS
    try:
        # CORRECTED PATHS for model and feature columns
        rf_model = joblib.load('crop_price_prediction/final_crop_price_predictor.joblib')
        ALL_COLUMNS = joblib.load('crop_price_prediction/feature_columns.joblib')
        return rf_model, ALL_COLUMNS
    except FileNotFoundError:
        st.error("Model or feature columns file not found. Please ensure 'final_crop_price_predictor.joblib' and 'feature_columns.joblib' are in the 'crop_price_prediction/' directory of your repository.")
        # Fallback for display purposes
        ALL_COLUMNS = ['Year', 'Month', 'Day', 'Grade_Encoded', 'District_Pune', 'Commodity_Wheat']
        rf_model = None
        return rf_model, ALL_COLUMNS
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model assets: {e}")
        # Fallback for display purposes
        ALL_COLUMNS = ['Year', 'Month', 'Day', 'Grade_Encoded', 'District_Pune', 'Commodity_Wheat']
        rf_model = None
        return rf_model, ALL_COLUMNS


rf_model, ALL_COLUMNS = load_assets()

# ... (rest of the code is unchanged until show_prediction_dashboard) ...

# --- 4. PREDICTION DASHBOARD FUNCTION (Stacked Images) ---
def show_prediction_dashboard():
    draw_navbar()

    # Changed st.title to st.header for better fit
    st.header("💡 Market Intelligence Dashboard")
    st.markdown("---")

    # Layout for stacked images (1.5 wide) and inputs (3 wide)
    dash_col_img, dash_col_inputs = st.columns([1.5, 3])

    with dash_col_img:
        st.markdown("### Focus on Inputs")

        st.markdown('<div class="stacked-images-container">', unsafe_allow_html=True)

        # Image 1 (Original)
        try:
            # CORRECTED PATH: Removed double 'crop_price_prediction/'
            st.image("crop_price_prediction/crop2.jpg", use_container_width=True,
                     caption="Select Your Specific Criteria.")
        except:
            st.warning("Image 'crop_price_prediction/crop2.jpg' not found. Using placeholder.")
            st.image("https://placehold.co/400x200/9ccc65/ffffff?text=Input+Parameters", use_container_width=True)

        # Image 2 (NEW, stacked below Image 1)
        try:
            st.image("crop_price_prediction/crop_page2.jpg", use_container_width=True, caption="Data-Based Analysis.")
        except:
            st.image("https://placehold.co/400x200/1E90FF/ffffff?text=Data+Driven+Insights", use_container_width=True,
                     caption="Analyze market data.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"***Select your crop, market, and month for the most accurate forecast. (Select your parameters for precise forecast).***")

    with dash_col_inputs:
        st.markdown("### 🎯 Set Prediction Parameters (Set Forecasting Parameters)")

        # --- INPUT SECTION 1: MARKET & GRADE ---
        with st.container(border=True):
            st.markdown("##### 📍 **Crop and Location Details (Crop & Location Details)**")
            input_cols_1 = st.columns(3)
            with input_cols_1[0]:
                selected_district = st.selectbox("Market District:", DISTRICT_OPTIONS, key='district_select')
            with input_cols_1[1]:
                selected_commodity = st.selectbox("Crop Commodity:", COMMODITY_OPTIONS, key='commodity_select')
            with input_cols_1[2]:
                st.markdown("##### ⭐ Quality Grade")
                grade_encoded = st.radio("Grade:", [1, 2, 3], index=2, horizontal=True, label_visibility="collapsed",
                                         help="1=Lowest, 3=Best", key='grade_radio')

            st.divider()

            # --- INPUT SECTION 2: TIME (Using Sliders) ---
            st.markdown("##### 📅 **Selection of Time for Sale (Select Selling Time)**")
            st.caption("Select the year and specific month for the forecast. (Select Year and Specific Month).")
            input_cols_2 = st.columns(2)

            with input_cols_2[0]:
                selected_year = st.slider("Prediction Year:", min_value=2024, max_value=2030, value=2025, step=1,
                                          key='year_slider')

            with input_cols_2[1]:
                selected_month = st.slider("Specific Forecast Month:", min_value=1, max_value=12, value=1, step=1,
                                           key='month_slider')

        # --- Action Button ---
        st.markdown("")
        predict_button = st.button("🚀 Generate Trends and Price Forecast", type="primary", use_container_width=True,
                                   key='main_forecast_button')

    # --- PREDICTION LOGIC (Triggers page switch) ---
    if predict_button:

        is_valid_selection = (selected_district != 'Select District...') and (
                selected_commodity != 'Select Commodity...')

        if not is_valid_selection:
            st.error("⚠️ Please select the Market District and Crop Commodity to proceed..")
            st.stop()

        if not rf_model:
            st.error("Model not loaded. Cannot run prediction.")
            st.stop()

        with st.spinner(f'Calculating 12-month forecast for {selected_commodity} in {selected_district}...'):

            # 1. Specific Prediction Input
            input_data = pd.Series(0, index=ALL_COLUMNS)
            input_data['Year'], input_data['Month'], input_data['Day'], input_data[
                'Grade_Encoded'] = selected_year, selected_month, 1, grade_encoded
            district_col_name = f'District_{selected_district}'
            commodity_col_name = f'Commodity_{selected_commodity}'
            if district_col_name in ALL_COLUMNS: input_data[district_col_name] = 1
            if commodity_col_name in ALL_COLUMNS: input_data[commodity_col_name] = 1

            predicted_price_specific = rf_model.predict(pd.DataFrame([input_data]))[0]

            # 2. Generate 12-month forecast (for the selected district)
            forecast_df = get_monthly_forecast(selected_district, selected_commodity, selected_year, grade_encoded)

            # 3. Generate comparison data (includes selected district and others)
            comparison_df = get_comparison_data(selected_commodity, selected_year, grade_encoded, selected_district,
                                                raw_districts, forecast_df)

        # --- 4. Store Results and Switch Page ---
        st.session_state.results = {
            'price': predicted_price_specific,
            'forecast_df': forecast_df,
            'comparison_df': comparison_df,
            'district': selected_district,
            'commodity': selected_commodity,
            'year': selected_year,
            'month': selected_month,
            'grade': grade_encoded
        }
        st.session_state.page = 'results'
        st.rerun()


# --- 5. RESULTS SCREEN FUNCTION (Attractive Price Output & Charts) ---
def show_results_screen():
    draw_navbar()

    results = st.session_state.results

    if not results or 'price' not in results:
        st.warning("No valid forecast data found. Returning to Dashboard.")
        st.session_state.page = 'dashboard'
        st.rerun()
        return

    # Changed st.title to st.header and adjusted text for better fit
    st.header(f"✅ Best Price Forecast ({results['commodity']} Price Projection)")
    st.markdown("---")

    # --- STYLISH MARATHI INTRO TEXT ---
    st.markdown(
        f"""
        <div style='background-color: #F0F9FF; padding: 20px; border-radius: 12px; border: 2px solid {ACCENT_BLUE}; text-align: center; margin-bottom: 25px;'>
            <h2 style='color: {ACCENT_BLUE}; margin: 0;'>**"Information is Success! All your results are ready.."**</h2>
            <p style='color: #333; margin: 5px 0 0 0; font-size: 0.9em;'>For the selected market, {results['district']}, your accurate price for the specific month, {results['month']}, is as follows:.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.balloons()

    # --- ROW 1: ATTRACTIVE PRICE OUTPUT AND 12-MONTH TREND ---
    price_cols = st.columns([1, 3])

    with price_cols[0]:

        # 🌟 Attractive Price Container (Specific Month Price)
        st.markdown(
            f"""
            <div style='background-color: {HIGHLIGHT_COLOR}; padding: 25px; border-radius: 10px; border-left: 8px solid {ACCENT_GREEN}; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
                <p style='font-size: 1.1em; color: #555;'>**Expected Selling Price for Month {results['month']}</p>
                <h1 style='color:{ACCENT_GREEN}; font-size: 3.5em; margin: 0;'>₹{results['price']:,.0f}</h1>
                <p style='font-size: 1.1em; color: #333; margin-top: 5px; font-weight: bold;'>per Quintal (100 kg)</p>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("---")

        # Attractive Key Details Box
        st.markdown(f"""
            <div style='padding: 15px; border-radius: 8px; border: 1px dashed {ACCENT_BLUE};'>
                <p style='font-weight: bold; color: {ACCENT_BLUE}; margin: 0;'>Key Details (Important Details):</p>
                <p style='margin: 5px 0 0 0;'>📅 **Prediction Year:** <span style='font-weight: bold;'>{results['year']}</span></p>
                <p style='margin: 0;'>⭐ **Quality Grade:** <span style='font-weight: bold;'>Grade {results['grade']}</span></p>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>*उत्तम ग्रेड (Grade {results['grade']}) Always delivers a higher price..*</p>
            </div>
            """, unsafe_allow_html=True
                    )

    with price_cols[1]:
        st.subheader(f"📈 12-Month Price Trend ({results['district']} Market Seasonal Analysis)")
        st.caption("See the expected prices throughout the year, based on historical fluctuations.")

        forecast_df = results['forecast_df']
        specific_month_data = forecast_df[forecast_df['Month'] == results['month']]

        base = alt.Chart(forecast_df).encode(
            x=alt.X('Month', axis=alt.Axis(format='d', title='Month of the Year'))
        )

        line_chart = base.mark_line(point=True, strokeWidth=3, color=ACCENT_GREEN).encode(
            y=alt.Y('Price', title='Predicted Price (₹/Quintal)', scale=alt.Scale(zero=False)),
            tooltip=['Month', alt.Tooltip('Price', format=',.2f')]
        )

        highlight = alt.Chart(specific_month_data).mark_circle(size=250, color=ACCENT_ORANGE).encode(
            x='Month',
            y='Price',
            tooltip=[alt.Tooltip('Price', format=',.2f')]
        )

        st.altair_chart(line_chart + highlight, use_container_width=True)

    st.markdown("---")

    # --- ROW 2: PRIMARY DISTRICT COMPARISON (BAR CHART: District vs Monthly Price) ---
    comparison_df = results['comparison_df']

    # Filter comparison data ONLY for the selected month
    comparison_for_month = comparison_df[comparison_df['Month'] == results['month']]

    st.subheader(f"📍 District Price Comparison for the Selected Month (Month {results['month']}))")
    st.caption(
        f"**This chart shows the expected price across other markets for your selected month. The Highest Price is highlighted in Green..**")

    # Determine the highest price for coloring
    max_price = comparison_for_month['Price'].max()

    # Bar chart: District on X (one side), Price on Y (the other side)
    bar_chart = alt.Chart(comparison_for_month).mark_bar().encode(
        # Y-axis: Price (the value)
        y=alt.Y('Price', title=f'Predicted Price in Month {results["month"]} (₹/Quintal)', scale=alt.Scale(zero=False)),
        # X-axis: District, sorted by Price descending
        x=alt.X('District', sort='-y', title='Market District'),
        color=alt.condition(
            alt.datum.Price == max_price,
            alt.value(ACCENT_GREEN),  # Green for the highest price
            alt.value(ACCENT_BLUE)  # Blue for others
        ),
        tooltip=['District', alt.Tooltip('Price', format=',.0f', title='Price')]
    )

    text_labels = bar_chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5  # Nudge text up slightly
    ).encode(
        text=alt.Text('Price', format=',.0f'),
        color=alt.value('black')
    )

    st.altair_chart(bar_chart + text_labels, use_container_width=True)

    st.markdown("---")

    # --- STYLISH MARATHI/ENGLISH CONCLUSION TEXT ---
    st.markdown(
        f"""
        <div style='background-color: #FFFBEA; padding: 25px; border-radius: 12px; border: 2px solid {ACCENT_ORANGE}; text-align: center;'>
            <h3 style='color: {ACCENT_ORANGE}; margin: 0 0 10px 0;'>**Make the best use of excellent timing and the market."** 💰</h3>
            <p style='color: #444; margin: 0; font-size: 1.1em; font-weight: bold;'>
                Use this intelligence to choose the best district and time to sell and maximize your profit.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    # --- ROW 3: FINAL IMAGE (Ensuring this stays) ---
    st.markdown("#### Future-Proof Your Agriculture. 🚀")
    try:
        # CORRECTED PATH
        st.image("crop_price_prediction/crop_last.jpg", use_container_width=True, caption="Informed decisions are key to profitability.")
    except:
        st.warning("Image 'crop_price_prediction/crop_last.jpg' not found. Using placeholder.")
        st.image("https://placehold.co/800x400/9ccc65/ffffff?text=Informed+Decisions",
                 use_container_width=True, caption="Placeholder: Informed decisions")


# --- 6. MAIN APP RUNNER ---

# 2. HEADER IMAGE IMPLEMENTATION (Full Width, 2cm Height)
try:
    st.markdown(
        f"""
        <img src='crop_price_prediction/top.jpeg' class='header-image' alt='App Header Image'>
        """,
        unsafe_allow_html=True
    )
except Exception:
    st.markdown(
        "<div style='height: 1cm; background-color: #333; color: white; text-align: center; line-height: 2cm; font-weight: bold; font-size: 1.2em;'>Crop Price Forecaster Header</div>",
        unsafe_allow_html=True)

if st.session_state.page == 'welcome':
    show_welcome_screen()
elif st.session_state.page == 'dashboard':
    show_prediction_dashboard()
elif st.session_state.page == 'results':
    show_results_screen()
