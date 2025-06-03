import streamlit as st
import numpy as np
import pickle
import time # For the loading spinner

# --- Page Configuration ---
st.set_page_config(
    page_title="Quick Stop Sales Predictor",
    page_icon="🏪", # A convenient store emoji icon
    layout="wide", # Use a wide layout for better spacing
    initial_sidebar_state="auto"
)

# --- Custom CSS for a cleaner look (optional, but can enhance aesthetics) ---
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 18px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 12px 0 rgba(0,0,0,0.3);
    }
    .stSuccess {
        background-color: #e6ffe6;
        color: #006600;
        border-left: 6px solid #4CAF50;
        padding: 10px;
        border-radius: 4px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    /* --- FIX: Ensure st.metric value text is visible --- */
    .stMetric > div[data-testid="stMetricValue"] {
        color: #333333; /* Dark gray color for better visibility */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header with Logo and Title ---
col_logo, col_title = st.columns([1, 4]) # Adjust column ratio as needed

with col_logo:
    # Placeholder logo image. Replace with your actual logo URL or local path.
    # For local files, ensure it's in the same directory or provide a correct path.
    # Example: st.image("path/to/your/logo.png", width=100)
    st.image("https://placehold.co/150x150/ADD8E6/000000?text=QuickStop", width=100)

with col_title:
    st.title("Quick Stop Sales Predictor 🏪")
    st.markdown("### Predict your daily sales based on various category inputs.")

st.markdown("---") # A horizontal line for separation

# --- Model Loading ---
# Ensure 'models/model (2).pkl' is the correct path relative to your app.py
model_path = 'models/model (2).pkl'
loaded_model = None # Initialize loaded_model to None

try:
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    st.success("Model loaded successfully! Ready for predictions.")
except FileNotFoundError:
    st.error(f"Error: Model file not found at '{model_path}'. Please ensure the file exists.")
    st.info("Make sure your 'models' folder is in the same directory as 'app.py' and contains 'model (2).pkl'.")
    st.stop() # Stop the app if the model can't be loaded
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()

st.markdown("---")

# --- Input Fields for Features ---
st.header("Enter Sales Category Data")
st.markdown("Adjust the values below to see how they impact the total sales prediction.")

# Use columns for a more organized input layout
col1, col2 = st.columns(2)

with col1:
    Groceries = st.number_input("Groceries ($)", min_value=0, value=100, step=10, help="Daily sales revenue from groceries.")
    Beer = st.number_input("Beer ($)", min_value=0, value=50, step=5, help="Daily sales revenue from beer.")
    Instant_Lottery = st.number_input("Instant Lottery ($)", min_value=0, value=10, step=1, help="Daily sales revenue from instant lottery tickets.")
    Online_Lottery = st.number_input("Online Lottery ($)", min_value=0, value=20, step=2, help="Daily sales revenue from online lottery tickets.")
    Tax = st.number_input("Tax ($)", min_value=0, value=10, step=1, help="Daily tax collected.")
    Grocery_Tax = st.number_input("Grocery (Tax) ($)", min_value=0, value=10, step=1, help="Daily sales revenue from taxable groceries.")


with col2:
    Electric_Ciggar = st.number_input("Electric Ciggar ($)", min_value=0, value=50, step=5, help="Daily sales revenue from electric cigarettes.")
    Grocery_Non_Tax = st.number_input("Grocery (Non-Tax) ($)", min_value=0, value=50, step=5, help="Daily sales revenue from non-taxable groceries.")
    PreRolls = st.number_input("PreRolls ($)", min_value=0, value=30, step=3, help="Daily sales revenue from pre-rolled items.")
    Bio_Gas = st.number_input("Bio Gas ($)", min_value=0, value=60, step=6, help="Daily sales revenue from bio-gas.")
    Tobacco = st.number_input("Tobacco ($)", min_value=0, value=80, step=8, help="Daily sales revenue from tobacco products.")


# --- Prediction Button and Logic ---
st.markdown("---") # Separator

if st.button("PREDICT TOTAL SALES"):
    if loaded_model is not None:
        with st.spinner('Predicting sales...'):
            time.sleep(1) # Simulate a short delay for prediction
            # Prepare the input data as a NumPy array
            # IMPORTANT: Ensure the order of features here matches the order
            # your model was trained on.
            input_data_array = np.array([
                Groceries, Beer, Instant_Lottery, Online_Lottery, Tax,
                Electric_Ciggar, Grocery_Non_Tax, PreRolls, Bio_Gas, Tobacco,
                Grocery_Tax
            ]).reshape(1, -1) # Reshape for a single sample

            try:
                prediction = loaded_model.predict(input_data_array)

                st.subheader("Predicted Daily Total Sales:")
                # Display prediction using st.metric for a prominent look
                st.metric(label="Estimated Sales", value=f"${prediction[0]:,.2f}")
                st.success("Prediction complete!")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please ensure the input data format matches what your model expects (e.g., number of features, scaling).")
    else:
        st.warning("Model not loaded. Please resolve the model loading issue before predicting.")

st.markdown("---")
st.info("This application provides an estimated sales figure based on your inputs. Results are for informational purposes.")
