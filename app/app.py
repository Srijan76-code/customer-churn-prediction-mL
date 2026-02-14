"""
app.py

Streamlit entry point for the Customer Churn Prediction web application.
Provides an interactive UI for uploading data, running predictions, and
visualising model results.
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# # ********************
#     # Page Configuration
# # ********************

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide"
)



# # ********************
#     # Load Model
# # ********************
# @st.cache_resource
# def load_model():
#     try:
#         with open("models/trained_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         return model
#     except Exception as e:
#         return None

# model = load_model()

st.title("Customer Churn Intelligence Dashboard")
st.caption("Predict and understand customer churn with our interactive dashboard.")


st.subheader(" Customer Input")

tab_manual,tab_csv, tab_single = st.tabs([
    "➕ Manual Multi-Customer",
    "📂 Upload CSV",
    "👤 Single Customer"
])



# ==============================
# 🥇 MODE 1 — Single Customer
# ==============================

with tab_single:

    st.markdown("### Enter Customer Details")

    # with st.form("single_customer_form", clear_on_submit=False):
    with st.expander("View Current Customer Details", expanded=True):
        st.markdown("#### Profile")

        
        p1, p2, p3, p4 = st.columns(4)

        with p1:
            gender = st.selectbox("Gender", ["Male", "Female"], key="sc_gender")

        with p2:
            married = st.selectbox("Married", ["Yes", "No"], key="sc_married")

        with p3:
            dependents = st.number_input(
                "Dependents", 0, 10, 0, key="sc_dependents"
            )

        with p4:
            referrals = st.number_input(
                "Referrals", 0, 20, 0, key="sc_referrals"
            )


        # st.divider()
        st.markdown("#### Account")

        a1, a2, a3, a4 = st.columns(4)

        with a1:
            offer = st.selectbox(
                "Offer",
                ["None", "Offer A", "Offer B", "Offer C"],
                key="sc_offer"
            )

        with a2:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                key="sc_contract"
            )

        with a3:
                paperless = st.selectbox(
                    "Paperless Billing",
                    ["Yes", "No"],
                    key="sc_paperless"
                )

        with a4:
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Credit Card", "Bank Withdrawal", "Mailed Check"],
                    key="sc_payment"
                )

        # st.divider()
        st.markdown("#### Customer Metrics")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            age = st.slider("Age", 18, 100, 40, key="sc_age")

        with m2:
            tenure = st.slider("Tenure (Months)", 0, 72, 12, key="sc_tenure")

        # spacers keep grid symmetry
        with m3:
            st.empty()

        with m4:
            st.empty()



        st.divider()
        st.markdown("#### Charges")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            monthly_charge = st.number_input(
                "Monthly Charge", 0.0, value=70.0, key="sc_monthly"
            )
            refunds = st.number_input(
                "Total Refunds", 0.0, value=0.0, key="sc_refunds"
            )

        with c2:
            total_charges = st.number_input(
                "Total Charges", 0.0, value=1000.0, key="sc_total"
            )
            extra_data = st.number_input(
                "Total Extra Data Charges", 0.0, value=0.0, key="sc_extra"
            )

        with c3:
            total_revenue = st.number_input(
                "Total Revenue", 0.0, value=1000.0, key="sc_revenue"
            )
            long_distance_total = st.number_input(
                "Total Long Distance Charges", 0.0, value=0.0, key="sc_total_ld"
            )

        with c4:
            avg_long_dist = st.number_input(
                "Avg Monthly Long Distance",
                0.0,
                value=10.0,
                key="sc_avg_ld"
            )
            avg_gb = st.number_input(
                "Avg Monthly GB Download",
                0.0,
                value=50.0,
                key="sc_avg_gb"
            )


        st.divider()
        st.markdown("#### Services")

            #  NOW INSIDE THE FORM (IMPORTANT FIX)
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="sc_phone")
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                key="sc_internet_service"
            )
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"], key="sc_tv")

        with s2:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"], key="sc_lines")
            internet_type = st.selectbox(
                "Internet Type",
                ["Cable", "DSL", "Fiber"],
                key="sc_internet_type"
            )
            streaming_movies = st.selectbox(
                    "Streaming Movies", ["Yes", "No"], key="sc_movies"
            )

        with s3:
            online_security = st.selectbox(
                "Online Security", ["Yes", "No"], key="sc_security"
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No"], key="sc_backup"
            )
            streaming_music = st.selectbox(
                "Streaming Music", ["Yes", "No"], key="sc_music"
            )

        with s4:
            device_protection = st.selectbox(
                "Device Protection Plan", ["Yes", "No"], key="sc_device"
            )
            tech_support = st.selectbox(
                "Premium Tech Support", ["Yes", "No"], key="sc_tech"
            )
            unlimited_data = st.selectbox(
                "Unlimited Data", ["Yes", "No"], key="sc_unlimited"
            )

    # ==============================
# 📊 LIVE CUSTOMER SNAPSHOT
# ==============================
    customer_data = {
    "Gender": gender,
    "Age": age,
    "Married": married,
    "Dependents": dependents,
    "Referrals": referrals,
    "Tenure (Months)": tenure,
    "Offer": offer,
    "Contract": contract,
    "Paperless Billing": paperless,
    "Payment Method": payment_method,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Internet Type": internet_type,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Streaming Music": streaming_music,
    "Unlimited Data": unlimited_data,
    "Avg Long Distance": avg_long_dist,
    "Avg GB Download": avg_gb,
    "Monthly Charge": monthly_charge,
    "Total Charges": total_charges,
    "Refunds": refunds,
    "Extra Data Charges": extra_data,
    "Total Long Distance": long_distance_total,
    "Total Revenue": total_revenue,
}

    st.markdown("### Live Customer Snapshot")

    with st.expander("View Current Customer Data", expanded=True):
        with st.container(border=False):
            st.dataframe(
            pd.DataFrame([customer_data]),
            use_container_width=True
        )


        # =================================
        #  SUBMIT
        # =================================

    predict_btn = st.button("Predict Churn", type="primary", use_container_width=False)




with tab_csv:

    st.markdown("### Batch Prediction via CSV")

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)

        with st.expander("View Current Customer Data", expanded=True):
            st.write("Preview:", df_uploaded.head())

        if st.button("Predict Batch", type="primary", use_container_width=False):
            st.success(f"Loaded {len(df_uploaded)} customers (connect model)")



# ==============================
# 🥉 TAB 3 — Synthetic Testing
# ==============================
with tab_manual:

    st.markdown("### Synthetic Customer Testing")
    st.caption("Generate random customers to stress-test the churn model.")



    input_col, btn_col, _ = st.columns([2, 1, 2])

    with input_col:
        n_rows = st.number_input(
            "Number of customers to generate",
            min_value=1,
            max_value=10000,
            value=5,
            step=1
        )

    with btn_col:
        # 🔥 PERFECT vertical alignment trick
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

        generate_btn = st.button(
            "Generate & Predict",
            use_container_width=True
        )


    # ==============================
    #  RANDOM DATA FUNCTION
    # ==============================
    def generate_random_customers(n):

        rng = np.random.default_rng()

        df = pd.DataFrame({
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 80, n),
            "Married": rng.choice(["Yes", "No"], n),
            "Dependents": rng.integers(0, 5, n),
            "Referrals": rng.integers(0, 10, n),
            "Tenure (Months)": rng.integers(0, 72, n),
            "Offer": rng.choice(["None", "Offer A", "Offer B", "Offer C"], n),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], n
            ),
            "Paperless Billing": rng.choice(["Yes", "No"], n),
            "Payment Method": rng.choice(
                ["Credit Card", "Bank Withdrawal", "Mailed Check"], n
            ),
            "Phone Service": rng.choice(["Yes", "No"], n),
            "Multiple Lines": rng.choice(["Yes", "No"], n),
            "Internet Service": rng.choice(
                ["DSL", "Fiber optic", "No"], n
            ),
            "Internet Type": rng.choice(["Cable", "DSL", "Fiber"], n),
            "Online Security": rng.choice(["Yes", "No"], n),
            "Online Backup": rng.choice(["Yes", "No"], n),
            "Device Protection": rng.choice(["Yes", "No"], n),
            "Tech Support": rng.choice(["Yes", "No"], n),
            "Streaming TV": rng.choice(["Yes", "No"], n),
            "Streaming Movies": rng.choice(["Yes", "No"], n),
            "Streaming Music": rng.choice(["Yes", "No"], n),
            "Unlimited Data": rng.choice(["Yes", "No"], n),
            "Avg Long Distance": rng.uniform(0, 50, n),
            "Avg GB Download": rng.uniform(0, 500, n),
            "Monthly Charge": rng.uniform(20, 120, n),
            "Total Charges": rng.uniform(100, 8000, n),
            "Refunds": rng.uniform(0, 50, n),
            "Extra Data Charges": rng.uniform(0, 100, n),
            "Total Long Distance": rng.uniform(0, 200, n),
            "Total Revenue": rng.uniform(100, 10000, n),
        })


        return df


    if generate_btn:

        with st.spinner("Generating synthetic customers..."):
            df_random = generate_random_customers(n_rows)

        st.success(f"Generated {len(df_random)} customers")

        with st.expander("View Current Customer Data", expanded=True):

            st.dataframe(df_random.head(50), use_container_width=True)
       
