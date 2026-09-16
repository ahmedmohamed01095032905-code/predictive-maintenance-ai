import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🏭",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =========================================
       MAIN APP
    ========================================= */

    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* =========================================
       HIDE STREAMLIT TOP BAR
    ========================================= */

    header[data-testid="stHeader"] {
        background-color: #000000 !important;
        height: 0px !important;
        min-height: 0px !important;
    }

    header[data-testid="stHeader"] > div {
        background-color: #000000 !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* =========================================
       SIDEBAR
    ========================================= */

    section[data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid #222222;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }


    /* =========================================
       HEADERS
    ========================================= */

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #888888;
        font-size: 15px;
        margin-bottom: 35px;
    }


    /* =========================================
       INPUT LABELS
    ========================================= */

    label {
        color: #CCCCCC !important;
        font-weight: 500 !important;
    }


    /* =========================================
       NUMBER INPUT
    ========================================= */

    div[data-baseweb="input"] {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="input"] input {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        background-color: #111111 !important;
    }

    input {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }


    /* =========================================
       SELECTBOX
    ========================================= */

    div[data-baseweb="select"] {
        background-color: #111111 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] input {
        color: #FFFFFF !important;
    }


    /* =========================================
       SELECTBOX DROPDOWN
    ========================================= */

    div[role="listbox"] {
        background-color: #111111 !important;
    }

    div[role="option"] {
        background-color: #111111 !important;
        color: #FFFFFF !important;
    }

    div[role="option"]:hover {
        background-color: #222222 !important;
    }


    /* =========================================
       BUTTON
    ========================================= */

    .stButton > button {
        width: 100%;
        background-color: #1479FF;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #0D63D6;
        color: #FFFFFF !important;
    }


    /* =========================================
       METRICS
    ========================================= */

    div[data-testid="stMetric"] {
        background-color: #111111;
        border: 1px solid #292929;
        border-radius: 12px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #888888 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700;
    }


    /* =========================================
       DATAFRAME
    ========================================= */

    div[data-testid="stDataFrame"] {
        border: 1px solid #252525;
        border-radius: 8px;
    }


    /* =========================================
       FOOTER
    ========================================= */

    .footer {
        text-align: center;
        color: #555555;
        font-size: 12px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #151515;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL FILES
# =========================================================

try:

    with open("logistic_failure_24h.pkl", "rb") as file:
        package = pickle.load(file)

    model = package["model"]
    THRESHOLD = package["threshold"]

    with open("ohe1_encoder.pkl", "rb") as file:
        machine_encoder = pickle.load(file)

    with open("ohe2_encoder.pkl", "rb") as file:
        operating_encoder = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

except Exception as e:

    st.error("Could not load the model files.")

    st.code(str(e))

    st.stop()


# =========================================================
# FEATURE ORDER
# =========================================================

feature_names = [
    "machine_id",
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
    "CNC",
    "Compressor",
    "Pump",
    "Robotic Arm",
    "idle",
    "normal",
    "peak"
]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:800;
            color:white;
        ">
            🏭 Predictive Maintenance
        </div>

        <div style="
            color:#777777;
            font-size:13px;
            margin-top:5px;
        ">
            Industrial AI System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Prediction",
            "Data Dashboard"
        ]
    )

    st.markdown("---")

    st.caption(
        "AI-based machine failure prediction"
    )


# =========================================================
# PAGE 1
# PREDICTION
# =========================================================

if page == "Prediction":

    st.markdown(
        '<div class="main-title">'
        'Predictive Maintenance AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Predict machine failure within the next 24 hours'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # MACHINE INFORMATION
    # =====================================================

    st.markdown("### Machine Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        machine_id = st.selectbox(
            "Machine ID",
            [f"M{i:02d}" for i in range(1, 21)]
        )

    with col2:

        machine_type = st.selectbox(
            "Machine Type",
            [
                "CNC",
                "Compressor",
                "Pump",
                "Robotic Arm"
            ]
        )

    with col3:

        operating_mode = st.selectbox(
            "Operating Mode",
            [
                "idle",
                "normal",
                "peak"
            ]
        )


    # =====================================================
    # SENSOR DATA
    # =====================================================

    st.markdown("### Sensor Data")

    col1, col2, col3 = st.columns(3)

    with col1:

        vibration_rms = st.number_input(
            "Vibration RMS",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1
        )

        temperature_motor = st.number_input(
            "Motor Temperature",
            min_value=-20.0,
            max_value=150.0,
            value=70.0,
            step=0.5
        )

    with col2:

        current_phase_avg = st.number_input(
            "Current Phase Average",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

        pressure_level = st.number_input(
            "Pressure Level",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.5
        )

    with col3:

        rpm = st.number_input(
            "RPM",
            min_value=0.0,
            max_value=10000.0,
            value=1500.0,
            step=50.0
        )

        hours_since_maintenance = st.number_input(
            "Hours Since Maintenance",
            min_value=0.0,
            max_value=5000.0,
            value=100.0,
            step=1.0
        )


    # =====================================================
    # OPERATING CONDITIONS
    # =====================================================

    st.markdown("### Operating Conditions")

    col1, col2 = st.columns(2)

    with col1:

        ambient_temp = st.number_input(
            "Ambient Temperature",
            min_value=-20.0,
            max_value=70.0,
            value=25.0,
            step=0.5
        )

    with col2:

        st.info(
            "Prediction target: "
            "Machine failure within the next 24 hours."
        )


    st.markdown("")


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    predict = st.button(
        "RUN PREDICTION"
    )


    # =====================================================
    # RUN PREDICTION
    # =====================================================

    if predict:

        try:

            # ---------------------------------------------
            # INPUT DATA
            # ---------------------------------------------

            input_df = pd.DataFrame(
                {
                    "machine_id": [
                        int(machine_id.replace("M", ""))
                    ],
                    "vibration_rms": [
                        vibration_rms
                    ],
                    "temperature_motor": [
                        temperature_motor
                    ],
                    "current_phase_avg": [
                        current_phase_avg
                    ],
                    "pressure_level": [
                        pressure_level
                    ],
                    "rpm": [
                        rpm
                    ],
                    "hours_since_maintenance": [
                        hours_since_maintenance
                    ],
                    "ambient_temp": [
                        ambient_temp
                    ],
                    "machine_type": [
                        machine_type
                    ],
                    "operating_mode": [
                        operating_mode
                    ]
                }
            )


            # ---------------------------------------------
            # MACHINE TYPE ENCODING
            # ---------------------------------------------

            machine_encoded = (
                machine_encoder
                .transform(
                    input_df[["machine_type"]]
                )
                .toarray()
            )

            machine_encoded_df = pd.DataFrame(
                machine_encoded,
                columns=machine_encoder.categories_[0]
            )


            # ---------------------------------------------
            # OPERATING MODE ENCODING
            # ---------------------------------------------

            operating_encoded = (
                operating_encoder
                .transform(
                    input_df[["operating_mode"]]
                )
                .toarray()
            )

            operating_encoded_df = pd.DataFrame(
                operating_encoded,
                columns=operating_encoder.categories_[0]
            )


            # ---------------------------------------------
            # COMBINE FEATURES
            # ---------------------------------------------

            model_input = pd.concat(
                [
                    input_df.drop(
                        [
                            "machine_type",
                            "operating_mode"
                        ],
                        axis=1
                    ),
                    machine_encoded_df,
                    operating_encoded_df
                ],
                axis=1
            )


            # ---------------------------------------------
            # EXACT FEATURE ORDER
            # ---------------------------------------------

            model_input = model_input.reindex(
                columns=feature_names,
                fill_value=0
            )


            # ---------------------------------------------
            # SCALING
            # ---------------------------------------------

            model_input_scaled = scaler.transform(
                model_input
            )


            # ---------------------------------------------
            # PREDICT PROBABILITY
            # ---------------------------------------------

            probability = (
                model
                .predict_proba(
                    model_input_scaled
                )[0, 1]
            )


            # ---------------------------------------------
            # APPLY THRESHOLD
            # ---------------------------------------------

            prediction = int(
                probability >= THRESHOLD
            )


            # ---------------------------------------------
            # RISK LEVEL
            # ---------------------------------------------

            if probability >= 0.70:

                risk_level = "HIGH"

            elif probability >= 0.40:

                risk_level = "MEDIUM"

            else:

                risk_level = "LOW"


            # =================================================
            # PREDICTION DASHBOARD
            # =================================================

            st.markdown(
                "## Prediction Dashboard"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "MACHINE",
                    machine_id
                )

            with col2:

                st.metric(
                    "TYPE",
                    machine_type
                )

            with col3:

                st.metric(
                    "FAILURE PROBABILITY",
                    f"{probability:.1%}"
                )

            with col4:

                st.metric(
                    "RISK LEVEL",
                    risk_level
                )


            # =================================================
            # MAIN RESULT
            # =================================================

            st.markdown(
                "### Prediction Result"
            )


            if prediction == 1:

                st.error(
                    f"⚠️ A FAILURE IS PREDICTED\n\n"
                    f"Failure probability within the next "
                    f"24 hours: {probability:.1%}"
                )

            else:

                st.success(
                    f"✓ NO FAILURE PREDICTED\n\n"
                    f"Failure probability within the next "
                    f"24 hours: {probability:.1%}"
                )


            # =================================================
            # MACHINE SUMMARY
            # =================================================

            st.markdown(
                "### Machine Summary"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "**Machine:**",
                    machine_id
                )

                st.write(
                    "**Machine Type:**",
                    machine_type
                )

                st.write(
                    "**Operating Mode:**",
                    operating_mode
                )

            with col2:

                st.write(
                    "**Vibration RMS:**",
                    f"{vibration_rms:.2f}"
                )

                st.write(
                    "**Motor Temperature:**",
                    f"{temperature_motor:.1f}"
                )

                st.write(
                    "**Hours Since Maintenance:**",
                    f"{hours_since_maintenance:.1f}"
                )


            # =================================================
            # ADDITIONAL PREDICTIONS
            # =================================================

            st.markdown(
                "### Additional Predictions"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.info(
                    "**Failure Type**\n\n"
                    "Not predicted by the current model."
                )

            with col2:

                st.info(
                    "**RUL Hours**\n\n"
                    "Not predicted by the current model."
                )

            with col3:

                st.info(
                    "**Repair Cost**\n\n"
                    "Not predicted by the current model."
                )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )


# =========================================================
# PAGE 2
# DATA DASHBOARD
# =========================================================

elif page == "Data Dashboard":

    st.markdown(
        '<div class="main-title">'
        'Data Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Explore the industrial machine predictive '
        'maintenance dataset'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # DATASET
    # =====================================================

    DATA_FILE = (
        "industrial_machine_predictive_maintenance.csv"
    )

    df_dashboard = None


    if os.path.exists(DATA_FILE):

        try:

            df_dashboard = pd.read_csv(
                DATA_FILE
            )

        except Exception as e:

            st.error(
                "Could not read the dataset."
            )

            st.code(
                str(e)
            )


    if df_dashboard is None:

        uploaded_file = st.file_uploader(
            "Upload your predictive maintenance CSV",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                df_dashboard = pd.read_csv(
                    uploaded_file
                )

            except Exception as e:

                st.error(
                    "Could not read the uploaded file."
                )

                st.code(
                    str(e)
                )


    # =====================================================
    # DASHBOARD
    # =====================================================

    if df_dashboard is not None:

        df_dashboard.columns = (
            df_dashboard.columns
            .str.strip()
        )


        # =================================================
        # KPI
        # =================================================

        total_records = len(
            df_dashboard
        )


        if "machine_id" in df_dashboard.columns:

            total_machines = (
                df_dashboard[
                    "machine_id"
                ]
                .nunique()
            )

        else:

            total_machines = 0


        if "failure_within_24h" in df_dashboard.columns:

            total_failures = int(
                df_dashboard[
                    "failure_within_24h"
                ].sum()
            )

            failure_rate = (
                total_failures / total_records
                if total_records > 0
                else 0
            )

        else:

            total_failures = 0
            failure_rate = 0


        st.markdown(
            "## Dataset Overview"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "TOTAL RECORDS",
                f"{total_records:,}"
            )


        with col2:

            st.metric(
                "MACHINES",
                total_machines
            )


        with col3:

            st.metric(
                "FAILURES",
                f"{total_failures:,}"
            )


        with col4:

            st.metric(
                "FAILURE RATE",
                f"{failure_rate:.1%}"
            )


        # =================================================
        # FILTERS
        # =================================================

        st.markdown(
            "## Filters"
        )


        filtered_df = (
            df_dashboard.copy()
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            if "machine_type" in filtered_df.columns:

                type_options = (
                    filtered_df[
                        "machine_type"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )

                selected_type = st.selectbox(
                    "Machine Type",
                    ["All"] + sorted(
                        type_options
                    )
                )

                if selected_type != "All":

                    filtered_df = filtered_df[
                        filtered_df[
                            "machine_type"
                        ] == selected_type
                    ]


        with col2:

            if "operating_mode" in filtered_df.columns:

                mode_options = (
                    filtered_df[
                        "operating_mode"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )

                selected_mode = st.selectbox(
                    "Operating Mode",
                    ["All"] + sorted(
                        mode_options
                    )
                )

                if selected_mode != "All":

                    filtered_df = filtered_df[
                        filtered_df[
                            "operating_mode"
                        ] == selected_mode
                    ]


        with col3:

            if "machine_id" in filtered_df.columns:

                machine_options = (
                    filtered_df[
                        "machine_id"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )

                selected_machine = st.selectbox(
                    "Machine",
                    ["All"] + sorted(
                        machine_options
                    )
                )

                if selected_machine != "All":

                    filtered_df = filtered_df[
                        filtered_df[
                            "machine_id"
                        ] == selected_machine
                    ]


        # =================================================
        # FAILURE DISTRIBUTION
        # =================================================

        if "failure_within_24h" in filtered_df.columns:

            st.markdown(
                "## Failure Distribution"
            )


            failure_counts = (
                filtered_df[
                    "failure_within_24h"
                ]
                .value_counts()
                .sort_index()
            )


            failure_chart = pd.DataFrame(
                {
                    "Status": [
                        "No Failure",
                        "Failure"
                    ],

                    "Records": [
                        failure_counts.get(
                            0,
                            0
                        ),

                        failure_counts.get(
                            1,
                            0
                        )
                    ]
                }
            )


            st.bar_chart(
                failure_chart.set_index(
                    "Status"
                )
            )


        # =================================================
        # FAILURE RATE BY MACHINE TYPE
        # =================================================

        if (
            "machine_type" in filtered_df.columns
            and
            "failure_within_24h"
            in filtered_df.columns
        ):

            st.markdown(
                "## Failure Rate by Machine Type"
            )


            type_failure = (
                filtered_df
                .groupby(
                    "machine_type"
                )[
                    "failure_within_24h"
                ]
                .mean()
                .sort_values(
                    ascending=False
                )
            )


            st.bar_chart(
                type_failure
            )


        # =================================================
        # SENSOR OVERVIEW
        # =================================================

        sensor_columns = [
            "vibration_rms",
            "temperature_motor",
            "current_phase_avg",
            "pressure_level",
            "rpm"
        ]


        available_sensors = [
            col
            for col in sensor_columns
            if col in filtered_df.columns
        ]


        if len(available_sensors) > 0:

            st.markdown(
                "## Sensor Overview"
            )


            sensor_chart = (
                filtered_df[
                    available_sensors
                ]
                .head(500)
            )


            st.line_chart(
                sensor_chart
            )


        # =================================================
        # FAILURE RATE BY MACHINE
        # =================================================

        if (
            "machine_id" in filtered_df.columns
            and
            "failure_within_24h"
            in filtered_df.columns
        ):

            st.markdown(
                "## Failure Rate by Machine"
            )


            machine_failure = (
                filtered_df
                .groupby(
                    "machine_id"
                )[
                    "failure_within_24h"
                ]
                .mean()
                .sort_values(
                    ascending=False
                )
            )


            st.bar_chart(
                machine_failure
            )


        # =================================================
        # DATA PREVIEW
        # =================================================

        st.markdown(
            "## Dataset Preview"
        )


        st.dataframe(
            filtered_df.head(100),
            use_container_width=True
        )


        # =================================================
        # DATA INFORMATION
        # =================================================

        st.markdown(
            "## Dataset Information"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.write(
                "**Rows:**",
                filtered_df.shape[0]
            )

            st.write(
                "**Columns:**",
                filtered_df.shape[1]
            )


        with col2:

            missing_values = int(
                filtered_df
                .isna()
                .sum()
                .sum()
            )

            st.write(
                "**Missing Values:**",
                missing_values
            )


    else:

        st.info(
            "Upload the dataset to view the "
            "Data Dashboard."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Predictive Maintenance AI System
    </div>
    """,
    unsafe_allow_html=True
)