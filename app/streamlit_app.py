import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# ======================================================
# PATH SETUP
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    layout="wide"
)


# ======================================================
# FEATURE LABELS AND EXPLANATIONS
# ======================================================

FEATURE_LABELS = {
    "Marital status": "Marital Status",
    "Application mode": "Application Mode",
    "Application order": "Application Order",
    "Course": "Study Program / Course",
    "Daytime/evening attendance": "Class Schedule",
    "Previous qualification": "Previous Qualification",
    "Previous qualification (grade)": "Previous Qualification Grade",
    "Nacionality": "Nationality",
    "Mother's qualification": "Mother's Education Level",
    "Father's qualification": "Father's Education Level",
    "Mother's occupation": "Mother's Occupation",
    "Father's occupation": "Father's Occupation",
    "Admission grade": "Admission Grade",
    "Displaced": "Displaced Student",
    "Educational special needs": "Educational Special Needs",
    "Debtor": "Debtor Status",
    "Tuition fees up to date": "Tuition Fees Up to Date",
    "Gender": "Gender",
    "Scholarship holder": "Scholarship Holder",
    "Age at enrollment": "Age at Enrollment",
    "International": "International Student",
    "Curricular units 1st sem (credited)": "1st Semester Credited Courses",
    "Curricular units 1st sem (enrolled)": "1st Semester Enrolled Courses",
    "Curricular units 1st sem (evaluations)": "1st Semester Evaluations",
    "Curricular units 1st sem (approved)": "1st Semester Approved Courses",
    "Curricular units 1st sem (grade)": "1st Semester Average Grade",
    "Curricular units 1st sem (without evaluations)": "1st Semester Courses Without Evaluation",
    "Curricular units 2nd sem (credited)": "2nd Semester Credited Courses",
    "Curricular units 2nd sem (enrolled)": "2nd Semester Enrolled Courses",
    "Curricular units 2nd sem (evaluations)": "2nd Semester Evaluations",
    "Curricular units 2nd sem (approved)": "2nd Semester Approved Courses",
    "Curricular units 2nd sem (grade)": "2nd Semester Average Grade",
    "Curricular units 2nd sem (without evaluations)": "2nd Semester Courses Without Evaluation",
    "Unemployment rate": "Unemployment Rate",
    "Inflation rate": "Inflation Rate",
    "GDP": "GDP"
}


FEATURE_HELP = {
    "Marital status": "Student's marital status at enrollment.",
    "Application mode": "The type or path used by the student to apply to the institution.",
    "Application order": "Application priority order. 0 means first choice, while higher values mean lower priority.",
    "Course": "The study program selected by the student.",
    "Daytime/evening attendance": "Whether the student attends daytime or evening classes.",
    "Previous qualification": "The student's education level before entering higher education.",
    "Previous qualification (grade)": "Grade from the student's previous qualification. The original scale is between 0 and 200.",
    "Nacionality": "Student's nationality. The dataset uses the original spelling: Nacionality.",
    "Mother's qualification": "Mother's highest education level.",
    "Father's qualification": "Father's highest education level.",
    "Mother's occupation": "Mother's occupation category.",
    "Father's occupation": "Father's occupation category.",
    "Admission grade": "Student's admission grade. The original scale is between 0 and 200.",
    "Displaced": "Whether the student moved from their usual residence to study.",
    "Educational special needs": "Whether the student has special educational needs.",
    "Debtor": "Whether the student has debt status.",
    "Tuition fees up to date": "Whether the student's tuition payment is up to date.",
    "Gender": "Student's gender in the dataset encoding.",
    "Scholarship holder": "Whether the student receives a scholarship.",
    "Age at enrollment": "Student's age when enrolling.",
    "International": "Whether the student is an international student.",
    "Curricular units 1st sem (credited)": "Number of curricular units credited in the first semester.",
    "Curricular units 1st sem (enrolled)": "Number of curricular units enrolled in the first semester.",
    "Curricular units 1st sem (evaluations)": "Number of evaluations in the first semester.",
    "Curricular units 1st sem (approved)": "Number of curricular units passed in the first semester.",
    "Curricular units 1st sem (grade)": "Average grade in the first semester. The original scale is between 0 and 20.",
    "Curricular units 1st sem (without evaluations)": "Number of first-semester curricular units without evaluation.",
    "Curricular units 2nd sem (credited)": "Number of curricular units credited in the second semester.",
    "Curricular units 2nd sem (enrolled)": "Number of curricular units enrolled in the second semester.",
    "Curricular units 2nd sem (evaluations)": "Number of evaluations in the second semester.",
    "Curricular units 2nd sem (approved)": "Number of curricular units passed in the second semester.",
    "Curricular units 2nd sem (grade)": "Average grade in the second semester. The original scale is between 0 and 20.",
    "Curricular units 2nd sem (without evaluations)": "Number of second-semester curricular units without evaluation.",
    "Unemployment rate": "Regional unemployment rate.",
    "Inflation rate": "Regional inflation rate.",
    "GDP": "Gross Domestic Product indicator from the dataset."
}


# ======================================================
# VALUE MAPPINGS FOR CATEGORICAL FEATURES
# ======================================================

YES_NO = {
    0: "No",
    1: "Yes"
}

GENDER = {
    0: "Female",
    1: "Male"
}

MARITAL_STATUS = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated"
}

DAYTIME_EVENING = {
    0: "Evening",
    1: "Daytime"
}

APPLICATION_MODE = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 - Different Plan",
    27: "Ordinance No. 533-A/99, item b3 - Other Institution",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course - International"
}

COURSE = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service - evening attendance",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management - evening attendance"
}

PREVIOUS_QUALIFICATION = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle or equivalent",
    38: "Basic education 2nd cycle or equivalent",
    39: "Technological specialization course",
    40: "Higher education - degree, 1st cycle",
    42: "Professional higher technical course",
    43: "Higher education - master, 2nd cycle"
}

NATIONALITY = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian"
}

PARENT_QUALIFICATION = {
    1: "Secondary Education - 12th Year or equivalent",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year - Old",
    12: "Other - 11th Year of Schooling",
    13: "2nd year complementary high school course",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle or equivalent",
    20: "Complementary High School Course",
    22: "Technical-professional course",
    25: "Complementary High School Course - not concluded",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th year of schooling",
    31: "General Course of Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without having a 4th year of schooling",
    37: "Basic education 1st cycle or equivalent",
    38: "Basic Education 2nd Cycle or equivalent",
    39: "Technological specialization course",
    40: "Higher education - degree, 1st cycle",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master, 2nd cycle",
    44: "Higher Education - Doctorate, 3rd cycle"
}

OCCUPATION = {
    0: "Student",
    1: "Representatives, directors and executive managers",
    2: "Specialists in intellectual and scientific activities",
    3: "Intermediate level technicians and professions",
    4: "Administrative staff",
    5: "Personal services, security workers and sellers",
    6: "Farmers and skilled workers in agriculture, fisheries and forestry",
    7: "Skilled workers in industry, construction and craftsmen",
    8: "Installation and machine operators",
    9: "Unskilled workers",
    10: "Armed Forces professions",
    90: "Other situation",
    99: "Blank / not specified",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces personnel",
    112: "Directors of administrative and commercial services",
    114: "Hotel, catering, trade and other services directors",
    121: "Specialists in physical sciences, mathematics and engineering",
    122: "Health professionals",
    123: "Teachers",
    124: "Finance, accounting and administrative specialists",
    125: "ICT specialists",
    131: "Intermediate science and engineering technicians",
    132: "Intermediate health technicians",
    134: "Legal, social, sports and cultural technicians",
    135: "ICT technicians",
    141: "Office workers and secretaries",
    143: "Data, accounting and registry operators",
    144: "Other administrative support staff",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers",
    154: "Protection and security services personnel",
    161: "Market-oriented farmers and skilled agricultural workers",
    163: "Subsistence farmers, fishermen, hunters and gatherers",
    171: "Skilled construction workers",
    172: "Skilled workers in metallurgy and metalworking",
    173: "Skilled workers in printing and precision manufacturing",
    174: "Skilled workers in electricity and electronics",
    175: "Food processing, woodworking, clothing and craft workers",
    181: "Fixed plant and machine operators",
    182: "Assembly workers",
    183: "Vehicle drivers and mobile equipment operators",
    191: "Cleaning workers",
    192: "Unskilled agricultural, animal production and fisheries workers",
    193: "Unskilled workers in extractive industry, construction and transport",
    194: "Meal preparation assistants",
    195: "Street vendors and street service providers"
}

VALUE_LABELS = {
    "Marital status": MARITAL_STATUS,
    "Application mode": APPLICATION_MODE,
    "Course": COURSE,
    "Daytime/evening attendance": DAYTIME_EVENING,
    "Previous qualification": PREVIOUS_QUALIFICATION,
    "Nacionality": NATIONALITY,
    "Mother's qualification": PARENT_QUALIFICATION,
    "Father's qualification": PARENT_QUALIFICATION,
    "Mother's occupation": OCCUPATION,
    "Father's occupation": OCCUPATION,
    "Displaced": YES_NO,
    "Educational special needs": YES_NO,
    "Debtor": YES_NO,
    "Tuition fees up to date": YES_NO,
    "Gender": GENDER,
    "Scholarship holder": YES_NO,
    "International": YES_NO
}


FEATURE_GROUPS = {
    "Personal and Enrollment Information": [
        "Marital status",
        "Age at enrollment",
        "Gender",
        "Nacionality",
        "International",
        "Displaced",
        "Educational special needs"
    ],
    "Application and Course Information": [
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Admission grade",
        "Previous qualification",
        "Previous qualification (grade)"
    ],
    "Family Background": [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation"
    ],
    "Financial and Support Information": [
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder"
    ],
    "1st Semester Academic Performance": [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)"
    ],
    "2nd Semester Academic Performance": [
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)"
    ],
    "Economic Indicators": [
        "Unemployment rate",
        "Inflation rate",
        "GDP"
    ]
}


# ======================================================
# LOAD ARTIFACTS
# ======================================================

@st.cache_resource
def load_artifacts():
    model_path_keras = MODEL_DIR / "dropout_prediction_model.keras"
    model_path_h5 = MODEL_DIR / "dropout_prediction_model.h5"

    if model_path_keras.exists():
        model = load_model(model_path_keras)
    elif model_path_h5.exists():
        model = load_model(model_path_h5)
    else:
        st.error("Model file not found. Please check the models folder.")
        st.stop()

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    feature_info = joblib.load(MODEL_DIR / "feature_info.pkl")

    return model, scaler, feature_names, feature_info


model, scaler, feature_names, feature_info = load_artifacts()


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def clean_numeric_value(value):
    try:
        if float(value).is_integer():
            return int(value)
        return float(value)
    except Exception:
        return value


def format_option(feature, value):
    value_clean = clean_numeric_value(value)
    mapping = VALUE_LABELS.get(feature)

    if mapping is None:
        return str(value_clean)

    label = mapping.get(value_clean, "Unknown option")
    return f"{value_clean} - {label}"


def get_default_option(options, median_value):
    median_value = clean_numeric_value(median_value)

    if median_value in options:
        return options.index(median_value)

    return 0


def render_input(feature):
    info = feature_info[feature]

    min_value = float(info["min"])
    max_value = float(info["max"])
    median_value = float(info["median"])

    label = FEATURE_LABELS.get(feature, feature)
    help_text = FEATURE_HELP.get(feature, "")

    # Categorical features with readable labels
    if feature in VALUE_LABELS:
        mapping = VALUE_LABELS[feature]

        unique_values = info.get("unique_values")

        if unique_values is not None:
            options = sorted([clean_numeric_value(v) for v in unique_values])
        else:
            options = sorted(mapping.keys())

        selected = st.selectbox(
            label=label,
            options=options,
            index=get_default_option(options, median_value),
            format_func=lambda x: format_option(feature, x),
            help=help_text
        )

        return selected

    # Numeric features
    if max_value <= 20 and min_value >= 0:
        step = 0.1
    elif max_value <= 200 and min_value >= 0:
        step = 1.0
    else:
        step = 0.1

    return st.number_input(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=median_value,
        step=step,
        help=help_text
    )


def get_risk_level(probability):
    if probability < 0.40:
        return "Low Risk"
    elif probability < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"


def get_recommendation(probability):
    if probability < 0.40:
        return (
            "Routine monitoring is recommended. The student is currently considered "
            "low risk based on the model prediction."
        )
    elif probability < 0.70:
        return (
            "Academic advisor consultation is recommended. The student should be "
            "monitored more closely, especially in academic performance and payment status."
        )
    else:
        return (
            "Early intervention is strongly recommended. The institution should consider "
            "academic counseling, financial support review, and continuous monitoring."
        )


def predict_dropout(input_df):
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    probability = model.predict(input_scaled, verbose=0).ravel()
    prediction = (probability >= 0.5).astype(int)

    return probability, prediction


def explain_prediction(probability):
    risk_level = get_risk_level(probability)

    if risk_level == "Low Risk":
        return "The model estimates that this student has a relatively low probability of dropping out."
    elif risk_level == "Medium Risk":
        return "The model estimates a moderate dropout risk. This student may need closer monitoring."
    else:
        return "The model estimates a high dropout risk. Early support is recommended."


# ======================================================
# MAIN PAGE
# ======================================================

st.title("Student Dropout Risk Prediction")

st.write(
    "This dashboard predicts whether a student is at risk of dropout using a TensorFlow Neural Network model. "
    "The input form has been designed with readable labels so users do not need to understand the raw numeric codes in the dataset."
)

with st.expander("How to use this dashboard"):
    st.write(
        """
        1. Open the **Manual Prediction** tab.
        2. Fill in the student's personal, academic, financial, and enrollment information.
        3. Each categorical option is displayed with a code and explanation, for example: `1 - Single`.
        4. Click **Predict Dropout Risk**.
        5. The dashboard will show dropout probability, predicted class, risk level, and recommended action.
        """
    )

with st.expander("Important note"):
    st.write(
        """
        This dashboard is intended as a portfolio demonstration and decision-support prototype.
        The prediction should not be used as the only basis for academic decisions. Human review is still required.
        """
    )

st.divider()


# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Manual Prediction",
        "Batch Prediction",
        "Feature Guide"
    ]
)


# ======================================================
# TAB 1: MANUAL PREDICTION
# ======================================================

with tab1:
    st.subheader("Manual Student Data Input")

    st.write(
        "Fill in the form below. The system will keep the original numeric values for the model, "
        "but displays readable explanations for users."
    )

    input_data = {}

    with st.form("manual_prediction_form"):
        for group_name, features in FEATURE_GROUPS.items():
            valid_features = [feature for feature in features if feature in feature_names]

            if len(valid_features) == 0:
                continue

            st.markdown(f"### {group_name}")

            cols = st.columns(2)

            for i, feature in enumerate(valid_features):
                with cols[i % 2]:
                    input_data[feature] = render_input(feature)

            st.divider()

        submitted = st.form_submit_button("Predict Dropout Risk")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]

        probability, prediction = predict_dropout(input_df)

        dropout_probability = float(probability[0])
        predicted_class = int(prediction[0])
        risk_level = get_risk_level(dropout_probability)
        recommendation = get_recommendation(dropout_probability)

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Dropout Probability",
                value=f"{dropout_probability * 100:.2f}%"
            )

        with col2:
            prediction_label = "Dropout" if predicted_class == 1 else "Graduate"
            st.metric(
                label="Predicted Class",
                value=prediction_label
            )

        with col3:
            st.metric(
                label="Risk Level",
                value=risk_level
            )

        st.info(explain_prediction(dropout_probability))

        st.subheader("Recommended Action")
        st.write(recommendation)

        with st.expander("Show model input data"):
            display_df = input_df.copy()

            for feature in display_df.columns:
                if feature in VALUE_LABELS:
                    display_df[feature] = display_df[feature].apply(
                        lambda value: format_option(feature, value)
                    )

            st.dataframe(display_df, use_container_width=True)


# ======================================================
# TAB 2: BATCH PREDICTION
# ======================================================

with tab2:
    st.subheader("Batch Prediction from CSV")

    st.write(
        "Upload a CSV file containing the same feature columns used during model training. "
        "The file may use comma or semicolon as separator."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, sep=";")

            if len(batch_df.columns) == 1:
                uploaded_file.seek(0)
                batch_df = pd.read_csv(uploaded_file)

            batch_df.columns = batch_df.columns.str.strip()

            st.write("Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            missing_features = [
                feature for feature in feature_names
                if feature not in batch_df.columns
            ]

            if missing_features:
                st.error("The uploaded file is missing required feature columns.")
                st.write(missing_features)
            else:
                batch_input = batch_df[feature_names].copy()

                probability, prediction = predict_dropout(batch_input)

                result_df = batch_df.copy()
                result_df["Dropout Probability"] = probability
                result_df["Predicted Class"] = np.where(
                    prediction == 1,
                    "Dropout",
                    "Graduate"
                )
                result_df["Risk Level"] = result_df["Dropout Probability"].apply(
                    get_risk_level
                )
                result_df["Recommendation"] = result_df["Dropout Probability"].apply(
                    get_recommendation
                )

                st.subheader("Batch Prediction Result")
                st.dataframe(result_df, use_container_width=True)

                csv_result = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Prediction Result",
                    data=csv_result,
                    file_name="student_dropout_prediction_result.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error("An error occurred while processing the file.")
            st.write(e)


# ======================================================
# TAB 3: FEATURE GUIDE
# ======================================================

with tab3:
    st.subheader("Feature Guide")

    st.write(
        "This section explains the meaning of each input feature used by the model."
    )

    guide_rows = []

    for feature in feature_names:
        guide_rows.append(
            {
                "Original Feature Name": feature,
                "Display Name": FEATURE_LABELS.get(feature, feature),
                "Explanation": FEATURE_HELP.get(feature, "-"),
                "Input Type": "Categorical" if feature in VALUE_LABELS else "Numeric"
            }
        )

    guide_df = pd.DataFrame(guide_rows)

    st.dataframe(guide_df, use_container_width=True)

    st.subheader("Categorical Code Reference")

    selected_feature = st.selectbox(
        "Select a categorical feature to view code meanings",
        options=list(VALUE_LABELS.keys())
    )

    mapping_df = pd.DataFrame(
        [
            {
                "Code": code,
                "Meaning": meaning
            }
            for code, meaning in VALUE_LABELS[selected_feature].items()
        ]
    )

    st.dataframe(mapping_df, use_container_width=True)