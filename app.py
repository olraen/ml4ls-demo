# app.py

import joblib
import pandas as pd
import streamlit as st


# Load model
model = joblib.load("bmi_model.joblib")


# Get features from saved model
preprocess = model.named_steps["preprocess"]

numeric_features = preprocess.transformers_[0][2]
selected_genes = [x for x in numeric_features if x != "age"]

gender_levels = preprocess.named_transformers_["gender"].categories_[0]

# Means used as convenient default values
numeric_means = preprocess.named_transformers_["numeric"].mean_


# App
st.title("BMI > 30 prediction")

st.write(
    "Enter age, gender and gene-expression values to estimate "
    "the probability of BMI > 30."
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=int(round(numeric_means[0])),
    
    step=1
)


# Gender
gender = st.selectbox(
    "Gender",
    gender_levels
)


# Genes
gene_values = {}

for i, gene in enumerate(selected_genes):
    gene_values[gene] = st.number_input(
        gene,
        value=float(numeric_means[i + 1])
    )


# Prediction
if st.button("Predict"):

    new_data = pd.DataFrame(
        {
            "age": [age],
            "gender": [gender],
            **{gene: [value] for gene, value in gene_values.items()}
        }
    )

    probability = model.predict_proba(new_data)[0, 1]

    st.metric(
        "Probability of BMI > 30",
        f"{probability:.1%}"
    )

st.markdown("### About the model")

with st.expander("Model details"):
    st.write("Feature selection: Logistic Lasso")
    st.write("Final model: Logistic regression")
    st.write("Selected genes:", ", ".join(selected_genes))

st.link_button(
    "View documentation on GitHub",
    "https://github.com/olraen/ml4ls-demo"
)

st.caption(
    "Teaching demonstration only."
)