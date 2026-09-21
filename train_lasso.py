# %% import modules
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json

# %% load data
data_bmi = pd.read_csv("data/data_bmi.tsv", sep="\t")
data_bmi.iloc[:, :10].dtypes

# convert to categorical data
categorical_columns = ["BMI>30", "gender"]
data_bmi[categorical_columns] = data_bmi[categorical_columns].astype("category")
data_bmi.iloc[:, :10].dtypes

# set outcome variable
outcome = "BMI>30"
y = data_bmi[outcome]

# prepare feature matrix
genes = data_bmi.columns.drop(["id", "BMI>30", "age", "gender"]).tolist()
features = data_bmi.columns.drop(["id", "BMI>30"]).tolist()
features[0:10]
X = data_bmi[features]

# %% split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=123
)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

y_train.value_counts()
y_test.value_counts()


# %% preprocessing

preprocess = ColumnTransformer(
    [
        ("numeric", StandardScaler(), ["age"] + genes),
        ("gender", OneHotEncoder(drop="if_binary"), ["gender"])
    ]
)

# %% logistic lasso
lasso = Pipeline(
    [
        ("preprocess", preprocess),
        (
            "model",
            LogisticRegression(
                l1_ratio=1,
                solver="liblinear",
                max_iter=5000
            )
        )
    ]
)


# %% Find best penalty using cross-validation

search = GridSearchCV(
    lasso,
    {"model__C": np.logspace(-3, 2, 30)},
    scoring="roc_auc",
    cv=5
)

search.fit(X_train, y_train)

print("Best C:", search.best_params_["model__C"])
print("CV ROC AUC:", search.best_score_)


# %% Look at Lasso coefficients

feature_names = search.best_estimator_[
    "preprocess"
].get_feature_names_out()

coefficients = search.best_estimator_[
    "model"
].coef_[0]

coef_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

coef_table["abs_coefficient"] = coef_table["coefficient"].abs()

coef_table = coef_table.sort_values(
    "abs_coefficient",
    ascending=False
)

print(coef_table.head(20))


# %% Select top 5 genes

gene_results = coef_table[
    coef_table["feature"].str.startswith("numeric__")
].copy()

gene_results["gene"] = gene_results["feature"].str.replace(
    "numeric__",
    "",
    regex=False
)

gene_results = gene_results[
    ~gene_results["gene"].eq("age")
]

selected_genes = (
    gene_results
    .query("coefficient != 0")
    .head(5)["gene"]
    .tolist()
)

print("Selected genes:", selected_genes)


# %% Fit small final model

final_features = ["age", "gender"] + selected_genes

final_preprocess = ColumnTransformer(
    [
        ("numeric", StandardScaler(), ["age"] + selected_genes),
        ("gender", OneHotEncoder(drop="if_binary"), ["gender"])
    ]
)

final_model = Pipeline(
    [
        ("preprocess", final_preprocess),
        ("model", LogisticRegression(max_iter=5000))
    ]
)

final_model.fit(
    X_train[final_features],
    y_train
)


# %% Evaluate on test data

probability = final_model.predict_proba(
    X_test[final_features]
)[:, 1]

prediction = final_model.predict(
    X_test[final_features]
)

print(
    "Confusion matrix:\n",
    confusion_matrix(y_test, prediction)
)

print(
    f"Test accuracy: {accuracy_score(y_test, prediction):.2f}"
)


print(
    f"Test ROC AUC: {roc_auc_score(y_test, probability):.2f}"
)

# %% Save model and model metadata

joblib.dump(
    final_model,
    "bmi_model.joblib"
)

metadata = {
    "model": "Logistic regression",
    "outcome": "BMI>30",
    "selected_genes": selected_genes,
    "features": final_features,
    "feature_selection": "Logistic Lasso",
    "lasso_best_C": search.best_params_["model__C"],
    "lasso_cv_roc_auc": search.best_score_,
    "test_roc_auc": roc_auc_score(y_test, probability),
    "test_accuracy": accuracy_score(y_test, prediction),
    "random_state": 123
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


# %% save coefficients used by the final model

final_feature_names = (
    final_model["preprocess"]
    .get_feature_names_out()
)

final_coefficients = (
    final_model["model"]
    .coef_[0]
)

model_coefficients = pd.DataFrame({
    "feature": final_feature_names,
    "coefficient": final_coefficients
})

model_coefficients.to_csv(
    "model_coefficients.csv",
    index=False
)

print(model_coefficients)