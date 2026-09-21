# %% import modules
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% load data
for path in Path(".").iterdir():
    print(path)

for path in Path("data").iterdir():
    print(path)
    
# %% load data

# load and preview data
data_bmi = pd.read_csv("data/data_bmi.tsv", sep = "\t")
data_bmi.info()
data_bmi.head()

data_bmi["BMI>30"]
data_bmi.iloc[:, :10].dtypes

# %% convert to categorical (BMI>30 and gender)
categorical_columns = ["BMI>30", "gender"]
data_bmi[categorical_columns] = data_bmi[categorical_columns].astype("category")
data_bmi.iloc[:, :10].dtypes

# %% Make example plot

sns.boxplot(
    data=data_bmi,
    x="BMI>30",
    y="AFF3",
    hue="BMI>30",
    palette="Set1",
    legend=False,
)

plt.xlabel("BMI > 30")
plt.ylabel("Gene expression")
plt.show()