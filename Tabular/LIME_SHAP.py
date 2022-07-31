import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import lime
import lime.lime_tabular
import shap

import urllib.request as request
import warnings
warnings.filterwarnings('ignore')


###############################################################################
#Data load & preprocessing
###############################################################################

# Define data dir.
root_dir = os.path.join(os.getcwd(), "Tabular")
data_dir = os.path.join(root_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# Download dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
save_fname = os.path.join(data_dir, 'adult.data')
request.urlretrieve(url, save_fname)

# Define dataset column names
cols = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num",
       "Marital Status","Occupation", "Relationship", "Race", "Sex",
       "Capital Gain", "Capital Loss","Hours per week", "Country", "Income"]

# Read dataset
df = pd.read_csv(save_fname, header=None, names=cols)
df.head(3)

# Encoding categorical feature & target
col_categorical, col_numerical, col_target = [], [], []

for col, type_ in df.dtypes.to_dict().items():
       if col == 'Income': col_target.append(col)
       elif str(type_) == 'int64': col_numerical.append(col)
       elif str(type_) == 'object': col_categorical.append(col)
       else: raise ValueError

scaler_LB = LabelEncoder()
df['target'] = scaler_LB.fit_transform(df[col_target]).tolist()
df_target_class = scaler_LB.classes_
print(df_target_class.ravel())

categorical_names = {}
for col in col_categorical:
    scaler_LB = LabelEncoder()
    df[col] = scaler_LB.fit_transform(df[col])
    categorical_names[col] = scaler_LB.classes_

cat_dummy = pd.get_dummies(df[col_categorical].astype(str))

col_categorical_oh = cat_dummy.columns.tolist()

df = pd.merge(df[col_numerical+['target']], cat_dummy, how='left', left_index=True, right_index=True)
features = col_numerical+col_categorical_oh

# split train/valid/testset
target = 'Income'
if "Set" not in df.columns:
    df["Set"] = np.random.choice(["train", "valid", "test"], p=[.6, .2, .2], size=(df.shape[0],))

for s in ['train', 'valid', 'test']:
       globals()[f"X_{s}"] = df.loc[df[df.Set == s].index][features]
       globals()[f"y_{s}"] = df.loc[df[df.Set == s].index]['target']


###############################################################################
# Train model
###############################################################################
model_gb = GradientBoostingClassifier(n_estimators=100, random_state=2022)
model_gb.fit(X_train, y_train)

print("Accuracy score : ", round(accuracy_score(y_test, model_gb.predict(X_test)), 2))


###############################################################################
# LIME & SHAP
###############################################################################

# LIME
predict_fn = lambda x: model_gb.predict_proba(x)
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values,
                                                   feature_names=X_train.columns,
                                                   categorical_features=col_categorical_oh,
                                                   class_names=df_target_class,
                                                   kernel_width=3)

# LIME result
idx = 4
exp = explainer.explain_instance(X_test.values[idx],
                                 predict_fn,
                                 num_features=5,
                                 num_samples=10000)
exp.show_in_notebook(show_all=False)
# exp.as_pyplot_figure()
# plt.show()

# SHAP
shap.initjs()
explainer = shap.TreeExplainer(model=model_gb, data=X_train)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[idx])

shap.plots.force(explainer.expected_value, shap_values.values[idx, :], features=X_test.iloc[idx, :])
shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values)
shap.plots.bar(shap_values)