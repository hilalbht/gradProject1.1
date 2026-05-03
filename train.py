import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

# DATA
df = pd.read_csv("liver_cirrhosis.csv")
df = df.drop_duplicates()
df = df.drop(columns=["Status", "N_Days"])
df["Age"] = df["Age"] / 365.25

X = df.drop("Stage", axis=1)
y = df["Stage"] - 1

FEATURES = X.columns.tolist()
X = X[FEATURES]

log_cols = ["Alk_Phos","SGOT","Tryglicerides","Bilirubin","Copper"]
cat_cols = ["Drug","Sex","Ascites","Hepatomegaly","Spiders","Edema"]
num_cols = [c for c in FEATURES if c not in log_cols + cat_cols]

def preprocessor():
    return ColumnTransformer([
        ("log", Pipeline([
            ("log", FunctionTransformer(np.log1p)),
            ("imp", SimpleImputer(strategy="median"))
        ]), log_cols),

        ("num", SimpleImputer(strategy="median"), num_cols),

        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols)
    ])

xgb = Pipeline([("pre", preprocessor()), ("model", XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42
))])

rf = Pipeline([("pre", preprocessor()), ("model", RandomForestClassifier(
    n_estimators=250, random_state=42
))])

lgbm = Pipeline([("pre", preprocessor()), ("model", lgb.LGBMClassifier(
    n_estimators=250, random_state=42
))])

stacking = StackingClassifier(
    estimators=[("xgb", xgb), ("rf", rf), ("lgbm", lgbm)],
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1
)

stacking.fit(X, y)

# 🔥 SAFE PACKAGE (EN ÖNEMLİ KISIM)
package = {
    "model": stacking,
    "features": FEATURES
}

joblib.dump(package, "model.pkl", compress=3)

print("MODEL SAVED (PRODUCTION READY)")