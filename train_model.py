import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
df = pd.read_csv("flight.csv")

print("Dataset shape:", df.shape)

# ----------------------------
# 2️⃣ Basic Cleaning
# ----------------------------
df.dropna(inplace=True)

# Convert Date column if exists
if "Date_of_Journey" in df.columns:
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], errors="coerce")
    df["journey_day"] = df["Date_of_Journey"].dt.day
    df["journey_month"] = df["Date_of_Journey"].dt.month
    df.drop("Date_of_Journey", axis=1, inplace=True)

# ----------------------------
# 3️⃣ Define Target & Features
# ----------------------------
if "Price" not in df.columns:
    raise ValueError("❌ 'Price' column not found in dataset")

y = df["Price"]
X = df.drop("Price", axis=1)

# Identify column types
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# ----------------------------
# 4️⃣ Preprocessing
# ----------------------------
numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# ----------------------------
# 5️⃣ Model Pipeline
# ----------------------------
model = RandomForestRegressor(
    n_estimators=30,      # reduce trees
    max_depth=10,         # limit tree size
    min_samples_split=5,  # reduce complexity
    random_state=42,
    n_jobs=-1
         # Use all CPU cores
)

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", model)
])

# ----------------------------
# 6️⃣ Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 7️⃣ Train Model
# ----------------------------
print("Training model...")
pipeline.fit(X_train, y_train)

# ----------------------------
# 8️⃣ Evaluate
# ----------------------------
y_pred = pipeline.predict(X_test)
score = r2_score(y_test, y_pred)

print(f"Model R2 Score: {score:.4f}")

# ----------------------------
# 9️⃣ Save Model
# ----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as model.pkl")


