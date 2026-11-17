import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df = pd.read_csv("/dataset.csv")

print("Dataset Loaded Successfully!")

print(df.head())

print("\nDataset Shape:")
print(df.shape)


# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Target column
target = "Aggregate rating"

# Remove rows where target is missing
df = df.dropna(subset=[target])

features = [
    "Country Code",
    "City",
    "Longitude",
    "Latitude",
    "Cuisines",
    "Average Cost for two",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu",
    "Price range",
    "Votes"
]

X = df[features]

# Target variable
y = df[target]
numerical_features = [
    "Country Code",
    "Longitude",
    "Latitude",
    "Average Cost for two",
    "Price range",
    "Votes"
]

categorical_features = [
    "City",
    "Cuisines",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu"
]


# Numerical data preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical data preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])


print("\nTraining Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2)



results = pd.DataFrame({
    "Actual Rating": y_test.values,
    "Predicted Rating": y_pred
})

print("\nPrediction Results:")
print(results.head(10))


new_restaurant = pd.DataFrame([{
    "Country Code": 1,
    "City": "Bangalore",
    "Longitude": 77.5946,
    "Latitude": 12.9716,
    "Cuisines": "North Indian",
    "Average Cost for two": 800,
    "Currency": "Indian Rupees(Rs.)",
    "Has Table booking": "Yes",
    "Has Online delivery": "Yes",
    "Is delivering now": "No",
    "Switch to order menu": "No",
    "Price range": 2,
    "Votes": 150
}])

prediction = model.predict(new_restaurant)

print("\nPredicted Restaurant Rating:", prediction[0])
