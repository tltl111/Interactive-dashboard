import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
num_samples = 1000

data = {
    "total_rooms": np.random.randint(100, 3000, size=num_samples),
    "total_bedrooms": np.random.randint(50, 1000, size=num_samples),
    "population": np.random.randint(100, 5000, size=num_samples),
    "households": np.random.randint(80, 2000, size=num_samples),
    "median_income": np.random.uniform(1, 10, size=num_samples),
}

data["median_house_value"] = 1000 + data["total_rooms"] * 10 + data["total_bedrooms"] * 5 \
                            + data["population"] * 2 + data["households"] * 3 + data["median_income"] * 1000

housing = pd.DataFrame(data)

print(housing.head())

X = housing.drop("median_house_value", axis=1)
y = housing["median_house_value"]

# print(X)
# print(y)

# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

X_train_transformed = num_pipeline.fit_transform(X_train)

print(X_train_transformed)