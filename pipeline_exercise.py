from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Step 1: Define the pipeline steps as a list of tuples
pipeline_steps = [
    ('imputer', SimpleImputer(strategy='median')),  # Step 1: Impute missing values
    ('scaler', StandardScaler()),                 # Step 2: Standardize numerical features
    ('regressor', LinearRegression())             # Step 3: Train a linear regression model
]

# Step 2: Create the pipeline using the Pipeline class
model_pipeline = Pipeline(steps=pipeline_steps)

# Step 3: Fit the pipeline to the training data (X_train and y_train)
model_pipeline.fit(X_train, y_train)

# Step 4: Make predictions on the test data (X_test)
y_pred = model_pipeline.predict(X_test)
