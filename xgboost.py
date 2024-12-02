import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("football_manager_data.csv")

#encoder = OneHotEncoder(sparse=False)
#media_desc_encoded = encoder.fit_transform(data[['Media Description']])
#media_desc_df = pd.DataFrame(media_desc_encoded, columns=encoder.get_feature_names_out(['Media Description']))
#data = pd.concat([data, media_desc_df], axis=1)
data = data.drop(columns=['Media Description'])

# Drop non-numeric columns (e.g., Position and Player_Name)
data = data.drop(columns=['Inf', 'Name', 'CA'])
data = data.drop(data.tail(2).index)

# Separate features (X) and target (y)
X = data.drop(columns=['PA'])
y = data['PA']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
xgb.fit(X_train, y_train)

# Predict on test data
y_pred = xgb.predict(X_test)


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")


feature_importance = xgb.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 8))
plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in XGBoost Model")
plt.show()