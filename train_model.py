import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
df = pd.read_csv("data.csv")
X = df[['distance', 'speed', 'traffic']]
y = df['co2_emission']
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, "co2_model.pkl")
print("✅ Model trained and saved as co2_model.pkl")
