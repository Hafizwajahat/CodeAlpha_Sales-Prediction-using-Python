"""CodeAlpha Task 4: Sales Prediction using Python
Dataset: Advertising.csv (columns: TV, Radio, Newspaper, Sales).
Usage: python sales_prediction.py Advertising.csv
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

path = sys.argv[1] if len(sys.argv) > 1 else "Advertising.csv"
df = pd.read_csv(path)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
df = df.drop_duplicates()
print(df.describe(), "\nMissing:\n", df.isnull().sum())

# --- EDA ---
sns.pairplot(df, kind="reg"); plt.savefig("pairplot.png", dpi=110); plt.close()
plt.figure(figsize=(5, 4)); sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.tight_layout(); plt.savefig("correlation.png", dpi=120); plt.close()
print("\nCorrelation with Sales:\n", df.corr()["Sales"].drop("Sales").sort_values(ascending=False))

# --- Modeling ---
X, y = df.drop(columns="Sales"), df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial (deg 2)": make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}
scores = {}
for name, m in models.items():
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring="r2").mean()
    m.fit(X_train, y_train); p = m.predict(X_test)
    scores[name] = r2_score(y_test, p)
    print(f"{name:20s} CV R2={cv:.3f} | Test R2={scores[name]:.3f} | "
          f"MAE={mean_absolute_error(y_test, p):.2f} | RMSE={np.sqrt(mean_squared_error(y_test, p)):.2f}")
best = max(scores, key=scores.get); print("Best model:", best)
plt.figure(figsize=(5, 5)); plt.scatter(y_test, models[best].predict(X_test))
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Sales"); plt.ylabel("Predicted Sales"); plt.title(best)
plt.savefig("actual_vs_pred.png", dpi=120); plt.close()

# --- Business insight: impact of ad spend ---
lr = models["Linear Regression"]
print("\nImpact of +$1 (thousand) ad spend on sales (units):")
for f, c in zip(X.columns, lr.coef_):
    print(f"  {f:10s}: {c:+.4f}")
base = X.mean().to_frame().T
for ch in X.columns:
    up = base.copy(); up[ch] *= 1.10
    print(f"  +10% {ch:9s} budget -> sales change {lr.predict(up)[0]-lr.predict(base)[0]:+.3f}")
