import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

#Load + Clean
df = pd.read_csv("data.csv", dtype={"CRS_DEP_TIME": str})
df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()
df["DEP_HOUR"] = df["CRS_DEP_TIME"].str[:2].astype(int)

#Features/Target
TARGET = "DEP_DELAY"
numeric_features = ["DISTANCE", "CRS_ELAPSED_TIME", "DEP_HOUR"]
categorical_features = ["OP_UNIQUE_CARRIER", "DAY_OF_WEEK", "MONTH"]
 
model_cols = numeric_features + categorical_features + [TARGET]
data = df[model_cols].dropna()
 
X = pd.get_dummies(data[numeric_features + categorical_features],
                    columns=categorical_features, drop_first=True)
y = data[TARGET]

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
 
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

#Cross validation on training-set
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
 
candidates = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

cv_scores = {}
print("5-fold CV RMSE on training data (model selection, no test-set peeking):")
for name, model in candidates.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
    rmse = -scores.mean()
    cv_scores[name] = rmse
    print(f"  {name}: {rmse:.3f} (+/- {scores.std():.3f})")
 
best_name = min(cv_scores, key=cv_scores.get)
print(f"\nSelected model (lowest CV RMSE): {best_name}")
 
#Hyperparameter tuning for model using training-set
if best_name == "RandomForestRegressor":
    param_grid = {"n_estimators": [100, 200], "max_depth": [8, 12, None]}
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
elif best_name == "Ridge":
    param_grid = {"alpha": [0.1, 1.0, 10.0, 50.0]}
    grid = GridSearchCV(
        Ridge(random_state=RANDOM_STATE), param_grid, cv=cv,
        scoring="neg_root_mean_squared_error",
    )
else:
    grid = None
 
if grid is not None:
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV RMSE after tuning: {-grid.best_score_:.3f}")
    best_model = grid.best_estimator_
else:
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)
    
#Final eval with test set
final_preds = best_model.predict(X_test)
mae = mean_absolute_error(y_test, final_preds)
mse = mean_squared_error(y_test, final_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, final_preds)

print(f"\nFinal held-out test performance ({best_name}):")
print(f"  MAE:  {mae:.3f}")
print(f"  MSE:  {mse:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  R^2:  {r2:.3f}")

#Feature importances
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False).head(15)
elif hasattr(best_model, "coef_"):
    importances = pd.Series(best_model.coef_, index=X.columns)
    importances = importances.reindex(
        importances.abs().sort_values(ascending=False).index
    ).head(15)
else:
    importances = None
 
if importances is not None:
    plt.figure(figsize=(8, 6))
    importances.sort_values().plot(kind="barh")
    plt.title(f"{best_name} - Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("regression_feature_importance.png")
    print("\nSaved regression_feature_importance.png")
    
#Predicted vs. actual plotted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, final_preds, alpha=0.2, s=8)
lims = [min(y_test.min(), final_preds.min()), max(y_test.max(), final_preds.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Delay (min)")
plt.ylabel("Predicted Delay (min)")
plt.title(f"{best_name}: Predicted vs Actual Delay")
plt.tight_layout()
plt.savefig("regression_pred_vs_actual.png")
print("Saved regression_pred_vs_actual.png")

