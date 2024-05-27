# %%
# SHAP for Feature Importance
model = RandomForestRegressor().fit(X, y)
X_numeric = X.astype(float)
explainer = shap.Explainer(model, X_numeric)
shap_values = explainer(X_numeric)
shap.summary_plot(shap_values, X_numeric)

# %%
# Correlation Analysis
df = runs_df[
    changed_configs_column_names + ["test_cos_sim_diff.correlation_coefficient"]
]
df = pd.get_dummies(df)
correlation = df.corr()["test_cos_sim_diff.correlation_coefficient"].sort_values(
    ascending=False
)
print("Correlation with mark_translation_acc:\n", correlation)

# %%
# Feature Importance using RandomForestRegressor
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(
    ascending=False
)
print("Feature importance:\n", feature_importance)
