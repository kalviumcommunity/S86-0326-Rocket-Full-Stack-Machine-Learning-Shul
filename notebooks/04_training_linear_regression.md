# Training a Linear Regression Model

After a baseline model, the next step is to train a model that actually learns from features. For regression problems, Linear Regression is often the first useful supervised learning algorithm to try.

It learns a linear relationship between input features and a continuous target variable. The result is simple, fast, and interpretable when the problem fits the assumptions.

## 1. What Linear Regression Does

Linear Regression predicts a continuous value from one or more features.

In its simplest form:

```text
y_hat = b + w1*x1 + w2*x2 + ... + wn*xn
```

Where:

- `y_hat` is the predicted value
- `x1 ... xn` are the input features
- `w1 ... wn` are learned coefficients
- `b` is the intercept

The model learns the coefficients by minimizing prediction error on the training set.

## 2. Training Objective

Linear Regression usually minimizes Mean Squared Error (MSE):

```text
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

Squaring the errors ensures positive values and penalizes large mistakes more heavily than small ones.

## 3. How Training Works

There are two common ways to fit a linear regression model:

- Closed-form solution: solves for the best weights directly
- Gradient descent: iteratively updates weights to reduce loss

In scikit-learn, `LinearRegression` uses an efficient least-squares solver, which is fast and numerically stable for most standard use cases.

## 4. Core Assumptions

Linear Regression works best when these assumptions are approximately true:

- Linearity: the relationship between features and target is roughly linear
- Independence: observations are independent of each other
- Homoscedasticity: residual variance is roughly constant
- Low multicollinearity: features are not nearly duplicates of each other
- Residuals are roughly normal if you care about statistical inference

Violating these assumptions does not always ruin prediction quality, but it can make coefficients unstable or difficult to interpret.

## 5. Baseline First

Always compare Linear Regression against a simple baseline, usually a `DummyRegressor` with the mean strategy.

That answer tells you whether the model is actually learning signal or just reproducing an average prediction.

## 6. Scikit-Learn Workflow

### Step 1: Import libraries

```python
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

### Step 2: Split the data

Always split before fitting anything.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 3: Train a baseline

```python
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)
```

### Step 4: Train Linear Regression

```python
model = LinearRegression()
model.fit(X_train, y_train)
model_preds = model.predict(X_test)
```

### Step 5: Evaluate both models

```python
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

baseline_metrics = evaluate(y_test, baseline_preds)
model_metrics = evaluate(y_test, model_preds)

print("Baseline:", baseline_metrics)
print("Linear Regression:", model_metrics)
```

## 7. Interpreting Metrics

- MSE: training objective, but less interpretable because it is squared
- RMSE: same units as the target, easier to explain
- MAE: average absolute error, less sensitive to outliers than RMSE
- R2: proportion of variance explained relative to predicting the mean

R2 near 0 means the model is about as good as predicting the mean. Positive values are usually better, but the real question is whether the improvement is meaningful compared to baseline.

## 8. Compare Against the Baseline

Never judge Linear Regression in isolation.

```python
results = {
    "Baseline": evaluate(y_test, baseline_preds),
    "Linear Regression": evaluate(y_test, model_preds),
}

for name, metrics in results.items():
    print(
        f"{name:18s} | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f} | R2: {metrics['R2']:.3f}"
    )
```

If the regression model does not beat the baseline, the features may not contain enough signal, or the relationship may not be linear.

## 9. Coefficient Interpretation

After fitting the model, coefficients can help explain how the features relate to the target.

```python
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_,
}).sort_values("Coefficient", ascending=False)

print(f"Intercept: {model.intercept_:.3f}")
print(coef_df.to_string(index=False))
```

Interpretation:

- Positive coefficient: increasing the feature raises the prediction
- Negative coefficient: increasing the feature lowers the prediction
- Larger magnitude: stronger effect, but only when features are on comparable scales

Coefficient size is not directly comparable unless features are standardized.

## 10. Feature Scaling

Plain Linear Regression does not require scaling for correct predictions, but scaling helps when:

- you want to compare coefficients
- you want faster gradient-based optimization
- you plan to use Ridge or Lasso later
- features have very different numeric ranges

Use a pipeline so scaling happens only on training data.

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)
scaled_preds = pipeline.predict(X_test)
```

## 11. Cross-Validation

A single train/test split can be noisy. Cross-validation gives a better sense of stability.

```python
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
print(f"CV R2 scores: {cv_scores.round(3)}")
print(f"Mean CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

If scores vary a lot across folds, the model may not generalize reliably.

## 12. Common Mistakes

- Fitting preprocessing on the full dataset before splitting
- Interpreting coefficients without checking scaling or multicollinearity
- Assuming a linear relationship without looking at the data
- Ignoring residual plots
- Reporting R2 without comparing to a baseline

## 13. When Linear Regression Works Well

Linear Regression is a strong choice when:

- the relationship is approximately linear
- interpretability matters
- feature engineering is good
- the dataset is not tiny and not extremely high-dimensional
- you want a strong baseline before trying more complex models

## 14. When It Struggles

Linear Regression is weaker when:

- the true relationship is highly non-linear
- the data has strong outliers
- important interactions are missing
- features are highly correlated

In those cases, consider Ridge, Lasso, polynomial features, or tree-based models.

## 15. Practical Checklist

Before calling the model successful, confirm that:

- it beats the mean baseline
- RMSE and MAE are acceptable in business terms
- R2 is meaningfully better than 0
- cross-validation scores are stable
- residuals do not show obvious patterns
- coefficients make sense
- preprocessing is handled in a pipeline

## 16. Final Takeaway

Linear Regression is simple, but it teaches the core ideas behind supervised learning: fit a model on training data, evaluate on held-out data, compare against a baseline, and only trust the result if the evidence supports it.

Build the baseline first. Then train the regression model. Compare honestly. Improve deliberately.