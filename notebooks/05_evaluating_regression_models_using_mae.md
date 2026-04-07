# Evaluating Regression Models Using MAE

Training a regression model is only half the task. The real question is how good its predictions actually are.

Mean Absolute Error (MAE) is one of the most intuitive regression metrics because it measures average prediction error in the same units as the target variable.

If you are predicting house prices in lakhs and your MAE is 3.5, your predictions are off by about ₹3.5 lakhs on average.

## 1. What MAE Measures

MAE is the average absolute difference between actual and predicted values:

```text
MAE = (1/n) * sum(|y_i - y_hat_i|)
```

Where:

- `y_i` is the true value
- `y_hat_i` is the predicted value
- `n` is the number of samples

The absolute value makes all errors positive, so over-predictions and under-predictions count equally.

## 2. Why MAE Is Useful

MAE is easy to explain to non-technical stakeholders because it answers a simple question:

- On average, how far off are our predictions?

That makes it especially useful when business users care about error in real-world units rather than squared error or variance explained.

## 3. MAE vs MSE vs RMSE

The three common regression error metrics behave differently:

- MAE: linear penalty, same units as the target, lower sensitivity to outliers
- MSE: quadratic penalty, squared target units, high outlier sensitivity
- RMSE: square root of MSE, same units as the target, still outlier-sensitive

Choose MAE when average error magnitude matters more than strongly penalizing rare large mistakes. Choose RMSE when large misses are especially costly.

## 4. Simple Example

Suppose the errors are `[-3, 5, 2, -2]`.

The absolute errors are `[3, 5, 2, 2]`, so the MAE is `3.0`.

That means predictions are off by 3 units on average.

## 5. Compute MAE in scikit-learn

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")
```

## 6. Compare Against a Baseline

MAE in isolation is not very meaningful. Always compare your model against a baseline, usually a `DummyRegressor` with the mean strategy.

```python
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
model_preds = model.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_preds)
model_mae = mean_absolute_error(y_test, model_preds)
improvement = baseline_mae - model_mae
pct_improve = (improvement / baseline_mae) * 100

print(f"Baseline MAE:  {baseline_mae:.2f}")
print(f"Model MAE:     {model_mae:.2f}")
print(f"Improvement:   {improvement:.2f} ({pct_improve:.1f}%)")
```

## 7. Interpreting MAE

A MAE value only becomes useful when you place it in context:

- Target scale: compare MAE to the typical size of the target
- Baseline performance: check whether the model beats a simple mean predictor
- Business tolerance: decide whether the error is acceptable for the use case

You can also express MAE as a percentage of the mean target value for a quick scale check.

## 8. Cross-Validation with MAE

Cross-validation gives a more stable estimate than a single train/test split.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
)

mae_scores = -cv_scores
print(f"CV MAE per fold: {mae_scores.round(2)}")
print(f"Mean CV MAE:     {mae_scores.mean():.3f}")
print(f"Std CV MAE:      {mae_scores.std():.3f}")
```

scikit-learn returns negative MAE because its scoring convention assumes larger values are better.

## 9. When MAE Is a Good Choice

Use MAE when:

- interpretability matters
- outliers should not dominate evaluation
- errors have roughly uniform cost
- you need a clear message for business stakeholders

## 10. When MAE Is Not Enough

MAE hides some important failure modes:

- It does not show directional bias
- It does not strongly penalize rare but severe misses
- It can look fine even when a few predictions are catastrophically wrong

That is why residual plots and RMSE are still useful alongside MAE.

## 11. Common Mistakes

- Reporting MAE without a baseline
- Mixing MAE with RMSE when comparing models
- Computing MAE on transformed targets and interpreting it in the wrong units
- Ignoring residual plots and systematic bias
- Using training MAE instead of held-out MAE

## 12. Practical Checklist

Before reporting MAE, confirm that:

- it was computed on test data only
- it is compared with a baseline on the same split
- it is meaningful relative to the target scale
- cross-validation supports the result
- residuals do not show a pattern
- you have also checked RMSE and R2

## 13. Final Takeaway

MAE directly answers the question every practitioner should ask first:

- On average, how wrong are we?

It is simple, transparent, and easy to communicate. Used with a baseline and cross-validation, it becomes a reliable way to judge whether a regression model is actually useful.