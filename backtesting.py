from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def backtest(model, alpha_grid, x, y, grid_search_cv, tune_every):
    mse_list = []
    predictions_df = pd.DataFrame(columns=["True Value", "Predicted Value"])
    for i in range(12, len(x) - 1):
        # Define training and testing sets
        X_train = x[:i]
        y_train = y[:i]
        X_test = x[i:i + 1]
        y_test = y[i:i + 1]
        
        # Fine-tuning every 12 iterations
        if i % tune_every == 0:
            time_series_cv  = TimeSeriesSplit()
            grid_search_cv = GridSearchCV(model, param_grid=alpha_grid, cv=time_series_cv, scoring='neg_mean_squared_error')
            grid_search_cv.fit(X_train, y_train)
            model = grid_search_cv.best_estimator_

        # Train the tuned model
        model.fit(X_train, y_train)
        
        # Predict the test set
        y_pred = model.predict(X_test)
        for date, true_val, pred_val in zip(y_test.index, y_test, y_pred):
            predictions_df.loc[date] = [true_val, pred_val]

        # Evaluate the prediction
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(np.sqrt(mse))
    
    return [np.mean(mse_list),predictions_df]