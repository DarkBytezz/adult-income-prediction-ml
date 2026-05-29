import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def tune_xgBoost(X_train, y_train):
    xgb_model =  xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric='logloss',
        random_state=42,
        n_jobs=1
    )
    
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(
        X_train,
        y_train
    )
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search.best_estimator_



def get_tuned_xgboost(
    X_train,
    y_train,
    model_path="models/xg_booster_tuned.pkl"
):

    if os.path.exists(model_path):
        print("Loading tuned XGBoost model...")

        return joblib.load(model_path)

    print("Training tuned XGBoost model...")

    model = tune_xgBoost(
        X_train,
        y_train
    )

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved to {model_path}")

    return model