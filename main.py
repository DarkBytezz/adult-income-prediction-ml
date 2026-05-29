from src.preprocessing import (
    load_data, 
    replace_question_marks, 
    fill_missing_values, 
    split_data, 
    build_preprocessor
)
from src.utils import(
    dataset_overview, 
    check_missing_values, 
    get_feature_type, 
    target_dist, 
    data_quality_report
) 
from src.visualize import (
    save_leaderboard,
    save_confusion_matrix,
)   
from src.train import train_model, get_tuned_xgboost
from src.evaluate import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib, os
if not os.path.exists("models"):
    os.makedirs("models")

print("Project Started")

df = load_data("data/adult.csv")

(
    
# dataset_overview(df)
# print(f"\nBefore replacing question marks: ")
# check_missing_values(df)

# print(f"\nAfter replacing question marks: ")
# check_missing_values(df)

# num_cols, categorical_cols = get_feature_type(df)

# print(f"Numerical columns: {num_cols}")
# print()
# print(f"Cateogrical columns: {categorical_cols}")


# count, count_precentage = target_dist(df)
# print(count)
# print(count_precentage)

)


df = replace_question_marks(df)
df = fill_missing_values(df)
report = data_quality_report(df)

X_train, X_test, y_train, y_test = split_data(df)

num_cols, categorical_cols = get_feature_type(X_train)
preprocessor = build_preprocessor(num_cols, categorical_cols)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

le = LabelEncoder()

y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)







models = {
    "Logistic Regression": LogisticRegression(
        random_state=42,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        max_iter=1000
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=3
    ),

    "Naive Bayes": BernoulliNB(),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),

    "XGBoost": xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    
}



if not os.path.exists("models/preprocessor.pkl"):
    joblib.dump(preprocessor, "models/preprocessor.pkl")

if not os.path.exists("models/label_encoder.pkl"):
    joblib.dump(le, "models/label_encoder.pkl")





trained_models = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")

    trained_models[model_name] = train_model(
        model,
        X_train_encoded,
        y_train_encoded
    )

xg_booster_tuned = get_tuned_xgboost(
    X_train_encoded,
    y_train_encoded
)
trained_models[
    "XGBoost Tuned"
] = xg_booster_tuned


results = []

save_confusion_matrix(
    xg_booster_tuned,
    X_test_encoded,
    y_test_encoded
)

for model_name, model in trained_models.items():
    print(f"EVALUATING {model_name}")
    results.append(
        evaluate_model(
            model_name,
            model,
            X_test_encoded,
            y_test_encoded
        )
    )

leaderboard = pd.DataFrame(results)

leaderboard = (
    leaderboard
    .sort_values(by="F1 Score", ascending=False)
    .reset_index(drop=True)
    .round(4)
    )

print(leaderboard)
print()
save_leaderboard(leaderboard)