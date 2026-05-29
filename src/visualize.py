import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


def save_leaderboard(leaderboard):

    leaderboard = leaderboard.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.axis("off")

    table = ax.table(
        cellText=leaderboard.values,
        colLabels=leaderboard.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ncols = len(leaderboard.columns)

    # Header
    for col in range(ncols):
        table[(0, col)].set_facecolor("#1E3A8A")
        table[(0, col)].set_text_props(
            color="white",
            weight="bold"
        )
        
    for col in range(ncols):
        table[(1, col)].set_facecolor("#FFD700")
    for col in range(ncols):
        table[(2, col)].set_facecolor("#C0C0C0")
    for col in range(ncols):
        table[(3, col)].set_facecolor("#CD7F32")
    last_row = len(leaderboard)

    for col in range(ncols):
        table[(last_row, col)].set_facecolor("#FFCCCC")
        
    f1_col = (
        leaderboard.columns
        .get_loc("F1 Score")
    )
    
    table[(1, f1_col)].set_text_props(
        weight="bold",
        color="darkgreen"
    )
    
    plt.title(
        "Adult Income Prediction - Model Leaderboard",
        fontsize=16,
        weight="bold",
        pad=20
    )
    
    plt.savefig(
        "reports/model_leaderboard.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def save_confusion_matrix(
        model,
        X_test,
        y_test
    ):

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title("XGBoost Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        "reports/xgboost_confusion_matrix.png",
        dpi=300
    )

    plt.close()
    
    
    