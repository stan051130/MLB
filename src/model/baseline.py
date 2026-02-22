import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

FEATURES = [
    #list of column names to use as input to the model
    "rolling_PTS",
    "rolling_FG_PCT",
    "rolling_AST",
    "rolling_REB",
    "rolling_target_win"
]

TARGET = "target_win"

def main():
    df = pd.read_csv("data/processed.csv")
    
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    base_rate = y_train.mean()
    y_prob_base = [base_rate] * len(y_test)
    y_pred_base = [1 if base_rate >= 0.5 else 0] * len(y_test)

    print("Base rate (train):", base_rate)
    print("Baseline accuracy:", accuracy_score(y_test, y_pred_base))
    print("Baseline log loss:", log_loss(y_test, y_prob_base))
    
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test,y_prob)
    
    print("accuracy score: ",acc)
    print("log loss:", ll)
    
if __name__ == "__main__":
    main()