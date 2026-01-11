import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

FEATURES = [
    #list of column names to use as input to the model
    "IDK yet"
]

TARGET = "HOME_WIN"

def main():
    df = pd.read_csv("data/processed.csv")
    
    X = df[FEATURES]
    y = df[TARGET]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X,y)
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:1]
    
    acc = accuracy_score(y, y_pred)
    ll = log_loss(y,y_prob)
    
if __name__ == "__main__":
    main()