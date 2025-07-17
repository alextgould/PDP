import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

def load_data(path='output/emotion_data.csv'):
    return pd.read_csv(path)

def train_and_save_models(df, folder='output'):
    X = df[['sleep_hours', 'steps', 'alcohol', 'social_mins', 'work_stress', 'nutrition_score']]
    y = df['predicted_emotion']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )

    param_grid = {
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.1, 0.2],
        'xgb__n_estimators': [50, 100]
    }

    os.makedirs(folder, exist_ok=True)

    for use_scaler in [False, True]:
        steps = []
        model_label = "xgb_nozscore" if not use_scaler else "xgb_zscore"

        if use_scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('xgb', XGBClassifier(eval_metric='mlogloss', random_state=42)))

        pipe = Pipeline(steps)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)

        print(f"\nTraining model: {model_label}")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        print(f"\nClassification report for {model_label}:\n")
        print(classification_report(y_test, y_pred))

        model_path = os.path.join(folder, f"{model_label}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    df = load_data()
    train_and_save_models(df)