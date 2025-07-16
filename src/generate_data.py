import numpy as np
import pandas as pd
import os
from scipy.special import softmax

def generate_emotion_data(n=10000, seed=42):
    np.random.seed(seed)

    sleep_hours = np.clip(np.random.normal(7, 1, n), 4, 10)
    steps = np.clip(np.random.normal(6000, 1200, n), 2000, 12000)

    alcohol_prob = np.random.rand(n)
    alcohol = np.where(alcohol_prob < 0.5, 0, np.random.poisson(3, n))
    alcohol = np.clip(alcohol, 0, 8)

    social_mins = np.clip(np.random.normal(90, 30, n), 0, 240)
    work_stress = np.clip(np.random.normal(5, 2, n), 0, 10)
    nutrition_score = np.clip(np.random.normal(6.5, 1.5, n), 0, 10)

    # Feature correlations
    sleep_hours -= alcohol * 0.1
    nutrition_score += (steps - 6000) / 6000

    noise = np.random.normal(0, 1, (n, 3))

    happy_score = (
        sleep_hours * 0.6 +
        (10 - work_stress) * 0.5 +
        social_mins * 0.03 +
        nutrition_score * 0.6 -
        alcohol * 0.3 +
        noise[:, 0]
    )

    energetic_score = (
        steps * 0.001 +
        sleep_hours * 0.5 -
        alcohol * 0.4 +
        nutrition_score * 0.4 -
        work_stress * 0.3 +
        noise[:, 1]
    )

    engaged_score = (
        work_stress * -0.2 +
        social_mins * 0.05 +
        nutrition_score * 0.3 +
        sleep_hours * 0.4 +
        noise[:, 2]
    )

    logits = np.stack([happy_score, energetic_score, engaged_score], axis=1)
    prob = softmax(logits, axis=1)
    predicted_class = np.array(['Happy', 'Energetic', 'Engaged'])[np.argmax(prob, axis=1)]

    df = pd.DataFrame({
        'sleep_hours': sleep_hours,
        'steps': steps,
        'alcohol': alcohol,
        'social_mins': social_mins,
        'work_stress': work_stress,
        'nutrition_score': nutrition_score,
        'happy_prob': prob[:, 0],
        'energetic_prob': prob[:, 1],
        'engaged_prob': prob[:, 2],
        'predicted_emotion': predicted_class
    })

    return df

def save_data(df, folder='output'):
    os.makedirs(folder, exist_ok=True)
    df.to_csv(f'{folder}/emotion_data.csv', index=False)
    df.to_excel(f'{folder}/emotion_data.xlsx', index=False)
    print(f"Saved dataset to {folder}/emotion_data.csv and .xlsx")

if __name__ == "__main__":
    df = generate_emotion_data()
    save_data(df)