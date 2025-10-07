import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

try:
    df = pd.read_csv("text.csv")
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])

    print("Training the model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Completed.\nAccuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "emotion_model_logistic.pkl")
    print("Model saved as emotion_model_logistic.pkl")

except FileNotFoundError:
    print("Error: shuffled_and_selected_dataset.csv not found. Please ensure the file exists in the correct location.")
except KeyError as e:
    print(f"Error: Column '{e}' not found in the dataset. Please check the CSV file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
