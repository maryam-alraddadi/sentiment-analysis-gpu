import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import nvtx

def main():
    print("Loading dataset...")

    try:
        df = pd.read_csv("IMDB Dataset.csv")
    except FileNotFoundError:
        print("Error: 'IMDB Dataset.csv' not found.")
        print("Please download it from Kaggle and place it in the same directory.")
        return

    
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    print(f"Loaded {len(df)} total reviews.")
    print(f"Processing {len(X_train)} training reviews...")

    # --- This is the key bottleneck section -
    with nvtx.annotate("1. Preprocessing & Vectorization (TF-IDF)", color="blue"):
        start_time = time.time()
        

        # TfidfVectorizer
        # tokenization, cleaning, and TF-IDF computation
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
        
        print("Starting TF-IDF fit_transform...")
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        print("Finished TF-IDF fit_transform.")
        
        end_time = time.time()
        print(f"TF-IDF took: {end_time - start_time:.2f} seconds")
    # -------------------------------------------

    # --- This is the second, smaller CPU task ---
    with nvtx.annotate("2. Model Training (CPU Classifier)", color="green"):
        start_time = time.time()
        print("Starting model training...")
        
        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)
        
        end_time = time.time()
        print(f"Model training took: {end_time - start_time:.2f} seconds")
    # --------------------------------------------

    print("Running predictions...")
   
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nBaseline CPU Accuracy: {acc * 100:.2f}%")
    print("Profiling complete. You can now open the .nsys-rep file.")

if __name__ == "__main__":
    main()