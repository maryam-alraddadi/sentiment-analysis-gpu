import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import torch
import torch.nn as nn
import torch.optim as optim
import nvtx

# --- PyTorch Model Definition ---
class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    try:
        df = pd.read_csv("IMDB Dataset.csv")
    except FileNotFoundError:
        try:
             # Fallback for Kaggle environment
            df = pd.read_csv("IMDB Dataset.csv")
        except FileNotFoundError:
            print("Error: 'IMDB Dataset.csv' not found.")
            return


    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    print(f"Loaded {len(df)} total reviews.")
    print(f"Processing {len(X_train)} training reviews...")

    # --- 1. Preprocessing (CPU - TF-IDF) ---
    with nvtx.annotate("1. Preprocessing & Vectorization (TF-IDF)", color="blue"):
        start_time = time.time()
        
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
        
        print("Starting TF-IDF fit_transform...")
        X_train_tfidf_sparse = tfidf_vectorizer.fit_transform(X_train)
        print("Finished TF-IDF fit_transform.")
        
        end_time = time.time()
        print(f"TF-IDF took: {end_time - start_time:.2f} seconds")

    # --- 2. Data Conversion for PyTorch ---
    print("Converting data to PyTorch tensors...")
    
    # Convert Sparse Matrix (Scipy) to Dense (Numpy) -> Tensor
    X_train_tensor = torch.tensor(X_train_tfidf_sparse.toarray(), dtype=torch.float32).to(device)
    
    # Convert labels: "positive" -> 1.0, "negative" -> 0.0
    y_train_num = np.where(y_train == 'positive', 1.0, 0.0)
    y_train_tensor = torch.tensor(y_train_num, dtype=torch.float32).view(-1, 1).to(device)

    # --- 3. Model Training (PyTorch) ---
    with nvtx.annotate("2. Model Training (PyTorch)", color="green"):
        start_time = time.time()
        print("Starting PyTorch model training...")
        
        model = LogisticRegressionPyTorch(num_features=20000).to(device)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        epochs = 100
        batch_size = 1024
        model.train()
        
        num_samples = X_train_tensor.size(0)
        
        for epoch in range(epochs):
            # Shuffle indices
            permutation = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                indices = permutation[i : i + batch_size]
                batch_x = X_train_tensor[indices]
                batch_y = y_train_tensor[indices]
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (epoch+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        end_time = time.time()
        print(f"Model training took: {end_time - start_time:.2f} seconds")

    # --- 4. Prediction ---
    print("Running predictions...")
    
    # Transform Test set
    X_test_tfidf_sparse = tfidf_vectorizer.transform(X_test)
    X_test_tensor = torch.tensor(X_test_tfidf_sparse.toarray(), dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted = (outputs > 0.5).float()
    
    # Move back to CPU for accuracy calculation
    y_pred = predicted.cpu().numpy()
    
    # Convert text labels to numeric for comparison
    y_test_num = np.where(y_test == 'positive', 1.0, 0.0)
    
    acc = accuracy_score(y_test_num, y_pred)
    print(f"\nPyTorch Baseline Accuracy: {acc * 100:.2f}%")
    print("Profiling complete.")

if __name__ == "__main__":
    main()