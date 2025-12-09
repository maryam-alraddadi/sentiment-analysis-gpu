import pandas as pd
import numpy as np
import math
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numba import cuda

# --- Configuration ---
TPB = 128           # Threads Per Block
VOCAB_SIZE = 20000  # Number of hash buckets (Features)

# --- 1. GPU Kernels (Numba) ---

@cuda.jit(device=True)
def get_hash(start_idx, end_idx, chars):
    """
    Device function to hash a word into an index (Feature Hashing).
    Uses the DJB2 algorithm.
    """
    hash_val = 5381
    for i in range(start_idx, end_idx):
        c = chars[i]
        # Simple ASCII lowercasing and filtering
        if (c >= 65 and c <= 90) or (c >= 97 and c <= 122): 
            if c <= 90:
                c += 32
            hash_val = ((hash_val << 5) + hash_val) + c
    return hash_val % VOCAB_SIZE

@cuda.jit
def compute_tf_kernel(chars, offsets, out_matrix):
    """
    Kernel 1: Tokenizes text and counts Term Frequency (TF).
    Optimization: Maps 1 Thread -> 1 Document.
    """
    doc_idx = cuda.grid(1)
    
    if doc_idx < len(offsets) - 1:
        start = offsets[doc_idx]
        end = offsets[doc_idx + 1]
        
        word_start = start
        for i in range(start, end):
            # Delimiter check (Space or Newline)
            if chars[i] == 32 or chars[i] == 10: 
                if i > word_start:
                    col_idx = get_hash(word_start, i, chars)
                    out_matrix[doc_idx, col_idx] += 1.0
                word_start = i + 1
        
        # Handle last word  
        if end > word_start:
            col_idx = get_hash(word_start, end, chars)
            out_matrix[doc_idx, col_idx] += 1.0

@cuda.jit
def compute_tfidf_kernel(tf_matrix, doc_count):
    """
    Kernel 2: Apply TF-IDF weighting and normalization.
    """
    row, col = cuda.grid(2)
    
    if row < tf_matrix.shape[0] and col < tf_matrix.shape[1]:
        tf = tf_matrix[row, col]
        if tf > 0:
            # Apply TF-IDF Weighting
            tfidf = tf * math.log(doc_count / (1.0 + tf))
            tf_matrix[row, col] = tfidf

# --- 2. Data Helpers ---

def strings_to_gpu_format(text_list):
    """
    Flattens a list of strings into a single byte array for the GPU.
    """
    encoded_texts = [str(s).encode('utf-8') for s in text_list]
    all_chars = np.frombuffer(b''.join(encoded_texts), dtype=np.uint8)
    
    offsets = [0]
    cumsum = 0
    for s in encoded_texts:
        cumsum += len(s)
        offsets.append(cumsum)
        
    return np.array(all_chars), np.array(offsets)

class NumbaBuffer:
    """
    Allow PyTorch to read Numba GPU memory without copying
    """
    def __init__(self, arr):
        self.arr = arr
    @property
    def __cuda_array_interface__(self):
        return self.arr.__cuda_array_interface__


# --- 3. Main Pipeline ---

def main():
    if not cuda.is_available():
        print("GPU not found")
        return

    print("Loading dataset...")
    if not os.path.exists("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"):
        print("Error: Dataset not found")
        return

    df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
    
    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    print(f"Processing {len(X_train)} training reviews on GPU...")
    total_start = time.time()
    
    # --- PHASE 1: Preprocessing on GPU (Numba) ---
    t0 = time.time()
    
    # 1. Transfer raw text to GPU
    train_chars, train_offsets = strings_to_gpu_format(X_train)
    d_chars = cuda.to_device(train_chars)
    d_offsets = cuda.to_device(train_offsets)
    d_matrix = cuda.device_array((len(X_train), VOCAB_SIZE), dtype=np.float32)
    
    # 2. Run Tokenization Kernel
    blocks_per_grid = (len(X_train) + (TPB - 1)) // TPB
    compute_tf_kernel[blocks_per_grid, TPB](d_chars, d_offsets, d_matrix)
    
    # 3. Run TF-IDF Kernel
    threads_2d = (16, 16)
    blocks_x = (d_matrix.shape[0] + threads_2d[0] - 1) // threads_2d[0]
    blocks_y = (d_matrix.shape[1] + threads_2d[1] - 1) // threads_2d[1]
    compute_tfidf_kernel[(blocks_x, blocks_y), threads_2d](d_matrix, len(X_train))
    
    cuda.synchronize()
    print(f"  > Vectorization took: {time.time() - t0:.2f}s")

    # --- PHASE 2: Training on GPU (PyTorch) ---
    print("Converting to PyTorch Tensors (Zero-Copy)...")
    
    X_train_tensor = torch.as_tensor(NumbaBuffer(d_matrix), device='cuda')
    
    # Prepare Labels
    y_train_num = np.where(y_train == 'positive', 1.0, 0.0).astype(np.float32)
    y_train_tensor = torch.tensor(y_train_num, device='cuda').view(-1, 1)

    # Define Model (Logistic Regression equivalent)
    model = nn.Sequential(
        nn.Linear(VOCAB_SIZE, 1),
        nn.Sigmoid()
    ).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print("Training Model...")
    BATCH_SIZE = 4096
    EPOCHS = 100
    
    model.train()
    t_train = time.time()
    
    for epoch in range(EPOCHS):
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    print(f"  > Training took: {time.time() - t_train:.2f}s")

    # --- PHASE 3: Predictions on GPU ---
    print("Running Predictions on Test Set...")
    
    # 1. Process Test Data (Reuse pipeline)
    test_chars, test_offsets = strings_to_gpu_format(X_test)
    d_test_chars = cuda.to_device(test_chars)
    d_test_offsets = cuda.to_device(test_offsets)
    d_test_matrix = cuda.device_array((len(X_test), VOCAB_SIZE), dtype=np.float32)
    
    blocks_test = (len(X_test) + (TPB - 1)) // TPB
    compute_tf_kernel[blocks_test, TPB](d_test_chars, d_test_offsets, d_test_matrix)
    compute_tfidf_kernel[(blocks_x, blocks_y), threads_2d](d_test_matrix, len(X_train))
    
    # 2. Predict using PyTorch
    X_test_tensor = torch.as_tensor(NumbaBuffer(d_test_matrix), device='cuda')
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        y_pred = (preds > 0.5).cpu().numpy() # Move the result array to CPU
        
    # Calculate Accuracy
    y_test_num = np.where(y_test == 'positive', 1.0, 0.0)
    acc = accuracy_score(y_test_num, y_pred)
    
    print("-" * 30)
    print(f"Final GPU Accuracy: {acc * 100:.2f}%")
    print(f"Total Execution Time: {time.time() - total_start:.2f}s")
    print("-" * 30)

if __name__ == "__main__":
    main()