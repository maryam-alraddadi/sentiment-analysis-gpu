#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <chrono>

#define VOCAB_SIZE 50000
#define MAX_DOC_LEN 1000
#define BATCH_SIZE 5000  

__device__ unsigned long hash_word_device(const char* str, int len) {
    unsigned long hash = 5381;
    for (int i = 0; i < len; i++) {
        char c = str[i];
        if (c >= 'A' && c <= 'Z') c += 32;
        hash = ((hash << 5) + hash) + c;
    }
    return hash % VOCAB_SIZE;
}

__global__ void tf_kernel(const char* all_chars, const int* offsets, float* matrix, int num_docs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_docs) return;

    int start = offsets[idx];
    int end = offsets[idx+1];
    
    int word_start = start;
    for (int i = start; i < end; i++) {
        char c = all_chars[i];
        if (c == ' ' || c == '\0') {
            if (i > word_start) {
                unsigned long vocab_idx = hash_word_device(all_chars + word_start, i - word_start);
                atomicAdd(&matrix[idx * VOCAB_SIZE + vocab_idx], 1.0f);
            }
            word_start = i + 1;
        }
    }
}

__global__ void tfidf_kernel(float* matrix, int num_docs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_docs * VOCAB_SIZE) return;

    float tf = matrix[idx];
    if (tf > 0.0f) {
        float idf = logf((float)num_docs / (1.0f + tf)); 
        matrix[idx] = tf * idf;
    }
}

void load_data(const char* filename, std::vector<char>& h_chars, std::vector<int>& h_offsets) {
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); 
    
    h_offsets.push_back(0);
    while (std::getline(file, line)) {
        for (char c : line) h_chars.push_back(c);
        h_chars.push_back('\0');
        h_offsets.push_back(h_chars.size());
        if (h_offsets.size() > 25000) break;
    }
}

int main() {
    // 1. Start Total Timer (CPU Wall Clock)
    auto start_total = std::chrono::high_resolution_clock::now();

    std::vector<char> h_chars;
    std::vector<int> h_offsets;
    load_data("IMDB Dataset.csv", h_chars, h_offsets);
    int total_docs = h_offsets.size() - 1;
    printf("Loaded %d docs.\n", total_docs);

    float *d_matrix;
    char *d_chars;
    int *d_offsets;
    
    cudaMalloc(&d_matrix, total_docs * VOCAB_SIZE * sizeof(float));
    cudaMalloc(&d_chars, h_chars.size() * sizeof(char));
    cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int));
    cudaMemset(d_matrix, 0, total_docs * VOCAB_SIZE * sizeof(float));

    int n_streams = 4;
    cudaStream_t streams[4];
    for (int i=0; i<n_streams; i++) cudaStreamCreate(&streams[i]);

    cudaMemcpy(d_chars, h_chars.data(), h_chars.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Start Vectorization Timer (CUDA Events)
    cudaEvent_t start_vec, stop_vec;
    cudaEventCreate(&start_vec);
    cudaEventCreate(&stop_vec);
    
    cudaEventRecord(start_vec); // Record Start

    int block_size = 256;
    for (int i = 0; i < total_docs; i += BATCH_SIZE) {
        int stream_id = (i / BATCH_SIZE) % n_streams;
        int current_batch = std::min(BATCH_SIZE, total_docs - i);
        int *batch_offsets = d_offsets + i; 
        float *batch_matrix = d_matrix + (i * VOCAB_SIZE);
        int grid_size = (current_batch + block_size - 1) / block_size;
        
        tf_kernel<<<grid_size, block_size, 0, streams[stream_id]>>>(
            d_chars, batch_offsets, batch_matrix, current_batch
        );
    }

    cudaDeviceSynchronize();
    
    int total_elements = total_docs * VOCAB_SIZE;
    tfidf_kernel<<<(total_elements + 255)/256, 256>>>(d_matrix, total_docs);
    
    cudaEventRecord(stop_vec); // Record Stop
    cudaEventSynchronize(stop_vec);

    // 3. Stop Total Timer
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_total = end_total - start_total;

    // Calculate CUDA Event Time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_vec, stop_vec);

    printf("------------------------------------------------\n");
    printf("Vectorization & TF-IDF Time: %.4f seconds\n", milliseconds / 1000.0f);
    printf("Total Execution Time:        %.4f seconds\n", diff_total.count());
    printf("------------------------------------------------\n");

    cudaFree(d_matrix);
    cudaFree(d_chars);
    cudaFree(d_offsets);
    return 0;
}