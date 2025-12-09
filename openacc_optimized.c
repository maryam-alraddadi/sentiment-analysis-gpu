#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>       
#include <openacc.h>

// --- Configuration ---
#define MAX_DOCS 5000       
#define MAX_LEN 1000         
#define VOCAB_SIZE 20000     
#define HASH_SEED 5381

// --- Helper for Wall-Clock Time ---
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#pragma acc routine seq
unsigned long hash_word(char *str, int len) {
    unsigned long hash = HASH_SEED;
    for (int i = 0; i < len; i++) {
        char c = str[i];
        if (c >= 'A' && c <= 'Z') c += 32; 
        hash = ((hash << 5) + hash) + c;
    }
    return hash % VOCAB_SIZE;
}

int load_csv(const char *filename, char *raw_text_flat) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;
    char buffer[MAX_LEN + 1024];
    int count = 0;
    fgets(buffer, sizeof(buffer), fp); 
    while (fgets(buffer, sizeof(buffer), fp) && count < MAX_DOCS) {
        char *dest = &raw_text_flat[count * MAX_LEN];
        char *src = buffer;
        if (*src == '"') {
            src++;
            int j = 0;
            while (*src && !(*src == '"' && *(src+1) == ',') && j < MAX_LEN-1) {
                if (*src == '"' && *(src+1) == '"') src++; 
                dest[j++] = *src++;
            }
            dest[j] = '\0';
        } else {
            int j = 0;
            while (*src && *src != ',' && j < MAX_LEN-1) dest[j++] = *src++;
            dest[j] = '\0';
        }
        count++;
    }
    fclose(fp);
    return count;
}

int main() {
    // 1. Start Total Timer
    double total_start = get_time();

    float *tfidf_matrix = (float*)calloc(MAX_DOCS * VOCAB_SIZE, sizeof(float));
    char *raw_text = (char*)malloc(MAX_DOCS * MAX_LEN * sizeof(char));

    printf("Loading Dataset...\n");
    int num_docs = load_csv("IMDB Dataset.csv", raw_text);
    if (num_docs == 0) { printf("Error: Dataset not found.\n"); return 1; }

    // 2. Start Vectorization Timer
    double vec_start = get_time();

    #pragma acc data copyin(raw_text[0:num_docs*MAX_LEN]) copyout(tfidf_matrix[0:num_docs*VOCAB_SIZE])
    {
        #pragma acc parallel loop vector_length(128)
        for (int i = 0; i < num_docs; i++) {
            char *doc = &raw_text[i * MAX_LEN];
            int word_start = 0;
            for (int j = 0; j < MAX_LEN; j++) {
                char c = doc[j];
                if (c == ' ' || c == '\0') {
                    if (j > word_start) {
                        unsigned long idx = hash_word(doc + word_start, j - word_start);
                        #pragma acc atomic update
                        tfidf_matrix[i * VOCAB_SIZE + idx] += 1.0f;
                    }
                    word_start = j + 1;
                    if (c == '\0') break;
                }
            }
        }

        #pragma acc kernels
        {
            for (int j = 0; j < VOCAB_SIZE; j++) {
                float df = 0;
                for (int i = 0; i < num_docs; i++) {
                    if (tfidf_matrix[i * VOCAB_SIZE + j] > 0) df++;
                }
                if (df > 0) {
                    float idf = logf((float)num_docs / (1.0f + df));
                    for (int i = 0; i < num_docs; i++) {
                        if (tfidf_matrix[i * VOCAB_SIZE + j] > 0) {
                            tfidf_matrix[i * VOCAB_SIZE + j] *= idf;
                        }
                    }
                }
            }
        }
    } 
    
    // 3. Stop Vectorization Timer
    double vec_end = get_time();

    // 4. Stop Total Timer
    double total_end = get_time();

    // Print Results
    printf("------------------------------------------------\n");
    printf("Vectorization & TF-IDF Time: %.4f seconds\n", vec_end - vec_start);
    printf("Total Execution Time:        %.4f seconds\n", total_end - total_start);
    printf("------------------------------------------------\n");

    free(tfidf_matrix);
    free(raw_text);
    return 0;
}