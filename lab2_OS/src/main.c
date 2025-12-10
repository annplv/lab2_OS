#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define MAX_THREADS 16

int num_threads;
int rows, cols;
int K;
int window_radius;
float **matrix;
float **erosion_result;
float **dilation_result;
pthread_mutex_t *row_mutex;
pthread_barrier_t barrier;

void apply_erosion(int row) {
    for (int j = 0; j < cols; j++) {
        float min_val = matrix[row][j];
        for (int dx = -window_radius; dx <= window_radius; dx++) {
            for (int dy = -window_radius; dy <= window_radius; dy++) {
                int ni = row + dx;
                int nj = j + dy;
                if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                    if (matrix[ni][nj] < min_val) {
                        min_val = matrix[ni][nj];
                    }
                }
            }
        }
        erosion_result[row][j] = min_val;
    }
}

void apply_dilation(int row) {
    for (int j = 0; j < cols; j++) {
        float max_val = matrix[row][j];
        for (int dx = -window_radius; dx <= window_radius; dx++) {
            for (int dy = -window_radius; dy <= window_radius; dy++) {
                int ni = row + dx;
                int nj = j + dy;
                if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                    if (matrix[ni][nj] > max_val) {
                        max_val = matrix[ni][nj];
                    }
                }
            }
        }
        dilation_result[row][j] = max_val;
    }
}

void* thread_func(void* arg) {
    int thread_id = *(int*)arg;
    int rows_per_thread = rows / num_threads;
    int start_row = thread_id * rows_per_thread;
    int end_row = (thread_id == num_threads - 1) ? rows : start_row + rows_per_thread;

    float **temp_erosion = (float**)malloc(rows * sizeof(float*));
    float **temp_dilation = (float**)malloc(rows * sizeof(float*));
    
    for (int i = start_row; i < end_row; i++) {
        temp_erosion[i] = (float*)malloc(cols * sizeof(float));
        temp_dilation[i] = (float*)malloc(cols * sizeof(float));
        
        for (int j = 0; j < cols; j++) {
            temp_erosion[i][j] = matrix[i][j];
            temp_dilation[i][j] = matrix[i][j];
        }
    }
    //эрозия
    for (int k = 0; k < K; k++) {
        for (int i = start_row; i < end_row; i++) {
            pthread_mutex_lock(&row_mutex[i]);
            memcpy(matrix[i], temp_erosion[i], cols * sizeof(float));
            pthread_mutex_unlock(&row_mutex[i]);
        }
        pthread_barrier_wait(&barrier);

        for (int i = start_row; i < end_row; i++) {
            apply_erosion(i);
        }
        pthread_barrier_wait(&barrier);

        for (int i = start_row; i < end_row; i++) {
            memcpy(temp_erosion[i], erosion_result[i], cols * sizeof(float));
        }
        pthread_barrier_wait(&barrier);
    }

    for (int i = start_row; i < end_row; i++) {
        memcpy(erosion_result[i], temp_erosion[i], cols * sizeof(float));
    }
    pthread_barrier_wait(&barrier);

    //дилатация
    for (int k = 0; k < K; k++) {
        for (int i = start_row; i < end_row; i++) {
            pthread_mutex_lock(&row_mutex[i]);
            memcpy(matrix[i], temp_dilation[i], cols * sizeof(float));
            pthread_mutex_unlock(&row_mutex[i]);
        }
        pthread_barrier_wait(&barrier);

        for (int i = start_row; i < end_row; i++) {
            apply_dilation(i);
        }
        pthread_barrier_wait(&barrier);
        
        for (int i = start_row; i < end_row; i++) {
            memcpy(temp_dilation[i], dilation_result[i], cols * sizeof(float));
        }
        pthread_barrier_wait(&barrier);
    }
    
    for (int i = start_row; i < end_row; i++) {
        memcpy(dilation_result[i], temp_dilation[i], cols * sizeof(float));
    }
    pthread_barrier_wait(&barrier);

    for (int i = start_row; i < end_row; i++) {
        free(temp_erosion[i]);
        free(temp_dilation[i]);
    }
    free(temp_erosion);
    free(temp_dilation);

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        return 1;
    }

    srand(time(NULL));

    num_threads = atoi(argv[1]);
    rows = atoi(argv[2]);
    cols = atoi(argv[3]);
    K = atoi(argv[4]);
    window_radius = atoi(argv[5]);

    if (num_threads > MAX_THREADS) {
        return 1;
    }

    matrix = (float**)malloc(rows * sizeof(float*));
    erosion_result = (float**)malloc(rows * sizeof(float*));
    dilation_result = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
        erosion_result[i] = (float*)malloc(cols * sizeof(float));
        dilation_result[i] = (float*)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (float)rand() / RAND_MAX * 100.0;
        }
    }

    printf("Исходная матрица:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    row_mutex = malloc(rows * sizeof(pthread_mutex_t));
    for (int i = 0; i < rows; ++i){ 
        pthread_mutex_init(&row_mutex[i], NULL);
    }
    pthread_barrier_init(&barrier, NULL, num_threads);

    clock_t start_time = clock();

    pthread_t threads[MAX_THREADS];
    int thread_ids[MAX_THREADS];

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t end_time = clock();
    double spent_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Результат эрозии:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", erosion_result[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Результат дилатации:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", dilation_result[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Время выполнения: %.6f секунд\n", spent_time);
    printf("Обработано элементов: %d\n", rows * cols * K * 2);
    printf("Производительность: %.2f элементов/сек\n", 
           (rows * cols * K * 2) / spent_time);

    pthread_barrier_destroy(&barrier);
    for (int i = 0; i < rows; ++i){ 
        pthread_mutex_destroy(&row_mutex[i]);
    }
    free(row_mutex);

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
        free(erosion_result[i]);
        free(dilation_result[i]);
    }
    free(matrix);
    free(erosion_result);
    free(dilation_result);

    return 0;
}

