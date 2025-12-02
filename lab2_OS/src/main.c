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
pthread_mutex_t mutex;

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

    for (int k = 0; k < K; k++) {
        for (int i = start_row; i < end_row; i++) {
            apply_erosion(i);
            apply_dilation(i);
        }

        pthread_mutex_lock(&mutex);
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = erosion_result[i][j];
            }
        }
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        return 1;
    }

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

    pthread_mutex_init(&mutex, NULL);

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

    pthread_mutex_destroy(&mutex);

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

