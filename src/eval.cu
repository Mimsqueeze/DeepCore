#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <numeric>
#include <cuda_runtime.h>
#include <iomanip>
#include <curand_kernel.h>

#define TRAIN_LABELS_FILE_PATH R"(.\data\train-labels.idx1-ubyte)"
#define TRAIN_IMAGES_FILE_PATH R"(.\data\train-images.idx3-ubyte)"
#define TEST_LABELS_FILE_PATH R"(.\data\t10k-labels.idx1-ubyte)"
#define TEST_IMAGES_FILE_PATH R"(.\data\t10k-images.idx3-ubyte)"

#define LABEL_START 8
#define IMAGE_START 16
#define BATCH_SIZE 1

#define NUM_TRAIN_IMAGES 60000
#define NUM_BATCHES (NUM_TRAIN_IMAGES/BATCH_SIZE)
#define NUM_TEST_IMAGES 10000
#define LEARNING_RATE 0.1f
#define NUM_EPOCHS 100

#define INPUT_SIZE 784
#define L1_SIZE 250
#define OUTPUT_SIZE 10

#define BLOCK_SIZE 256
#define DEBUG_MODE true

#define WEIGHTS_AND_BIASES_FILE_PATH R"(.\src\models\wandb.bin)"

#define CHECK_CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS_ERROR(err) { \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << _cudaGetErrorEnum(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

const char* _cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "<unknown>";
    }
}

using namespace std;

cublasHandle_t HANDLE;
const unsigned SEED = chrono::system_clock::now().time_since_epoch().count();
const float ALPHA = 1.0f;
const float BETA = 0.0f;

void multiply_tensor_T(float *gpu_A, float *gpu_B, float *gpu_C, int batch_size, int m, int n, int k) {
    // Allocate arrays for the pointers
    float** A_array = new float*[batch_size];
    float** B_array = new float*[batch_size];
    float** C_array = new float*[batch_size];

    for (int i = 0; i < batch_size; ++i) {
        A_array[i] = gpu_A + i * m * k;
        B_array[i] = gpu_B + i * k * n;
        C_array[i] = gpu_C + i * m * n;
    }

    float** gpu_A_array;
    float** gpu_B_array;
    float** gpu_C_array;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A_array, batch_size * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B_array, batch_size * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_C_array, batch_size * sizeof(float*)));

    CHECK_CUDA_ERROR(cudaMemcpy(gpu_A_array, A_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B_array, B_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_C_array, C_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

    // Transpose A and B
    CHECK_CUBLAS_ERROR(cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                                          &ALPHA, (const float**)gpu_A_array, k,
                                          (const float**)gpu_B_array, n,
                                          &BETA, gpu_C_array, m, batch_size));

    // Free dynamically allocated arrays
    delete[] A_array;
    delete[] B_array;
    delete[] C_array;

    CHECK_CUDA_ERROR(cudaFree(gpu_A_array));
    CHECK_CUDA_ERROR(cudaFree(gpu_B_array));
    CHECK_CUDA_ERROR(cudaFree(gpu_C_array));
}

void multiply_tensor(float *gpu_A, float *gpu_B, float *gpu_C, int batch_size, int m, int n, int k) {
    // Allocate arrays for the pointers
    float** A_array = new float*[batch_size]{0};
    float** B_array = new float*[batch_size]{0};
    float** C_array = new float*[batch_size]{0};

    for (int i = 0; i < batch_size; ++i) {
        A_array[i] = gpu_A + i * m * k;
        B_array[i] = gpu_B + i * k * n;
        C_array[i] = gpu_C + i * m * n;
    }

    float** gpu_A_array;
    float** gpu_B_array;
    float** gpu_C_array;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A_array, batch_size * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B_array, batch_size * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_C_array, batch_size * sizeof(float*)));

    CHECK_CUDA_ERROR(cudaMemcpy(gpu_A_array, A_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B_array, B_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_C_array, C_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

    // Perform batched matrix multiplication
    CHECK_CUBLAS_ERROR(cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                          &ALPHA, (const float**)gpu_A_array, m,
                                          (const float**)gpu_B_array, k,
                                          &BETA, gpu_C_array, m, batch_size));

    // Free dynamically allocated arrays
    delete[] A_array;
    delete[] B_array;
    delete[] C_array;

    CHECK_CUDA_ERROR(cudaFree(gpu_A_array));
    CHECK_CUDA_ERROR(cudaFree(gpu_B_array));
    CHECK_CUDA_ERROR(cudaFree(gpu_C_array));
}

// CUDA kernel to add the bias vector to the activation matrix
__global__ void add_bias_kernel(float *gpu_A, float *gpu_B, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        int row = idx % rows;
        gpu_A[idx] += gpu_B[row];
    }
}

void add_bias(float *gpu_A, float *gpu_B, int rows, int cols) {
    int num_blocks = (rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_A, gpu_B, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// CUDA kernel to apply ReLU function to the activation matrix
__global__ void apply_ReLU_kernel(float *gpu_A, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        if (gpu_A[idx] < 0) {
            gpu_A[idx] = 0;
        }
    }
}

void apply_ReLU(float *gpu_A, int rows, int cols) {
    int num_blocks = (rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_ReLU_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_A, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// CUDA kernel to apply softmax function to the activation matrix
__global__ void softmax_kernel(float *gpu_A, float *NORM, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        gpu_A[idx] = exp(gpu_A[idx]) / NORM[idx / rows];
    }
}

// CUDA kernel to compute softmax normalizing constants
__global__ void softmax_norm_kernel(float *gpu_A, float *NORM, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        atomicAdd(&NORM[idx / rows], exp(gpu_A[idx]));
    }
}

void apply_softmax(float *gpu_A, int rows, int cols) {
    // Declare GPU memory
    float *gpu_NORM;

    // Allocate and initialize GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_NORM, 1 * cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(gpu_NORM, 0, 1 * cols * sizeof(float)));

    // Launch CUDA kernel
    int num_blocks = (rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_norm_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_A, gpu_NORM, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Launch CUDA kernel
    softmax_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_A, gpu_NORM, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaFree(gpu_NORM));
}

// // CUDA kernel to compute the dC/dA2 Jacobian
// __global__ void compute_dC_dA2(float *gpu_dC_dA2, float *gpu_A2, float *gpu_Y, int rows, int cols) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalElements = rows * cols;
//     if (idx < totalElements) {
//         int col = idx / rows; // j
//         int row = idx % rows; // i
//         int col_T = row;
//         int row_T = col;
//         int rows_T = cols;
//         // Using cross-entropy loss function
//         // gpu_dC_dA2[idx] = -gpu_Y[col_T * rows_T + row_T] / gpu_A2[col_T * rows_T + row_T];

//         // Using mean-square error loss function
//         gpu_dC_dA2[idx] = gpu_A2[col_T * rows_T + row_T] - gpu_Y[col_T * rows_T + row_T];
//     }
// }



// // CUDA kernel to update params in gpu_P
// __global__ void subtract_derivs(float *gpu_P, float *gpu_dP, float learning_rate, int rows, int cols, int slices) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalElements = rows * cols;
//     if (idx < totalElements) { // let's just use a loop here
//         float avg_deriv = 0;
//         for (int slice = 0; slice < slices; slice++) {
//             avg_deriv += gpu_dP[idx + (slice * rows * cols)];
//         }
//         avg_deriv /= slices;
//         gpu_P[idx] = gpu_P[idx] - learning_rate * avg_deriv;
//     }
// }

// Function to print a tensor
void print_tensor(string name, const float *gpu_tensor, int slices, int rows, int cols) {
    float *tensor = new float[slices * rows * cols];
    CHECK_CUDA_ERROR(cudaMemcpy(tensor, gpu_tensor, slices * rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    cout << name;
    printf(" (%d, %d, %d)\n", slices, rows, cols);
    cout << fixed << setprecision(3);
    for (int b = 0; b < slices; b++) {
        cout << "Slice " << b << ": " << endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << tensor[b * rows * cols + j * rows + i] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    delete[] tensor;
}

void forward_prop(float *gpu_X, float *gpu_A1, float *gpu_A2, float *gpu_W1, float *gpu_B1, float *gpu_W2, float *gpu_B2) {

    // Perform A1 = W1*X
    multiply_tensor(gpu_W1, gpu_X, gpu_A1, 1, L1_SIZE, BATCH_SIZE, INPUT_SIZE);

    // Perform A1 = A1 + B1
    add_bias(gpu_A1, gpu_B1, L1_SIZE, BATCH_SIZE);

    // Perform A1 = ReLU(A1)
    apply_ReLU(gpu_A1, L1_SIZE, BATCH_SIZE);

    // Perform A2 = W2*A1
    multiply_tensor(gpu_W2, gpu_A1, gpu_A2, 1, OUTPUT_SIZE, BATCH_SIZE, L1_SIZE);

    // Perform A2 = A2 + B2
    add_bias(gpu_A2, gpu_B2, OUTPUT_SIZE, BATCH_SIZE);

    // Perform A2 = softmax(A2)
    apply_softmax(gpu_A2, OUTPUT_SIZE, BATCH_SIZE);
}

void get_label_batch(float (&Y)[OUTPUT_SIZE * BATCH_SIZE], const int offsets[NUM_TRAIN_IMAGES], int index) {
    ifstream labels_file(TRAIN_LABELS_FILE_PATH, ios::in | ios::binary);
    if (labels_file.is_open()) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            labels_file.seekg(LABEL_START + offsets[index + i]);
            int label;
            labels_file.read((char *) &label, 1);
            for (int j = 0; j < 10; j++) {
                if (j == label) {
                    Y[j + i * 10] = 1;
                } else {
                    Y[j + i * 10] = 0;
                }
            }
        }
        labels_file.close();
    } else {
        cout << "Error: Failed to open file " << TRAIN_LABELS_FILE_PATH << endl;
        exit(1);
    }
}

void get_image_batch(float (&X)[INPUT_SIZE * BATCH_SIZE], const int offsets[NUM_TRAIN_IMAGES], int index) {
    ifstream images_file(TRAIN_IMAGES_FILE_PATH, ios::in | ios::binary);
    if (images_file.is_open()) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            images_file.seekg(IMAGE_START + 784 * offsets[index + i]);
            for (int j= 0; j < 784; j++) {
                int value = 0;
                images_file.read((char *) &value, 1);
                X[j + i * 784] = value/255.0; // Transform value from range [0, 255] to range [0, 1]
            }
        }
        images_file.close();
    } else {
        cout << "Error: Failed to open file " << TRAIN_IMAGES_FILE_PATH << endl;
        exit(1);
    }
}

void print_batch(float (&X)[INPUT_SIZE * BATCH_SIZE], float (&Y)[OUTPUT_SIZE*BATCH_SIZE]) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Print label
        cout << "The following number is: ";
        for (int label = 0; label < 10; label++) {
            if (Y[label + i * 10] == 1) {
                cout << label << "\n";
                break;
            }
        }
        // Print image
        for (int value = 0; value < 784; value++) {
            if (value != 0 && value % 28 == 0) {
                cout << "\n";
            }
            if (X[value + i * 784] < 0.5) {
                cout << "@.@"; // Represents dark pixel
            } else {
                cout << " . "; // Represents light pixel
            }
        }
        cout << "\n";
    }
}

int get_num_correct(float *gpu_A, float *gpu_Y, int rows, int cols) {
    // Allocate CPU memory
    float *A = new float[rows * cols];
    float *Y = new float[rows * cols];

    // Copy from CPU to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(A, gpu_A, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(Y, gpu_Y, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    int num_correct = 0;

    for (int col = 0; col < cols; col++) {
        int predicted_class;
        float predicted_probability = 0;
        for (int row = 0; row < rows; row++) {
            if (A[col * rows + row] > predicted_probability) {
                predicted_class = row;
                predicted_probability = A[col * rows + row];
            }
        }
        if (Y[col * rows + predicted_class] == 1) {
            num_correct++;
        }
    }

    // Free CPU memory
    delete[] A;
    delete[] Y;

    return num_correct;
}

void gradient_descent(float *gpu_W1, float *gpu_W2, float *gpu_B1, float *gpu_B2) {
    // Number of correct predictions
    int total_correct = 0;

    // Create array of offsets each associated with a label/image pair
    int data_offsets[NUM_TRAIN_IMAGES];

    // Fill with numbers 0 to NUM_TRAIN_IMAGES-1 in increasing order
    iota(data_offsets, data_offsets + NUM_TRAIN_IMAGES, 0);

    // Randomly shuffle array of offsets to randomize image selection in mini-batches
    shuffle(data_offsets, data_offsets + NUM_TRAIN_IMAGES, default_random_engine(SEED));

    float X[INPUT_SIZE * BATCH_SIZE];
    float Y[OUTPUT_SIZE * BATCH_SIZE];

    // Declare GPU memory
    float *gpu_X;
    float *gpu_Y;
    float *gpu_A1;
    float *gpu_A2;

    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_X, INPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A1, L1_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));

    // Perform gradient descent for each mini-batch
    for (int i = 0; i < NUM_TRAIN_IMAGES; i += BATCH_SIZE) {
    
        // Get image and label batch
        get_image_batch(X, data_offsets, i);
        get_label_batch(Y, data_offsets, i);

        // Copy from CPU to GPU
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_X, X, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_Y, Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Forward propagate to get activations A1 and A2
        forward_prop(gpu_X, gpu_A1, gpu_A2, gpu_W1, gpu_B1, gpu_W2, gpu_B2);

        // Add the number of correct predictions from the mini-batch
        int batch_correct = get_num_correct(gpu_A2, gpu_Y, OUTPUT_SIZE, BATCH_SIZE);
        total_correct += batch_correct;

        // Update console
        cout << "BATCH " << (i / BATCH_SIZE) + 1 << "/" << NUM_BATCHES << " COMPLETE" << endl;
        cout << "BATCH ACCURACY: " << 1.0 * batch_correct / BATCH_SIZE << endl;
        cout << "ACCURACY: " << 1.0 * total_correct / (i + BATCH_SIZE) << endl;
        cout << endl;
    }

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(gpu_X));
    CHECK_CUDA_ERROR(cudaFree(gpu_Y));
    CHECK_CUDA_ERROR(cudaFree(gpu_A1));
    CHECK_CUDA_ERROR(cudaFree(gpu_A2));
}


streamoff read(float *X, streamoff position, const string &path, int rows, int cols) {

    // Open file
    ifstream file(path, ios::in | ios::binary);

    if (file.is_open()) {
        // Extract matrix X from offset position
        file.seekg(position);

        double temp = 0;
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                file.read((char *) &temp, sizeof(double));
                X[j*rows + i] = (float) temp;
            }
        }
        // Save the resulting position
        position = file.tellg();

        // Close the file
        file.close();
    } else {
        cout << "Error: Failed to open file WANDB";
        exit(1);
    }

    return position;
}

float W1[L1_SIZE * INPUT_SIZE]{0};
float B1[L1_SIZE]{0};
float W2[OUTPUT_SIZE * L1_SIZE]{0};
float B2[OUTPUT_SIZE]{0};

int main() {
    // Initialize cuBLAS handle
    cublasCreate(&HANDLE);

    // Declare GPU memory
    float *gpu_W1;
    float *gpu_W2;
    float *gpu_B1;
    float *gpu_B2;

    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W1, L1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W2, OUTPUT_SIZE * L1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B1, L1_SIZE * 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B2, OUTPUT_SIZE * 1 * sizeof(float)));

    // print_tensor("B1", gpu_B1, 1, L1_SIZE, 1);
    // print_tensor("B2", gpu_B2, 1, OUTPUT_SIZE, 1);

    streamoff read_position = 0;
    read_position = read(W1, read_position, WEIGHTS_AND_BIASES_FILE_PATH, L1_SIZE, INPUT_SIZE);
    read_position = read(B1, read_position, WEIGHTS_AND_BIASES_FILE_PATH, L1_SIZE, 1);
    read_position = read(W2, read_position, WEIGHTS_AND_BIASES_FILE_PATH, OUTPUT_SIZE, L1_SIZE);
    read(B2, read_position, WEIGHTS_AND_BIASES_FILE_PATH, OUTPUT_SIZE, 1);

    CHECK_CUDA_ERROR(cudaMemcpy(gpu_W1, W1, L1_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B1, B1, L1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_W2, W2, OUTPUT_SIZE * L1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B2, B2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Perform gradient descent
    gradient_descent(gpu_W1, gpu_W2, gpu_B1, gpu_B2);
    
    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(gpu_W1));
    CHECK_CUDA_ERROR(cudaFree(gpu_W2));
    CHECK_CUDA_ERROR(cudaFree(gpu_B1));
    CHECK_CUDA_ERROR(cudaFree(gpu_B2));
    
    // Free HANDLE
    cublasDestroy(HANDLE);

    return 0;
}