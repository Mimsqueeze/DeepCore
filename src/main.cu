#include <fstream>
#include <iostream>
#include <string>
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
#define BATCH_SIZE 50

#define NUM_TRAIN_IMAGES 60000
#define NUM_BATCHES (NUM_TRAIN_IMAGES/BATCH_SIZE)
#define NUM_TEST_IMAGES 10000
#define LEARNING_RATE 0.1f
#define NUM_EPOCHS 100

#define INPUT_SIZE 784
#define L1_SIZE 600
#define OUTPUT_SIZE 10

#define BLOCK_SIZE 256

#define SAVE_WEIGHTS_AND_BIASES true
#define PRINT_BATCH_AND_PREDICTIONS false

#define WEIGHTS_AND_BIASES_DIR R"(.\models\)"

#define CHECK_CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS_ERROR(err) { \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << _cudaGetErrorEnum(err) << " at line " << __LINE__ << std::endl; \
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

void multiply_tensor_T(float *gpu_result, float *gpu_A, float *gpu_B, int slices, int m, int n, int k) {
    // Allocate arrays for the pointers
    float** A_array = new float*[slices];
    float** B_array = new float*[slices];
    float** result_array = new float*[slices];

    for (int i = 0; i < slices; ++i) {
        A_array[i] = gpu_A + i * m * k;
        B_array[i] = gpu_B + i * k * n;
        result_array[i] = gpu_result + i * m * n;
    }

    float** gpu_A_array;
    float** gpu_B_array;
    float** gpu_C_array;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A_array, slices * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B_array, slices * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_C_array, slices * sizeof(float*)));

    CHECK_CUDA_ERROR(cudaMemcpy(gpu_A_array, A_array, slices * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B_array, B_array, slices * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_C_array, result_array, slices * sizeof(float*), cudaMemcpyHostToDevice));

    // Transpose A and B
    CHECK_CUBLAS_ERROR(cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                                          &ALPHA, (const float**)gpu_A_array, k,
                                          (const float**)gpu_B_array, n,
                                          &BETA, gpu_C_array, m, slices));

    // Free dynamically allocated arrays
    delete[] A_array;
    delete[] B_array;
    delete[] result_array;

    CHECK_CUDA_ERROR(cudaFree(gpu_A_array));
    CHECK_CUDA_ERROR(cudaFree(gpu_B_array));
    CHECK_CUDA_ERROR(cudaFree(gpu_C_array));
}

void multiply_tensor(float *gpu_result, float *gpu_A, float *gpu_B, int slices, int m, int n, int k) {
    // Allocate arrays for the pointers
    float** A_array = new float*[slices]{0};
    float** B_array = new float*[slices]{0};
    float** result_array = new float*[slices]{0};

    for (int i = 0; i < slices; ++i) {
        A_array[i] = gpu_A + i * m * k;
        B_array[i] = gpu_B + i * k * n;
        result_array[i] = gpu_result + i * m * n;
    }

    float** gpu_A_array;
    float** gpu_B_array;
    float** gpu_C_array;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A_array, slices * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B_array, slices * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_C_array, slices * sizeof(float*)));

    CHECK_CUDA_ERROR(cudaMemcpy(gpu_A_array, A_array, slices * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B_array, B_array, slices * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_C_array, result_array, slices * sizeof(float*), cudaMemcpyHostToDevice));

    // Perform batched matrix multiplication
    CHECK_CUBLAS_ERROR(cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                          &ALPHA, (const float**)gpu_A_array, m,
                                          (const float**)gpu_B_array, k,
                                          &BETA, gpu_C_array, m, slices));

    // Free dynamically allocated arrays
    delete[] A_array;
    delete[] B_array;
    delete[] result_array;

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

// CUDA kernel to compute the dC/dA2 Jacobian
__global__ void compute_dC_dA2_kernel(float *gpu_dC_dA2, float *gpu_A2, float *gpu_Y, int slices, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = slices * rows * cols;
    if (idx < totalElements) {
        int slice = idx / (rows * cols); // k
        int col = (idx / rows) % cols; // j

        // Using cross-entropy loss function
        gpu_dC_dA2[idx] = -gpu_Y[slice * cols + col] / gpu_A2[slice * cols + col];

        // Using mean-square error loss function
        // gpu_dC_dA2[idx] = gpu_A2[slice * cols + col] - gpu_Y[slice * cols + col];
    }
}

void compute_dC_dA2(float *gpu_dC_dA2, float *gpu_A2, float *gpu_Y, int slices, int rows, int cols) {
    int num_blocks = (slices * rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_dC_dA2_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_dC_dA2, gpu_A2, gpu_Y, slices, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void compute_dA2_dZ2_kernel(float* gpu_dA2_dZ2, float *gpu_A2, int slices, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = slices * rows * cols;
    if (idx < totalElements) {
        int slice = idx / (rows * cols); // k
        int col = (idx / rows) % cols; // j
        int row = idx % rows; // i
        float a_i = gpu_A2[slice * rows + row];
        float a_j = gpu_A2[slice * rows + col];
        if (row == col) {
            gpu_dA2_dZ2[idx] = a_i * (1.0f - a_j);
        } else {
            gpu_dA2_dZ2[idx] = a_i * (-a_j);
        }
    }
}

void compute_dA2_dZ2(float* gpu_dA2_dZ2, float *gpu_A2, int slices, int rows, int cols) {
    int num_blocks = (slices * rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_dA2_dZ2_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_dA2_dZ2, gpu_A2, slices, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void compute_dZ2_dA1(float* gpu_dZ2_dA1, float *gpu_W2, int slices, int rows, int cols) {
    for (int slice = 0; slice < slices; slice++) {
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_dZ2_dA1 + slice * rows * cols, gpu_W2, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

// CUDA kernel to compute the dA1/dZ1 Jacobian
__global__ void compute_dA1_dZ1_kernel(float* gpu_dA1_dZ1, float *gpu_A1, int slices, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = slices * rows * cols;
    if (idx < totalElements) {
        int slice = idx / (rows * cols); // k
        int col = (idx / rows) % cols; // j
        int row = idx % rows; // i
        if (col == row && gpu_A1[slice * rows + row] > 0) {
            gpu_dA1_dZ1[idx] = 1;
        } else {
            gpu_dA1_dZ1 = 0;
        }
    }
}

void compute_dA1_dZ1(float* gpu_dA1_dZ1, float *gpu_A1, int slices, int rows, int cols) {
    int num_blocks = (slices * rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_dA1_dZ1_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_dA1_dZ1, gpu_A1, BATCH_SIZE, L1_SIZE, L1_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// CUDA kernel to update params in gpu_P
__global__ void update_params_kernel(float *gpu_P, float *gpu_dP, float learning_rate, int rows, int cols, int slices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) { // let's just use a loop here
        float avg_deriv = 0;
        for (int slice = 0; slice < slices; slice++) {
            avg_deriv += gpu_dP[idx + (slice * rows * cols)];
        }
        avg_deriv /= slices;
        gpu_P[idx] = gpu_P[idx] - learning_rate * avg_deriv;
    }
}

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
    multiply_tensor(gpu_A1, gpu_W1, gpu_X, 1, L1_SIZE, BATCH_SIZE, INPUT_SIZE);

    // Perform A1 = A1 + B1
    add_bias(gpu_A1, gpu_B1, L1_SIZE, BATCH_SIZE);

    // Perform A1 = ReLU(A1)
    apply_ReLU(gpu_A1, L1_SIZE, BATCH_SIZE);

    // Perform A2 = W2*A1
    multiply_tensor(gpu_A2, gpu_W2, gpu_A1, 1, OUTPUT_SIZE, BATCH_SIZE, L1_SIZE);

    // Perform A2 = A2 + B2
    add_bias(gpu_A2, gpu_B2, OUTPUT_SIZE, BATCH_SIZE);

    // Perform A2 = softmax(A2)
    apply_softmax(gpu_A2, OUTPUT_SIZE, BATCH_SIZE);
}

void back_prop(float *gpu_dC_dW1, float *gpu_dC_dB1, float *gpu_dC_dW2, float *gpu_dC_dB2, 
               float *gpu_X, float *gpu_Y, float *gpu_A1, float *gpu_A2, float *gpu_W1, float *gpu_W2) {

    // Declare GPU memory
    float *gpu_dC_dA2;
    float *gpu_dA2_dZ2;
    float *gpu_dC_dZ2; // layer 2 local error

    float *gpu_dZ2_dA1;
    float *gpu_dA1_dZ1;
    float *gpu_dZ2_dZ1; // layer 1 local error

    float *gpu_dC_dZ1; // layer 1 and level 2 accumulated error

    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dA2, BATCH_SIZE * 1 * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dA2_dZ2, BATCH_SIZE * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dZ2, BATCH_SIZE * 1 * OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dZ2_dA1, BATCH_SIZE * OUTPUT_SIZE * L1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dA1_dZ1, BATCH_SIZE * L1_SIZE * L1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dZ2_dZ1, BATCH_SIZE * OUTPUT_SIZE * L1_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dZ1, BATCH_SIZE * 1 * L1_SIZE * sizeof(float)));

    // print_tensor("Y", gpu_Y, 1, OUTPUT_SIZE, BATCH_SIZE);
    // print_tensor("A2", gpu_A2, 1, OUTPUT_SIZE, BATCH_SIZE);

    // Compute dC/dA2 (nx1x10)
    compute_dC_dA2(gpu_dC_dA2, gpu_A2, gpu_Y, BATCH_SIZE, 1, OUTPUT_SIZE);

    // print_tensor("dC/dA2", gpu_dC_dA2, BATCH_SIZE, 1, OUTPUT_SIZE);

    // Compute dA2/dZ2 (nx10x10)
    compute_dA2_dZ2(gpu_dA2_dZ2, gpu_A2, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE);

    // print_tensor("dA2/dZ2", gpu_dA2_dZ2, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE);

    // Compute layer 2 local error dC/dZ2 (nx1x10)
    // dC/dZ2 = dC/dA2 * dA2/dZ2
    multiply_tensor(gpu_dC_dZ2, gpu_dC_dA2, gpu_dA2_dZ2, BATCH_SIZE, 1, OUTPUT_SIZE, OUTPUT_SIZE);

    // print_tensor("dC/dZ2", gpu_dC_dZ2, BATCH_SIZE, 1, OUTPUT_SIZE);

    // Compute dZ2/dA1 (nx10x100)
    // It's just n times the weight matrix!
    compute_dZ2_dA1(gpu_dZ2_dA1, gpu_W2, BATCH_SIZE, OUTPUT_SIZE, L1_SIZE);

    // print_tensor("W2", gpu_W2, 1, OUTPUT_SIZE, L1_SIZE);

    // print_tensor("dZ2/dA1", gpu_dZ2_dA1, BATCH_SIZE, OUTPUT_SIZE, L1_SIZE);

    // Compute dA1/dZ1 (nx100x100)
    compute_dA1_dZ1(gpu_dA1_dZ1, gpu_A1, BATCH_SIZE, L1_SIZE, L1_SIZE);

    // print_tensor("A1", gpu_A1, 1, L1_SIZE, BATCH_SIZE);

    // print_tensor("dA1/dZ1", gpu_dA1_dZ1, BATCH_SIZE, L1_SIZE, L1_SIZE);

    // Compute layer 1 local error dZ2/dZ1 (nx10x100)
    // dZ2/dZ1 = dZ2/dA1 * dA1/dZ1
    multiply_tensor(gpu_dZ2_dZ1, gpu_dZ2_dA1, gpu_dA1_dZ1, BATCH_SIZE, OUTPUT_SIZE, L1_SIZE, L1_SIZE);
    // print_tensor("dZ2/dZ1", gpu_dZ2_dZ1, BATCH_SIZE, OUTPUT_SIZE, L1_SIZE);

    // Compute layer 1 and level 2 accumulated error dC/dZ1 (nx1x100)
    multiply_tensor(gpu_dC_dZ1, gpu_dC_dZ2, gpu_dZ2_dZ1, BATCH_SIZE, 1, L1_SIZE, OUTPUT_SIZE);
    // print_tensor("dC/dZ1", gpu_dC_dZ1, BATCH_SIZE, 1, L1_SIZE);

    // Compute dC/dW2 (nx10x100)
    // Use shortcut: dC/dW2 = transpose(dC/dZ2) * transpose(A1)
    multiply_tensor_T(gpu_dC_dW2, gpu_dC_dZ2, gpu_A1, BATCH_SIZE, OUTPUT_SIZE, L1_SIZE, 1);
    // print_tensor("dC/dW2", gpu_dC_dW2, BATCH_SIZE, OUTPUT_SIZE, L1_SIZE);

    // Compute dC/dB2 (nx1x10)
    // Use shortcut: dC/dB2 = dC/dZ2
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_dC_dB2, gpu_dC_dZ2, BATCH_SIZE * 1 * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    // print_tensor("dC/dB2", gpu_dC_dB2, BATCH_SIZE, 1, OUTPUT_SIZE);

    // Compute dC/dW1 (nx100x784)
    // Use shortcut: dC/dW1 = transpose(dC/dZ1) * transpose(X)
    multiply_tensor_T(gpu_dC_dW1, gpu_dC_dZ1, gpu_X, BATCH_SIZE, L1_SIZE, INPUT_SIZE, 1);
    // print_tensor("dC/dW1", gpu_dC_dW1, BATCH_SIZE, L1_SIZE, INPUT_SIZE);

    // Compute dC/dB1 (nx1x100)
    // Use shortcut: dC/dB1 = dC/dZ1
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_dC_dB1, gpu_dC_dZ1, BATCH_SIZE * 1 * L1_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    // print_tensor("dC/dB1", gpu_dC_dB1, BATCH_SIZE, 1, L1_SIZE);

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(gpu_dC_dA2));
    CHECK_CUDA_ERROR(cudaFree(gpu_dA2_dZ2));
    CHECK_CUDA_ERROR(cudaFree(gpu_dC_dZ2));

    CHECK_CUDA_ERROR(cudaFree(gpu_dZ2_dA1));
    CHECK_CUDA_ERROR(cudaFree(gpu_dA1_dZ1));
    CHECK_CUDA_ERROR(cudaFree(gpu_dZ2_dZ1));

    CHECK_CUDA_ERROR(cudaFree(gpu_dC_dZ1));
}

void update_params(float *gpu_W1, float *gpu_B1, float *gpu_W2, float *gpu_B2, 
                   float *gpu_dC_dW1, float *gpu_dC_dB1, float *gpu_dC_dW2, float *gpu_dC_dB2) {
    // Launch CUDA kernel
    int num_blocks = (L1_SIZE * INPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_params_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_W1, gpu_dC_dW1, LEARNING_RATE, L1_SIZE, INPUT_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Launch CUDA kernel
    num_blocks = (L1_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_params_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_B1, gpu_dC_dB1, LEARNING_RATE, L1_SIZE, 1, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Launch CUDA kernel
    num_blocks = (OUTPUT_SIZE * L1_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_params_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_W2, gpu_dC_dW2, LEARNING_RATE, OUTPUT_SIZE, L1_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Launch CUDA kernel
    num_blocks = (OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_params_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_B2, gpu_dC_dB2, LEARNING_RATE, OUTPUT_SIZE, 1, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void get_label_batch(float (&Y)[OUTPUT_SIZE * BATCH_SIZE], const int *offsets, int index, string path) {
    ifstream labels_file(path, ios::in | ios::binary);
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
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }
}

void get_image_batch(float (&X)[INPUT_SIZE * BATCH_SIZE], const int *offsets, int index, string path) {
    ifstream images_file(path, ios::in | ios::binary);
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
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
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

void print_batch_and_predictions(float *gpu_X, float *gpu_Y, float *gpu_A2) {
    // CPU matrices
    float X[INPUT_SIZE * BATCH_SIZE];
    float Y[OUTPUT_SIZE * BATCH_SIZE];
    float A2[OUTPUT_SIZE * BATCH_SIZE];

    // Copy from GPU to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(X, gpu_X, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(Y, gpu_Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(A2, gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Print image
        for (int value = 0; value < 784; value++) {
            if (value != 0 && value % 28 == 0) {
                cout << endl;
            }
            if (X[value + i * 784] < 0.5) {
                cout << "@.@"; // Represents dark pixel
            } else {
                cout << " . "; // Represents light pixel
            }
        }
        cout << endl;
        // Print predicted and actual label
        cout << "PREDICTED LABEL: ";
        float pred_prob = 0;
        int pred_label = -1;
        for (int label = 0; label < 10; label++) {
            if (A2[label + i * 10] > pred_prob) {
                pred_label = label;
                pred_prob = A2[label + i * 10];
            }
        }
        cout << pred_label;
        cout << " - ACTUAL LABEL: ";
        for (int label = 0; label < 10; label++) {
            if (Y[label + i * 10] == 1) {
                cout << label << endl;
                break;
            }
        }
        cout << endl;
    }
}

int gradient_descent(float *gpu_W1, float *gpu_W2, float *gpu_B1, float *gpu_B2) {
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
    float *gpu_dC_dW1;
    float *gpu_dC_dW2;
    float *gpu_dC_dB1;
    float *gpu_dC_dB2;

    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_X, INPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A1, L1_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dW1, BATCH_SIZE * L1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dW2, BATCH_SIZE * OUTPUT_SIZE * L1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dB1, BATCH_SIZE * L1_SIZE * 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dB2, BATCH_SIZE * OUTPUT_SIZE * 1 * sizeof(float)));

    // Perform gradient descent for each mini-batch
    for (int i = 0; i < NUM_TRAIN_IMAGES; i += BATCH_SIZE) {
    
        // Get image and label batch
        get_image_batch(X, data_offsets, i, TRAIN_IMAGES_FILE_PATH);
        get_label_batch(Y, data_offsets, i, TRAIN_LABELS_FILE_PATH);

        // Copy from CPU to GPU
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_X, X, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_Y, Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Forward propagate to get activations A1 and A2
        forward_prop(gpu_X, gpu_A1, gpu_A2, gpu_W1, gpu_B1, gpu_W2, gpu_B2);

        // Print batch and preductions
        if (PRINT_BATCH_AND_PREDICTIONS) {
            print_batch_and_predictions(gpu_X, gpu_Y, gpu_A2);
        }

        // Add the number of correct predictions from the mini-batch
        int batch_correct = get_num_correct(gpu_A2, gpu_Y, OUTPUT_SIZE, BATCH_SIZE);
        total_correct += batch_correct;

        // Back propagate to get dC/W1, dC/dB1, dC/dW2, dC/dB2
        back_prop(gpu_dC_dW1, gpu_dC_dB1, gpu_dC_dW2, gpu_dC_dB2, gpu_X, gpu_Y, gpu_A1, gpu_A2, gpu_W1, gpu_W2);

        // Update parameters
        update_params(gpu_W1, gpu_B1, gpu_W2, gpu_B2, gpu_dC_dW1, gpu_dC_dB1, gpu_dC_dW2, gpu_dC_dB2);

        // Update console
        cout << "\r";
        cout << "BATCH " << (i / BATCH_SIZE) + 1 << "/" << NUM_BATCHES << " ";
        cout << "[";
        float percentage_completion = (((float) i / BATCH_SIZE) + 1) / NUM_BATCHES;
        bool arrow = true;
        for (int j = 1; j <= 32; j++) {
            if (percentage_completion >= (float) j / 32) {
                cout << "=";
            } else {
                if (arrow) {
                    cout << ">";
                    arrow = false;
                } else {
                    cout << ".";
                }
            }
        }

        cout << "] - BATCH ACCURACY: ";
        printf("%.3f", (float) batch_correct / BATCH_SIZE);
        cout << " - TOTAL ACCURACY: ";
        printf("%.3f", (float) total_correct / (i + BATCH_SIZE));
        cout << flush;;
    }
    cout << endl;

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(gpu_X));
    CHECK_CUDA_ERROR(cudaFree(gpu_Y));
    CHECK_CUDA_ERROR(cudaFree(gpu_A1));
    CHECK_CUDA_ERROR(cudaFree(gpu_A2));
    CHECK_CUDA_ERROR(cudaFree(gpu_dC_dW1));
    CHECK_CUDA_ERROR(cudaFree(gpu_dC_dW2));
    CHECK_CUDA_ERROR(cudaFree(gpu_dC_dB1));
    CHECK_CUDA_ERROR(cudaFree(gpu_dC_dB2));

    return total_correct;
}

__global__ void he_init_kernel(float *gpu_W, int m, int n, unsigned SEED) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        curandState state;
        curand_init(SEED, idx, 0, &state);
        gpu_W[idx] = curand_normal(&state) * sqrtf(2.0f / m);
    }
}

void he_init(float *gpu_W, int m, int n) {
    int num_blocks = (m * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    he_init_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_W, m, n, SEED);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void xavier_init_kernel(float *gpu_W, int m, int n, unsigned SEED) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        curandState state;
        curand_init(SEED, idx, 0, &state);
        gpu_W[idx] = curand_normal(&state) * sqrtf(1.0f / (m + n));
    }
}

void xavier_init(float *gpu_W, int m, int n) {
    int num_blocks = (m * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    he_init_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_W, m, n, SEED);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

streamoff save(float *gpu_X, int rows, int cols, streamoff position, const string &path) {
    // Declare file
    ofstream file;

    // Open file
    if (position == 0) {
        file = ofstream(path, ios::out | ios::binary);
    } else {
        file = ofstream(path, ios::app | ios::binary);
    }

    // Allocate memory on CPU
    float *X = new float[rows * cols];

    // Copy memory from GPU to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(X, gpu_X, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));


    if (file.is_open()) {
        // Save matrix X into the offset position
        file.seekp(position);
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                file.write((char *) &X[j * rows + i], sizeof(float));
            }
        }
        // Save the resulting position
        position = file.tellp();
    

        // Close the file
        file.close();
    } else {
        cerr << "ERROR: FAILED TO OPEN FILE " << path;
        exit(1);
    }


    // Free CPU memory
    delete[] X;

    return position;
}

int main() {
    // Initialize cuBLAS handle
    cublasCreate(&HANDLE);

    // Declare GPU memory
    float *gpu_W1;
    float *gpu_B1;
    float *gpu_W2;
    float *gpu_B2;
    float *gpu_test_X;
    float *gpu_test_Y;
    float *gpu_test_A1;
    float *gpu_test_A2;

    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W1, L1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B1, L1_SIZE * 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W2, OUTPUT_SIZE * L1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B2, OUTPUT_SIZE * 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_test_X, INPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_test_Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_test_A1, L1_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_test_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));

    // Initialize weights with He initialization method
    he_init(gpu_W1, L1_SIZE, INPUT_SIZE);
    xavier_init(gpu_W2, OUTPUT_SIZE, L1_SIZE);

    // print_tensor("W1", gpu_W1, 1, L1_SIZE, INPUT_SIZE);
    // print_tensor("W2", gpu_W2, 1, OUTPUT_SIZE, L1_SIZE);

    // Initialize biases to 0
    CHECK_CUDA_ERROR(cudaMemset(gpu_B1, 0, L1_SIZE * 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(gpu_B2, 0, OUTPUT_SIZE * 1 * sizeof(float)));

    // print_tensor("B1", gpu_B1, 1, L1_SIZE, 1);
    // print_tensor("B2", gpu_B2, 1, OUTPUT_SIZE, 1);

    float test_X[INPUT_SIZE * BATCH_SIZE];
    float test_Y[OUTPUT_SIZE * BATCH_SIZE];

    // Create array of offsets each associated with a label/image pair
    int data_offsets[NUM_TEST_IMAGES];

    // Fill with numbers 0 to NUM_TRAIN_IMAGES-1 in increasing order
    iota(data_offsets, data_offsets + NUM_TEST_IMAGES, 0);

    // Perform gradient descent

    // For each epoch, perform gradient descent and update weights and biases
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        cout << "EPOCH " << epoch << "/" << NUM_EPOCHS << endl;

        // Get start time
        auto start = chrono::high_resolution_clock::now();

        // Store number of correct predictions
        int train_correct = gradient_descent(gpu_W1, gpu_W2, gpu_B1, gpu_B2);

        // Get end time
        auto end = chrono::high_resolution_clock::now();

        // Calculate duration of time passed
        double duration = (double) chrono::duration_cast<chrono::microseconds>(end - start).count()/1000000.0;

        // Calculate remaining time
        int seconds = (int) duration*(NUM_EPOCHS - epoch);
        int minutes= seconds/60;
        int hours= minutes/60;
        minutes %= 60;
        seconds %= 60;

        // Find performance on testing set
        int test_correct = 0;
        for (int i = 0; i < NUM_TEST_IMAGES; i += BATCH_SIZE) {
        
            // Get image and label batch
            get_image_batch(test_X, data_offsets, i, TEST_IMAGES_FILE_PATH);
            get_label_batch(test_Y, data_offsets, i, TEST_LABELS_FILE_PATH);

            // Copy from CPU to GPU
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_test_X, test_X, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_test_Y, test_Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            // Forward propagate to get activations A1 and A2
            forward_prop(gpu_test_X, gpu_test_A1, gpu_test_A2, gpu_W1, gpu_B1, gpu_W2, gpu_B2);

            // Add the number of correct predictions from the mini-batch
            int batch_correct = get_num_correct(gpu_test_A2, gpu_test_Y, OUTPUT_SIZE, BATCH_SIZE);
            test_correct += batch_correct;
        }

        // Print the results of the epoch
        cout << "TRAIN ACCURACY: " << train_correct << "/" << NUM_TRAIN_IMAGES;
        printf(" (%.2f%%)", 100.0 * train_correct / NUM_TRAIN_IMAGES);
        cout << " - TEST ACCURACY: " << test_correct << "/" << NUM_TEST_IMAGES;
        printf(" (%.2f%%)", 100.0 * test_correct / NUM_TEST_IMAGES);
        cout << " - TIME ELAPSED: ";
        printf("%.2fs", duration);
        cout << " - ETA: ";
        printf("%02d:%02d:%02d\n", hours, minutes, seconds);
        cout << endl;
    }

    cout << "FINISHED TRAINING." << endl;

    if (SAVE_WEIGHTS_AND_BIASES) {
        cout << "SAVING WEIGHTS AND BIASES TO FILE...\n";
        streamoff write_position = 0;
        string path = WEIGHTS_AND_BIASES_DIR + to_string(INPUT_SIZE) + "-" + to_string(L1_SIZE) + "-" + to_string(OUTPUT_SIZE) + ".bin";
        write_position = save(gpu_W1, L1_SIZE, INPUT_SIZE, write_position, path);
        write_position = save(gpu_B1, L1_SIZE, 1, write_position, path);
        write_position = save(gpu_W2, OUTPUT_SIZE, L1_SIZE, write_position, path);
        save(gpu_B2, OUTPUT_SIZE, 1, write_position, path);
    }

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(gpu_W1));
    CHECK_CUDA_ERROR(cudaFree(gpu_B1));
    CHECK_CUDA_ERROR(cudaFree(gpu_W2));
    CHECK_CUDA_ERROR(cudaFree(gpu_B2));
    CHECK_CUDA_ERROR(cudaFree(gpu_test_X));
    CHECK_CUDA_ERROR(cudaFree(gpu_test_Y));
    CHECK_CUDA_ERROR(cudaFree(gpu_test_A1));
    CHECK_CUDA_ERROR(cudaFree(gpu_test_A2));

    // Free HANDLE
    cublasDestroy(HANDLE);

    return 0;
}