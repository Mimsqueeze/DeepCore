#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <numeric>
#include <cuda_runtime.h>

#define TRAIN_LABELS_FILE_PATH R"(.\data\train-labels.idx1-ubyte)"
#define TRAIN_IMAGES_FILE_PATH R"(.\data\train-images.idx3-ubyte)"
#define TEST_LABELS_FILE_PATH R"(.\data\t10k-labels.idx1-ubyte)"
#define TEST_IMAGES_FILE_PATH R"(.\data\t10k-images.idx3-ubyte)"

#define LABEL_START 8
#define IMAGE_START 16
#define BATCH_SIZE 32

#define NUM_TRAIN_IMAGES 60000
#define NUM_BATCHES (NUM_TRAIN_IMAGES/BATCH_SIZE)
#define NUM_TEST_IMAGES 10000
#define LEARNING_RATE 0.1
#define NUM_EPOCHS 100

#define INPUT_SIZE 784
#define L1_SIZE 128
#define OUTPUT_SIZE 10

#define BLOCK_SIZE 256

#define CHECK_CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
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

float W1[L1_SIZE * INPUT_SIZE]{0};
float B1[L1_SIZE]{0};
float W2[OUTPUT_SIZE * L1_SIZE]{0};
float B2[OUTPUT_SIZE]{0};

float dC_dW1[L1_SIZE * INPUT_SIZE]{0};
float dC_dB1[L1_SIZE]{0};
float dC_dW2[OUTPUT_SIZE * L1_SIZE]{0};
float dC_dB2[OUTPUT_SIZE]{0};

float X[INPUT_SIZE * BATCH_SIZE];
float Y[OUTPUT_SIZE * BATCH_SIZE]{0};

float A1[L1_SIZE * BATCH_SIZE]{0};
float A2[OUTPUT_SIZE * BATCH_SIZE]{0};

using namespace std;

cublasHandle_t handle;

// CUDA kernel to add the bias vector to the activation matrix
__global__ void add_bias(float* A, float* B, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        A[idx] += B[idx % cols];
    }
}

// CUDA kernel to apply ReLU function to the activation matrix
__global__ void apply_ReLU(float* A, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        if (A[idx] < 0) {
            A[idx] = 0;
        }
    }
}

// CUDA kernel to apply softmax function to the activation matrix
__global__ void apply_softmax(float* A, float* NORM, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        A[idx] = exp(A[idx]) / NORM[idx / rows];
    }
}

// CUDA kernel to compute softmax normalizing constants
__global__ void compute_softmax_norm(float* A, float* NORM, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        atomicAdd(&NORM[idx / rows], exp(A[idx]));
    }
}

// CUDA kernel to compute the dC/dA2 Jacobian
__global__ void compute_dC_dA2(float* gpu_dC_dA2, float *gpu_A2, float *gpu_Y, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        int col = idx / rows; // j
        int row = idx % rows; // i
        int col_T = row;
        int row_T = col;
        int rows_T = cols;
        // Using cross-entropy loss function
        gpu_dC_dA2[idx] = -gpu_Y[col_T * rows_T + row_T] / gpu_A2[col_T * rows_T + row_T];
    }
}

// CUDA kernel to compute the dA2/dZ2 Jacobian
__global__ void compute_dA2_dZ2(float* gpu_dA2_dZ2, float *gpu_A2, int rows, int cols, int slices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols * slices;
    if (idx < totalElements) {
        int slice = idx / (rows * cols); // k
        int col = (idx / rows) % cols; // j
        int row = idx % rows; // i
        int a_i = gpu_A2[slice * rows + row];
        int a_j = gpu_A2[slice * rows + col];
        if (row == col) {
            gpu_dA2_dZ2[idx] = a_i * (1 - a_j);
        } else {
            gpu_dA2_dZ2[idx] = a_i * (-a_j);
        }
    }
}

void forward_prop(float (&A1)[L1_SIZE * BATCH_SIZE], float (&A2)[OUTPUT_SIZE * BATCH_SIZE], float (&X)[INPUT_SIZE * BATCH_SIZE], float (&W1)[L1_SIZE * INPUT_SIZE], float (&B1)[L1_SIZE], float (&W2)[OUTPUT_SIZE * L1_SIZE], float (&B2)[OUTPUT_SIZE]) {
    // Constants
    int num_blocks;
    const float alpha = 1.0;
    const float beta = 0.0;

    // Perform A1 = W1*X

    // Device memory allocation
    float *gpu_W1, *gpu_X, *gpu_A1;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W1, L1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_X, INPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A1, L1_SIZE * BATCH_SIZE * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_W1, W1, L1_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_X, X, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Perform matrix-matrix multiplication using cuBLAS
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, L1_SIZE, BATCH_SIZE, INPUT_SIZE, &alpha, gpu_W1, L1_SIZE, gpu_X, INPUT_SIZE, &beta, gpu_A1, L1_SIZE));

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(A1, gpu_A1, L1_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CHECK_CUDA_ERROR(cudaFree(gpu_W1));
    CHECK_CUDA_ERROR(cudaFree(gpu_X));

    // Perform A1 = A1 + B1

    // Device memory allocation
    float* gpu_B1;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B1, L1_SIZE * sizeof(float)));

    // Copy host data to device
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B1, B1, L1_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Launch CUDA kernel
    num_blocks = (L1_SIZE * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias<<<num_blocks, BLOCK_SIZE>>>(gpu_A1, gpu_B1, L1_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(A1, gpu_A1, L1_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(gpu_B1));

    // Perform A1 = ReLU(A1)
   
    // Launch CUDA kernel
    num_blocks = (L1_SIZE * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_ReLU<<<num_blocks, BLOCK_SIZE>>>(gpu_A1, L1_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(A1, gpu_A1, L1_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Perform A2 = W2*A1

    // Device memory allocation
    float *gpu_W2, *gpu_A2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W2, OUTPUT_SIZE * L1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_W2, W2, OUTPUT_SIZE * L1_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Perform matrix-matrix multiplication using cuBLAS
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUTPUT_SIZE, BATCH_SIZE, L1_SIZE, &alpha, gpu_W2, OUTPUT_SIZE, gpu_A1, L1_SIZE, &beta, gpu_A2, OUTPUT_SIZE));

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(A2, gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CHECK_CUDA_ERROR(cudaFree(gpu_W2));
    CHECK_CUDA_ERROR(cudaFree(gpu_A1));

    // Perform A2 = A2 + B2

    // Device memory allocation
    float* gpu_B2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B2, OUTPUT_SIZE * sizeof(float)));

    // Copy host data to device
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B2, B2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Launch CUDA kernel
    num_blocks = (OUTPUT_SIZE * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias<<<num_blocks, BLOCK_SIZE>>>(gpu_A2, gpu_B2, OUTPUT_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(A2, gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(gpu_B2));

    // Perform A2 = softmax(A2)

    // Allocate memory on device
    float* gpu_NORM;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_NORM, 1 * BATCH_SIZE * sizeof(float)));

    // Launch CUDA kernel
    num_blocks = (1 * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_softmax_norm<<<num_blocks, BLOCK_SIZE>>>(gpu_A2, gpu_NORM, OUTPUT_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Launch CUDA kernel
    num_blocks = (1 * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_softmax<<<num_blocks, BLOCK_SIZE>>>(gpu_A2, gpu_NORM, OUTPUT_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(A2, gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(gpu_NORM));
    CHECK_CUDA_ERROR(cudaFree(gpu_A2));
}

void back_prop(float (&dC_dW1)[L1_SIZE * INPUT_SIZE], float (&dC_dB1)[L1_SIZE], float (&dC_dW2)[OUTPUT_SIZE * L1_SIZE], float (&dC_dB2)[OUTPUT_SIZE], float (&X)[INPUT_SIZE * BATCH_SIZE], float (&Y)[OUTPUT_SIZE * BATCH_SIZE], float (&A1)[L1_SIZE * BATCH_SIZE], float (&A2)[OUTPUT_SIZE * BATCH_SIZE], float (&W1)[L1_SIZE * INPUT_SIZE], float (&W2)[OUTPUT_SIZE * L1_SIZE]) {
    // Constants
    int num_blocks;
    const float alpha = 1.0;
    const float beta = 0.0;

    // Compute layer 2 local error

    // Allocate memory on device
    float *gpu_dC_dA2, *gpu_dA2_dZ2;
    float *gpu_dC_dZ2; // Layer 2 local error
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dA2, BATCH_SIZE * 1 * OUTPUT_SIZE * sizeof(float))); // nx1x10
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dA2_dZ2, BATCH_SIZE * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float))); // nx10x10
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dZ2, BATCH_SIZE * 1 * OUTPUT_SIZE * sizeof(float))); // nx1x10

    // Compute dC/dA2: nx1x10

    // Allocate memory on device
    float *gpu_Y, *gpu_A2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_Y, Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_A2, A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch CUDA kernel
    num_blocks = (BATCH_SIZE * 1 * OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_dC_dA2<<<num_blocks, BLOCK_SIZE>>>(gpu_dC_dA2, gpu_A2, gpu_Y, BATCH_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(gpu_Y));

    // Compute dA2/dZ2 nx10x10

    // Launch CUDA kernel
    num_blocks = (BATCH_SIZE * OUTPUT_SIZE * OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_dA2_dZ2<<<num_blocks, BLOCK_SIZE>>>(gpu_dA2_dZ2, gpu_A2, OUTPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Compute layer 1 local error
    float *gpu_dZ2_dA1, *gpu_dA1_dZ1;
    float *gpu_dC_dZ1; // Layer 1 local error

    cout << "PASS";
}

void update_params() {

}

void get_label_batch(float (&Y)[OUTPUT_SIZE * BATCH_SIZE], const int offsets[NUM_TRAIN_IMAGES], int index) {
    ifstream labels_file(TRAIN_LABELS_FILE_PATH, ios::in | ios::binary);
    if (labels_file.is_open()) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            labels_file.seekg(LABEL_START + offsets[index + i]);
            int label;
            labels_file.read((char *) &label, 1);
            Y[label + i * 10] = 1;
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

void gradient_descent(float (&W1)[L1_SIZE * INPUT_SIZE], float (&B1)[L1_SIZE], float (&W2)[OUTPUT_SIZE * L1_SIZE], float (&B2)[OUTPUT_SIZE]) {

    // Number of correct predictions
    int correct = 0;

    // Create array of offsets each associated with a label/image pair
    int data_offsets[NUM_TRAIN_IMAGES];

    // Fill with numbers 0 to NUM_TRAIN_IMAGES-1 in increasing order
    iota(data_offsets, data_offsets + NUM_TRAIN_IMAGES, 0);

    // Randomly shuffle array of offsets to randomize image selection in mini-batches
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(data_offsets, data_offsets + NUM_TRAIN_IMAGES, default_random_engine(seed));

    // Perform gradient descent for each mini-batch
    for (int i = 0; i < NUM_TRAIN_IMAGES; i += BATCH_SIZE) {

        // Get image and label batch
        get_image_batch(X, data_offsets, i);
        get_label_batch(Y, data_offsets, i);

        // Debug: Print batch images and labels
        // print_batch(X, Y);

        // Forward propagate to get activations A1 and A2
        forward_prop(A1, A2, X, W1, B1, W2, B2);

        // Debug: Print activations of the last layer
        // for (int i = 0; i < OUTPUT_SIZE*BATCH_SIZE; i++) {
        //     cout << A2[i] << " ";
        // }
        // for (int i = 0; i < OUTPUT_SIZE*BATCH_SIZE; i++) {
        //     cout << Y[i] << " ";
        // }

        // Back propagate to get dC/W1, dC/dB1, dC/dW2, dC/dB2
        back_prop(dC_dW1, dC_dB1, dC_dW2, dC_dB2, X, Y, A1, A2, W1, W2);

        // // Add derivatives from mini-batch, in other words add the "nudges"
        // dW1 += bp.dW1;
        // dB1 += bp.dB1;
        // dW2 += bp.dW2;
        // dB2 += bp.dB2;

        // // Add the number of correct predictions from the mini-batch
        // correct += get_num_correct(get_predictions(fp.A2, BATCH_SIZE), Y, BATCH_SIZE);
    }

    // Divide each derivative or "nudge" value by the number of batches to find the average among all batches
    // dW1 /= NUM_BATCHES;
    // dB1 /= NUM_BATCHES;
    // dW2 /= NUM_BATCHES;
    // dB2 /= NUM_BATCHES;

    // // Update the parameters W1, B1, W2, and B2 with the "nudges"
    // update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate);

    // return correct;
}

// Can GPU optimize
void he_init(float *weights, int m, int n) {
    random_device rd; // Random device for seeding
    mt19937 gen(rd()); // Mersenne Twister generator
    normal_distribution<> d(0, sqrt(2.0 / n)); // He normal distribution
    for (int i = 0; i < m * n; i++) {
        weights[i] = d(gen);
    }
}

int main() {

    // Initialize cuBLAS
    cublasCreate(&handle);

    // Initialize weights with He initialization method
    he_init(W1, L1_SIZE, INPUT_SIZE);
    he_init(W2, OUTPUT_SIZE, L1_SIZE);

    gradient_descent(W1, B1, W2, B2);
    cublasDestroy(handle);
    return 0;
}