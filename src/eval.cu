#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <numeric>
#include <cuda_runtime.h>
#include <iomanip>

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
#define LEARNING_RATE 0.1f
#define NUM_EPOCHS 100

#define INPUT_SIZE 784
#define L1_SIZE 250
#define OUTPUT_SIZE 10

#define BLOCK_SIZE 256

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

float W1[L1_SIZE * INPUT_SIZE]{0};
float B1[L1_SIZE]{0};
float W2[OUTPUT_SIZE * L1_SIZE]{0};
float B2[OUTPUT_SIZE]{0};

float X[INPUT_SIZE * BATCH_SIZE];
float Y[OUTPUT_SIZE * BATCH_SIZE]{0};

float A1[L1_SIZE * BATCH_SIZE]{0};
float A2[OUTPUT_SIZE * BATCH_SIZE]{0};

float *gpu_Y;
float *gpu_W1;
float *gpu_X;
float *gpu_A1;
float* gpu_B1;
float *gpu_W2;
float *gpu_A2;
float* gpu_B2;
float* gpu_NORM;

// Constants for computation
int num_blocks;
const float ALPHA = 1.0f;
const float BETA = 0.0f;

using namespace std;

cublasHandle_t handle;

// CUDA kernel to add the bias vector to the activation matrix
__global__ void add_bias(float* A, float* B, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) {
        int row = idx % rows;
        A[idx] += B[row];
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

// Function to print a matrix
void printMatrix(float* matrix, int rows, int cols) {
    std::cout << std::fixed << std::setprecision(2); // Set precision to 2 decimal places
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[j * rows + i] << " "; // Set width for formatting
        }
        std::cout << std::endl;
    }
}

void forward_prop(float (&A1)[L1_SIZE * BATCH_SIZE], float (&A2)[OUTPUT_SIZE * BATCH_SIZE], float (&X)[INPUT_SIZE * BATCH_SIZE], float (&W1)[L1_SIZE * INPUT_SIZE], float (&B1)[L1_SIZE], float (&W2)[OUTPUT_SIZE * L1_SIZE], float (&B2)[OUTPUT_SIZE]) {

    // Copy data from CPU to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_X, X, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_W1, W1, L1_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B1, B1, L1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_W2, W2, OUTPUT_SIZE * L1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_B2, B2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Perform A1 = W1*X

    // Perform matrix-matrix multiplication using cuBLAS
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, L1_SIZE, BATCH_SIZE, INPUT_SIZE, &ALPHA, gpu_W1, L1_SIZE, gpu_X, INPUT_SIZE, &BETA, gpu_A1, L1_SIZE));

    // Perform A1 = A1 + B1

    // Launch CUDA kernel
    num_blocks = (L1_SIZE * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias<<<num_blocks, BLOCK_SIZE>>>(gpu_A1, gpu_B1, L1_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Perform A1 = ReLU(A1)
   
    // Launch CUDA kernel
    num_blocks = (L1_SIZE * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_ReLU<<<num_blocks, BLOCK_SIZE>>>(gpu_A1, L1_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Perform A2 = W2*A1

    // Perform matrix-matrix multiplication using cuBLAS
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUTPUT_SIZE, BATCH_SIZE, L1_SIZE, &ALPHA, gpu_W2, OUTPUT_SIZE, gpu_A1, L1_SIZE, &BETA, gpu_A2, OUTPUT_SIZE));

    // Perform A2 = A2 + B2

    // Launch CUDA kernel
    num_blocks = (OUTPUT_SIZE * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias<<<num_blocks, BLOCK_SIZE>>>(gpu_A2, gpu_B2, OUTPUT_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Perform A2 = softmax(A2)

    // Initialize gpu_NORM
    CHECK_CUDA_ERROR(cudaMemset(gpu_NORM, 0, 1 * BATCH_SIZE * sizeof(float)));

    // Launch CUDA kernel
    num_blocks = (OUTPUT_SIZE * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_softmax_norm<<<num_blocks, BLOCK_SIZE>>>(gpu_A2, gpu_NORM, OUTPUT_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Launch CUDA kernel
    num_blocks = (OUTPUT_SIZE * BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_softmax<<<num_blocks, BLOCK_SIZE>>>(gpu_A2, gpu_NORM, OUTPUT_SIZE, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the result back to the CPU
    CHECK_CUDA_ERROR(cudaMemcpy(A1, gpu_A1, L1_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(A2, gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
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

void print_batch(float (&A1)[L1_SIZE * BATCH_SIZE], float (&A2)[OUTPUT_SIZE * BATCH_SIZE], float (&X)[INPUT_SIZE * BATCH_SIZE], float (&Y)[OUTPUT_SIZE*BATCH_SIZE]) {
    for (int j = 0; j < BATCH_SIZE; j++) {
        // Print image
        for (int value = 0; value < INPUT_SIZE; value++) {
            if (value != 0 && value % 28 == 0) {
                cout << "\n";
            }
            if (X[value + j * INPUT_SIZE] < 0.5) {
                cout << "@.@"; // Represents dark pixel
            } else {
                cout << " . "; // Represents light pixel
            }
        }
        cout << "\n";
        // Print label
        cout << "The predicted class is: ";
        for (int label = 0; label < OUTPUT_SIZE; label++) {
            if (A2[label + j * OUTPUT_SIZE] > 0.5) {
                cout << label << "\n";
                break;
            }
        }
        cout << "The actual class is: ";
        for (int label = 0; label < OUTPUT_SIZE; label++) {
            if (Y[label + j * OUTPUT_SIZE] == 1) {
                cout << label << "\n";
                break;
            }
        }
        cout << "The predicted vs. actual vector is: " << endl;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            cout << A2[i + j*OUTPUT_SIZE] << " " << Y[i + j*OUTPUT_SIZE] << endl;
        }
        cout << endl;
    }
}

int get_num_correct(float (&A2)[OUTPUT_SIZE*BATCH_SIZE], float (&Y)[OUTPUT_SIZE*BATCH_SIZE]) {
    int num_correct = 0;
    for (int col = 0; col < BATCH_SIZE; col++) {
        int predicted_class;
        float predicted_probability = 0;
        for (int row = 0; row < OUTPUT_SIZE; row++) {
            if (A2[col*OUTPUT_SIZE + row] > predicted_probability) {
                predicted_class = row;
                predicted_probability = A2[col*OUTPUT_SIZE + row];
            }
        }
        if (Y[col*OUTPUT_SIZE + predicted_class] == 1) {
            num_correct++;
        }
    }
    return num_correct;
}

void test_model(float (&W1)[L1_SIZE * INPUT_SIZE], float (&B1)[L1_SIZE], float (&W2)[OUTPUT_SIZE * L1_SIZE], float (&B2)[OUTPUT_SIZE]) {

    // Number of correct predictions
    int total_correct = 0;

    // Create array of offsets each associated with a label/image pair
    int data_offsets[NUM_TRAIN_IMAGES];

    // Fill with numbers 0 to NUM_TRAIN_IMAGES-1 in increasing order
    iota(data_offsets, data_offsets + NUM_TRAIN_IMAGES, 0);

    // Perform gradient descent for each mini-batch
    for (int i = 0; i < NUM_TRAIN_IMAGES; i += BATCH_SIZE) { // can optimize this

        // Reset X and Y
        fill(X, X + INPUT_SIZE * BATCH_SIZE, 0.0f);
        fill(Y, Y + OUTPUT_SIZE * BATCH_SIZE, 0.0f);

        // Get image and label batch
        get_image_batch(X, data_offsets, i);
        get_label_batch(Y, data_offsets, i);

        // Forward propagate to get activations A1 and A2
        forward_prop(A1, A2, X, W1, B1, W2, B2);

        // Debug: Print batch images and labels
        // print_batch(A1, A2, X, Y);

        // Add the number of correct predictions from the mini-batch
        int batch_correct = get_num_correct(A2, Y);
        total_correct += batch_correct;

        // Update console
        cout << "BATCH " << (i / BATCH_SIZE) + 1 << "/" << NUM_BATCHES << " COMPLETE" << endl;
        cout << "BATCH ACCURACY: " << 1.0 * batch_correct / BATCH_SIZE << endl;
        cout << "ACCURACY: " << 1.0 * total_correct / (i + BATCH_SIZE) << endl;
        cout << endl;
    }
}

void gpu_mem_init() {
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W1, L1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_X, INPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A1, L1_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B1, L1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W2, OUTPUT_SIZE * L1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A2, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B2, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_NORM, 1 * BATCH_SIZE * sizeof(float)));
}

void gpu_mem_free() {
    CHECK_CUDA_ERROR(cudaFree(gpu_W1));
    CHECK_CUDA_ERROR(cudaFree(gpu_X));
    CHECK_CUDA_ERROR(cudaFree(gpu_B1));
    CHECK_CUDA_ERROR(cudaFree(gpu_W2));
    CHECK_CUDA_ERROR(cudaFree(gpu_A1));
    CHECK_CUDA_ERROR(cudaFree(gpu_B2));
    CHECK_CUDA_ERROR(cudaFree(gpu_NORM));
    CHECK_CUDA_ERROR(cudaFree(gpu_A2));
    CHECK_CUDA_ERROR(cudaFree(gpu_Y));
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

int main() {
    // Initialize cuBLAS
    cublasCreate(&handle);

    gpu_mem_init();

    streamoff read_position = 0;
    read_position = read(W1, read_position, WEIGHTS_AND_BIASES_FILE_PATH, L1_SIZE, INPUT_SIZE);
    read_position = read(B1, read_position, WEIGHTS_AND_BIASES_FILE_PATH, L1_SIZE, 1);
    read_position = read(W2, read_position, WEIGHTS_AND_BIASES_FILE_PATH, OUTPUT_SIZE, L1_SIZE);
    read(B2, read_position, WEIGHTS_AND_BIASES_FILE_PATH, OUTPUT_SIZE, 1);

    test_model(W1, B1, W2, B2);

    gpu_mem_free();
    
    cublasDestroy(handle);
    return 0;
}