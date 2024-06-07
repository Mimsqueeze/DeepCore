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
#define BATCH_SIZE 32

#define NUM_TRAIN_IMAGES 60000
#define NUM_BATCHES (NUM_TRAIN_IMAGES/BATCH_SIZE)
#define NUM_TEST_IMAGES 10000
#define LEARNING_RATE 0.1f
#define NUM_EPOCHS 100

#define INPUT_SIZE 784
#define L1_SIZE 15
#define OUTPUT_SIZE 10

#define BLOCK_SIZE 256
#define DEBUG_MODE true

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

void tensor_multiply_T(const float* h_A, const float* h_B, float* h_C, int batchSize, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, batchSize * m * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, batchSize * k * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, batchSize * m * n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, batchSize * m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, batchSize * k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate arrays for the pointers
    float** Aarray = new float*[batchSize];
    float** Barray = new float*[batchSize];
    float** Carray = new float*[batchSize];

    for (int i = 0; i < batchSize; ++i) {
        Aarray[i] = d_A + i * m * k;
        Barray[i] = d_B + i * k * n;
        Carray[i] = d_C + i * m * n;
    }

    float** d_Aarray;
    float** d_Barray;
    float** d_Carray;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Aarray, batchSize * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Barray, batchSize * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Carray, batchSize * sizeof(float*)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_Aarray, Aarray, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Barray, Barray, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Carray, Carray, batchSize * sizeof(float*), cudaMemcpyHostToDevice));

    // Transpose A and B
    CHECK_CUBLAS_ERROR(cublasSgemmBatched(HANDLE, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                                          &alpha, (const float**)d_Aarray, k,
                                          (const float**)d_Barray, n,
                                          &beta, d_Carray, m, batchSize));

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, batchSize * m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Free dynamically allocated arrays
    delete[] Aarray;
    delete[] Barray;
    delete[] Carray;

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaFree(d_Aarray));
    CHECK_CUDA_ERROR(cudaFree(d_Barray));
    CHECK_CUDA_ERROR(cudaFree(d_Carray));
}

void tensor_multiply(const float* h_A, const float* h_B, float* h_C, int batchSize, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, batchSize * m * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, batchSize * k * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, batchSize * m * n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, batchSize * m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, batchSize * k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate arrays for the pointers
    float** Aarray = new float*[batchSize];
    float** Barray = new float*[batchSize];
    float** Carray = new float*[batchSize];

    for (int i = 0; i < batchSize; ++i) {
        Aarray[i] = d_A + i * m * k;
        Barray[i] = d_B + i * k * n;
        Carray[i] = d_C + i * m * n;
    }

    float** d_Aarray;
    float** d_Barray;
    float** d_Carray;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Aarray, batchSize * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Barray, batchSize * sizeof(float*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Carray, batchSize * sizeof(float*)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_Aarray, Aarray, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Barray, Barray, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Carray, Carray, batchSize * sizeof(float*), cudaMemcpyHostToDevice));

    CHECK_CUBLAS_ERROR(cublasSgemmBatched(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                          &alpha, (const float**)d_Aarray, m,
                                          (const float**)d_Barray, k,
                                          &beta, d_Carray, m, batchSize));

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, batchSize * m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Free dynamically allocated arrays
    delete[] Aarray;
    delete[] Barray;
    delete[] Carray;

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaFree(d_Aarray));
    CHECK_CUDA_ERROR(cudaFree(d_Barray));
    CHECK_CUDA_ERROR(cudaFree(d_Carray));
}

// // CUDA kernel to add the bias vector to the activation matrix
// __global__ void add_bias(float* A, float* B, int rows, int cols) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalElements = rows * cols;
//     if (idx < totalElements) {
//         int row = idx % rows;
//         A[idx] += B[row];
//     }
// }

// // CUDA kernel to apply ReLU function to the activation matrix
// __global__ void apply_ReLU(float* A, int rows, int cols) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalElements = rows * cols;
//     if (idx < totalElements) {
//         if (A[idx] < 0) {
//             A[idx] = 0;
//         }
//     }
// }

// // CUDA kernel to apply softmax function to the activation matrix
// __global__ void apply_softmax(float* A, float* NORM, int rows, int cols) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalElements = rows * cols;
//     if (idx < totalElements) {
//         A[idx] = exp(A[idx]) / NORM[idx / rows];
//     }
// }

// // CUDA kernel to compute softmax normalizing constants
// __global__ void compute_softmax_norm(float* A, float* NORM, int rows, int cols) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalElements = rows * cols;
//     if (idx < totalElements) {
//         atomicAdd(&NORM[idx / rows], exp(A[idx]));
//     }
// }

// // CUDA kernel to compute the dC/dA2 Jacobian
// __global__ void compute_dC_dA2(float* gpu_dC_dA2, float *gpu_A2, float *gpu_Y, int rows, int cols) {
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
// __global__ void subtract_derivs(float* gpu_P, float *gpu_dP, float learning_rate, int rows, int cols, int slices) {
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
void print_tensor(string name, const float* gpu_tensor, int slices, int rows, int cols) {
    float *tensor = new float[slices * rows * cols];
    CHECK_CUDA_ERROR(cudaMemcpy(tensor, gpu_tensor, slices * rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    printf("%s (%d, %d, %d)\n", name, slices, rows, cols);
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

    // Perform matrix-matrix multiplication using cuBLAS
    CHECK_CUBLAS_ERROR(cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, L1_SIZE, BATCH_SIZE, INPUT_SIZE, &ALPHA, gpu_W1, L1_SIZE, gpu_X, INPUT_SIZE, &BETA, gpu_A1, L1_SIZE));

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
    CHECK_CUBLAS_ERROR(cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, OUTPUT_SIZE, BATCH_SIZE, L1_SIZE, &ALPHA, gpu_W2, OUTPUT_SIZE, gpu_A1, L1_SIZE, &BETA, gpu_A2, OUTPUT_SIZE));

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

void back_prop(float (&dC_dW1)[BATCH_SIZE * 1 * (L1_SIZE * INPUT_SIZE)], float (&dC_dB1)[BATCH_SIZE * 1 * (L1_SIZE * 1)], float (&dC_dW2)[BATCH_SIZE * 1 * (OUTPUT_SIZE * L1_SIZE)], float (&dC_dB2)[BATCH_SIZE * 1 * (OUTPUT_SIZE * 1)], float (&X)[INPUT_SIZE * BATCH_SIZE], float (&Y)[OUTPUT_SIZE * BATCH_SIZE], float (&A1)[L1_SIZE * BATCH_SIZE], float (&A2)[OUTPUT_SIZE * BATCH_SIZE], float (&W1)[L1_SIZE * INPUT_SIZE], float (&W2)[OUTPUT_SIZE * L1_SIZE]) {

}

void update_params(float (&W1)[L1_SIZE * INPUT_SIZE], float (&B1)[L1_SIZE], float (&W2)[OUTPUT_SIZE * L1_SIZE], float (&B2)[OUTPUT_SIZE], float (&dC_dW1)[BATCH_SIZE * 1 * (L1_SIZE * INPUT_SIZE)], float (&dC_dB1)[BATCH_SIZE * 1 * (L1_SIZE * 1)], float (&dC_dW2)[BATCH_SIZE * 1 * (OUTPUT_SIZE * L1_SIZE)], float (&dC_dB2)[BATCH_SIZE * 1 * (OUTPUT_SIZE * 1)]) {

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

// int get_num_correct(float (&A2)[OUTPUT_SIZE*BATCH_SIZE], float (&Y)[OUTPUT_SIZE*BATCH_SIZE]) {
//     int num_correct = 0;
//     for (int col = 0; col < BATCH_SIZE; col++) {
//         int predicted_class;
//         float predicted_probability = 0;
//         for (int row = 0; row < OUTPUT_SIZE; row++) {
//             if (A2[col*OUTPUT_SIZE + row] > predicted_probability) {
//                 predicted_class = row;
//                 predicted_probability = A2[col*OUTPUT_SIZE + row];
//             }
//         }
//         if (Y[col*OUTPUT_SIZE + predicted_class] == 1) {
//             num_correct++;
//         }
//     }
//     return num_correct;
// }

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
    float *gpu_NORM;

    // Perform gradient descent for each mini-batch
    for (int i = 0; i < NUM_TRAIN_IMAGES; i += BATCH_SIZE) { // can optimize this
    
        // Get image and label batch
        get_image_batch(X, data_offsets, i);
        get_label_batch(Y, data_offsets, i);

        // Copy from CPU to GPU
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_X, X, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_Y, Y, OUTPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Forward propagate to get activations A1 and A2
        forward_prop(gpu_X, gpu_A1, gpu_A2, gpu_W1, gpu_B1, gpu_W2, gpu_B2);

        // Add the number of correct predictions from the mini-batch
        int batch_correct = get_num_correct(gpu_A2, gpu_Y);
        total_correct += batch_correct;
        
        // Back propagate to get dC/W1, dC/dB1, dC/dW2, dC/dB2
        back_prop(gpu_dC_dW1, gpu_dC_dB1, gpu_dC_dW2, gpu_dC_dB2, gpu_X, gpu_Y, gpu_A1, gpu_A2, gpu_W1, gpu_W2);

        // Update parameters
        update_params(gpu_W1, gpu_B1, gpu_W2, gpu_B2, gpu_dC_dW1, gpu_dC_dB1, gpu_dC_dW2, gpu_dC_dB2);

        // Update console
        cout << "BATCH " << (i / BATCH_SIZE) + 1 << "/" << NUM_BATCHES << " COMPLETE" << endl;
        cout << "BATCH ACCURACY: " << 1.0 * batch_correct / BATCH_SIZE << endl;
        cout << "ACCURACY: " << 1.0 * total_correct / (i + BATCH_SIZE) << endl;
        cout << endl;
    }
}

__global__ void he_init_kernel(float* gpu_W, int m, int n, unsigned SEED) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        curandState state;
        curand_init(SEED, idx, 0, &state);
        gpu_W[idx] = curand_normal(&state) * sqrtf(2.0f / m);
    }
}

void he_init(float* gpu_W, int m, int n) {
    int num_blocks = (m * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    he_init_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_W, L1_SIZE, INPUT_SIZE, SEED);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

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

    // Initialize weights with He initialization method
    he_init(gpu_W1, L1_SIZE, INPUT_SIZE);
    he_init(gpu_W2, OUTPUT_SIZE, L1_SIZE);

    // print_tensor("W1", gpu_W1, 1, L1_SIZE, INPUT_SIZE);
    // print_tensor("W2", gpu_W2, 1, OUTPUT_SIZE, L1_SIZE);

    // Initialize biases to 0
    CHECK_CUDA_ERROR(cudaMemset(gpu_B1, 0, L1_SIZE * 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(gpu_B2, 0, OUTPUT_SIZE * 1 * sizeof(float)));

    // print_tensor("B1", gpu_B1, 1, L1_SIZE, 1);
    // print_tensor("B2", gpu_B2, 1, OUTPUT_SIZE, 1);

    // Perform gradient descent
    gradient_descent(gpu_W1, gpu_B1, gpu_W2, gpu_B2);

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(gpu_W1));
    CHECK_CUDA_ERROR(cudaFree(gpu_W2));
    CHECK_CUDA_ERROR(cudaFree(gpu_B1));
    CHECK_CUDA_ERROR(cudaFree(gpu_B2));
    
    // Free HANDLE
    cublasDestroy(HANDLE);

    return 0;
}