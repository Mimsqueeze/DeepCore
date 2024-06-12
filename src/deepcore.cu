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
#include <stdlib.h>
#include <memory>
#include <sstream>

#define BLOCK_SIZE 256

cublasHandle_t HANDLE;
const unsigned SEED = std::chrono::system_clock::now().time_since_epoch().count();
const float ALPHA = 1.0f;
const float BETA = 0.0f;

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

enum Activation {
    NO_ACTIVATION = -1, RELU = 0, SOFTMAX = 1
};

enum Loss {
    NO_LOSS = -1, CROSS_ENTROPY = 0, MSE = 1
};

enum LayerType {
    NO_LAYER = -1, FLATTEN = 0, DENSE = 1
};

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
__global__ void compute_dC_dAL_kernel(float *gpu_dC_dAL, float *gpu_AL, float *gpu_Y, int slices, int rows, int cols, Loss func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = slices * rows * cols;
    if (idx < totalElements) {
        int slice = idx / (rows * cols); // k
        int col = (idx / rows) % cols; // j

        if (func == CROSS_ENTROPY) {
            // Using cross-entropy loss function
            gpu_dC_dAL[idx] = -gpu_Y[slice * cols + col] / gpu_AL[slice * cols + col];
        } else if (func == MSE)  {
            // Using mean-square error loss function
            gpu_dC_dAL[idx] = gpu_AL[slice * cols + col] - gpu_Y[slice * cols + col];
        }
    }
}

void compute_dC_dAL(float *gpu_dC_dAL, float *gpu_AL, float *gpu_Y, int slices, int rows, int cols, Loss func) {
    int num_blocks = (slices * rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_dC_dAL_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_dC_dAL, gpu_AL, gpu_Y, slices, rows, cols, func);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void compute_dA_dZ_kernel(float* gpu_dA_dZ, float *gpu_A, int slices, int rows, int cols, Activation func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = slices * rows * cols;
    if (idx < totalElements) {
        int slice = idx / (rows * cols); // k
        int col = (idx / rows) % cols; // j
        int row = idx % rows; // i
        float a_i = gpu_A[slice * rows + row];
        float a_j = gpu_A[slice * rows + col];
        if (func == SOFTMAX) {
            if (row == col) {
                gpu_dA_dZ[idx] = a_i * (1.0f - a_j);
            } else {
                gpu_dA_dZ[idx] = a_i * (-a_j);
            }
        } else if (func == RELU) {
            if (col == row && a_i > 0) {
                gpu_dA_dZ[idx] = 1;
            } else {
                gpu_dA_dZ[idx] = 0;
            }
        }

    }
}

void compute_dA_dZ(float* gpu_dA_dZ, float *gpu_A, int slices, int rows, int cols, Activation func) {
    int num_blocks = (slices * rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_dA_dZ_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_dA_dZ, gpu_A, slices, rows, cols, func);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void compute_dZ_dA_prev(float* gpu_dZ_dA_prev, float *gpu_W, int slices, int rows, int cols) {
    for (int slice = 0; slice < slices; slice++) {
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_dZ_dA_prev + slice * rows * cols, gpu_W, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

// CUDA kernel to update params in gpu_P
__global__ void update_params_kernel(float *gpu_P, float *gpu_dP, float learning_rate, int rows, int cols, int batches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx < totalElements) { // let's just use a loop here
        float avg_deriv = 0;
        for (int batch = 0; batch < batches; batch++) {
            avg_deriv += gpu_dP[idx + (batch * rows * cols)];
        }
        avg_deriv /= batches;
        gpu_P[idx] = gpu_P[idx] - learning_rate * avg_deriv;
    }
}

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

void print_batch(float *gpu_X, int num_features, int batch_size, float *gpu_Y, int num_classes) {
    // CPU matrices
    float *X = new float[num_features * batch_size];
    float *Y = new float[num_classes * batch_size];

    // Copy from GPU to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(X, gpu_X, num_features * batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(Y, gpu_Y, num_classes * batch_size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch_size; i++) {
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
        cout << " - ACTUAL LABEL: ";
        for (int label = 0; label < 10; label++) {
            if (Y[label + i * 10] == 1) {
                cout << label << endl;
                break;
            }
        }
        cout << endl;
    }
    delete[] X;
    delete[] Y;
}

class DeepCore {
public:
    DeepCore() : layers() {} 

    class Layer { // let's just make this a dense layer for now
    public:
        virtual int init_batch(int prev_num_nodes, int batch_size) = 0;
        virtual int init(int prev_num_nodes) = 0;
        virtual void destroy() = 0;
        virtual void destroy_batch() = 0;
        virtual float *forward_prop(float *gpu_A_prev, int batch_size) = 0;
        virtual float *back_prop(float *gpu_dC_dA, int batch_size) = 0;
        virtual void update_params(float learning_rate, int batch_size) = 0;
        virtual void save(ofstream &file) = 0;
        string name, output_shape;
        int param_count;
        static void save_matrix(ofstream &file, float *gpu_X, int rows, int cols) {
            // Allocate memory on CPU
            float *X = new float[rows * cols];

            // Copy memory from GPU to CPU
            CHECK_CUDA_ERROR(cudaMemcpy(X, gpu_X, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    file.write((char *) &X[j * rows + i], sizeof(float));
                }
            }
            // Free CPU memory
            delete[] X;
        }
        static void read_matrix(ifstream &file, float *gpu_X, int rows, int cols) {
            // Allocate memory on CPU
            float *X = new float[rows * cols];

            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    file.read((char *) &X[j * rows + i], sizeof(float));
                }
            }

            // Copy memory from CPU to GPU
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_X, X, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

            // Free CPU memory
            delete[] X;
        }
    };

    class Dense : public Layer {
    public:
        Dense(int num_nodes, Activation activation_func)
            : num_nodes(num_nodes), activation_func(activation_func) {}
        Dense(ifstream &file) {
            file.read((char *) &prev_num_nodes, sizeof(int)); // next 4 bytes
            file.read((char *) &num_nodes, sizeof(int)); // next 4 bytes
            file.read((char *) &activation_func, sizeof(int)); // next 4 bytes

            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W, num_nodes * prev_num_nodes * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B, num_nodes * 1 * sizeof(float)));

            // read weights and biases from file
            read_matrix(file, gpu_W, num_nodes, prev_num_nodes);
            read_matrix(file, gpu_B, num_nodes, 1);
        }
        int init(int prev_num_nodes) override {
            this->prev_num_nodes = prev_num_nodes;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W, num_nodes * prev_num_nodes * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B, num_nodes * 1 * sizeof(float)));

            if (activation_func == RELU) { // initialize weights
                he_init(gpu_W, num_nodes, prev_num_nodes);
            } else if (activation_func == SOFTMAX) {
                xavier_init(gpu_W, num_nodes, prev_num_nodes);
            }

            // initialized biases
            CHECK_CUDA_ERROR(cudaMemset(gpu_B, 0, num_nodes * 1 * sizeof(float)));

            return num_nodes;
        }
        int init_batch(int prev_num_nodes, int batch_size) override {
            this->prev_num_nodes = prev_num_nodes;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A, num_nodes * batch_size * sizeof(float)));

            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dA_dZ, batch_size * num_nodes * num_nodes * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dZ_dA_prev, batch_size * num_nodes * prev_num_nodes * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dA_dA_prev, batch_size * num_nodes * prev_num_nodes * sizeof(float)));

            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dZ, batch_size * 1 * num_nodes * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dW, batch_size * num_nodes * prev_num_nodes * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dB, batch_size * 1 * num_nodes * sizeof(float)));

            // initialize fields for output string
            name = "Dense";
            output_shape = "(" + to_string(batch_size) + ", " + to_string(num_nodes) + ", 1)";
            param_count = (prev_num_nodes * num_nodes) + num_nodes;

            return num_nodes;
        }
        void destroy() override {
            CHECK_CUDA_ERROR(cudaFree(gpu_W));
            CHECK_CUDA_ERROR(cudaFree(gpu_B));
        }
        void destroy_batch() override {
            CHECK_CUDA_ERROR(cudaFree(gpu_A));

            CHECK_CUDA_ERROR(cudaFree(gpu_dA_dZ));
            CHECK_CUDA_ERROR(cudaFree(gpu_dZ_dA_prev));
            CHECK_CUDA_ERROR(cudaFree(gpu_dA_dA_prev));

            CHECK_CUDA_ERROR(cudaFree(gpu_dC_dZ));
            CHECK_CUDA_ERROR(cudaFree(gpu_dC_dW));
            CHECK_CUDA_ERROR(cudaFree(gpu_dC_dB));
        }
        float *forward_prop(float *gpu_A_prev, int batch_size) override {
            this->gpu_A_prev = gpu_A_prev;

            // Perform A = W*prev_A
            multiply_tensor(gpu_A, gpu_W, gpu_A_prev, 1, num_nodes, batch_size, prev_num_nodes);

            // Perform A = A + B
            add_bias(gpu_A, gpu_B, num_nodes, batch_size);

            if (activation_func == RELU) {
                // Perform A = ReLU(A)
                apply_ReLU(gpu_A, num_nodes, batch_size);
            } else if (activation_func == SOFTMAX) {
                apply_softmax(gpu_A, num_nodes, batch_size);
            }
            return gpu_A;
        }
        float *back_prop(float *gpu_dC_dA, int batch_size) override {
            compute_dA_dZ(gpu_dA_dZ, gpu_A, batch_size, num_nodes, num_nodes, activation_func);
            compute_dZ_dA_prev(gpu_dZ_dA_prev, gpu_W, batch_size, num_nodes, prev_num_nodes);
            multiply_tensor(gpu_dA_dA_prev, gpu_dA_dZ, gpu_dZ_dA_prev, batch_size, num_nodes, prev_num_nodes, num_nodes);

            multiply_tensor(gpu_dC_dZ, gpu_dC_dA, gpu_dA_dZ, batch_size, 1, num_nodes, num_nodes);
            multiply_tensor_T(gpu_dC_dW, gpu_dC_dZ, gpu_A_prev, batch_size, num_nodes, prev_num_nodes, 1);
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_dC_dB, gpu_dC_dZ, batch_size * 1 * num_nodes * sizeof(float), cudaMemcpyDeviceToDevice));

            float *gpu_dC_dA_prev;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dA_prev, batch_size * 1 * prev_num_nodes * sizeof(float)));
            multiply_tensor(gpu_dC_dA_prev, gpu_dC_dA, gpu_dA_dA_prev, batch_size, 1, prev_num_nodes, num_nodes);
            return gpu_dC_dA_prev; // must be freed in caller
        }
        void update_params(float learning_rate, int batch_size) override {
            // Launch CUDA kernel
            int num_blocks = (num_nodes * prev_num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
            update_params_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_W, gpu_dC_dW, learning_rate, num_nodes, prev_num_nodes, batch_size);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Launch CUDA kernel
            num_blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
            update_params_kernel<<<num_blocks, BLOCK_SIZE>>>(gpu_B, gpu_dC_dB, learning_rate, num_nodes, 1, batch_size);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        void save(ofstream &file) {
            LayerType layer_type = DENSE;
            file.write((char *) &layer_type, sizeof(int)); // first 4 bytes
            file.write((char *) &prev_num_nodes, sizeof(int)); // next 4 bytes
            file.write((char *) &num_nodes, sizeof(int)); // next 4 bytes
            file.write((char *) &activation_func, sizeof(int)); // next 4 bytes
            save_matrix(file, gpu_W, num_nodes, prev_num_nodes);
            save_matrix(file, gpu_B, num_nodes, 1);
        }
    private:
        int prev_num_nodes;
        int num_nodes;
        Activation activation_func;
        float *gpu_A;
        float *gpu_W;
        float *gpu_B;

        float *gpu_dA_dZ;
        float *gpu_dZ_dA_prev;
        float *gpu_dA_dA_prev;

        float *gpu_dC_dZ;
        float *gpu_dC_dW;
        float *gpu_dC_dB;

        float *gpu_A_prev; // reference to activations of previous layer
    };

    class Flatten : public Layer {
    public:
        Flatten(int num_nodes) : num_nodes(num_nodes) {}
        Flatten(ifstream &file) {
            file.read((char *) &num_nodes, sizeof(int)); // next 4 bytes
        }
        int init(int prev_num_nodes) override {
            return num_nodes;
        }
        int init_batch(int prev_num_nodes, int batch_size) override {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A, num_nodes * batch_size * sizeof(float)));

            // initialize fields for output string
            name = "Flatten";
            output_shape = "(" + to_string(batch_size) + ", " + to_string(num_nodes) + ", 1)";
            param_count = 0;

            return num_nodes;
        }
        void destroy() override {}
        void destroy_batch() override {
            CHECK_CUDA_ERROR(cudaFree(gpu_A));
        }
        float *forward_prop(float *gpu_A_prev, int batch_size) override {
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_A, gpu_A_prev, num_nodes * batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
            return gpu_A;
        }
        float *back_prop(float *gpu_dC_dA, int batch_size) override {
            return gpu_dC_dA;
        }
        void update_params(float learning_rate, int batch_size) override {}
        void save(ofstream &file) {
            LayerType layer_type = FLATTEN;
            file.write((char *) &layer_type, sizeof(int)); // first 4 bytes
            file.write((char *) &num_nodes, sizeof(int)); // next 4 bytes
        }
    private:
        int num_nodes;
        float *gpu_A;
    };

    void add(unique_ptr<DeepCore::Layer> layer) {
        layers.push_back(move(layer));
    }
    void compile(Loss loss_func) {
        this->loss_func = loss_func;
        // Initialize layers
        int prev_num_nodes = -1;
        for (const auto &layer : layers) {
            prev_num_nodes = (*layer).init(prev_num_nodes);
        }
    }
    void destroy() {
        for (const auto &layer : layers) {
            layer->destroy();
        }
    }
    void fit(float *X, int num_features, int num_samples, float *Y, int num_classes, int batch_size = 50, int epochs = 10, float learning_rate = 0.1, 
             float *validation_X = nullptr, int num_validation = -1, float *validation_Y = nullptr) {

        // Note: X must be of dimension num_features x num_samples
        // Y is of dimension num_classes x num_samples

        // Initialize layers dependent on batch size
        int prev_num_nodes = -1;
        for (const auto &layer : layers) {
            prev_num_nodes = (*layer).init_batch(prev_num_nodes, batch_size);
        }

        // Print the compiled model
        stringstream model_string;
        model_string << "COMPILED MODEL:" << endl;
        model_string << "______________________________________________________________________" << endl;
        model_string << " Layer (type)                 Output Shape                  Param #   " << endl;
        model_string << "======================================================================" << endl;

        int total_params = 0;
        for (const auto& layer : layers) {
            model_string << " " << left << setw(29) << layer->name 
            << left << setw(30) << layer->output_shape 
            << left << setw(10) << layer->param_count << endl;
            total_params += layer->param_count;
        }

        model_string << "======================================================================" << endl;
        model_string << "Total trainable params: " << total_params << "\n";
        model_string << "______________________________________________________________________" << endl;
        
        cout << model_string.str();

        // Adjust num_samples for batch_size
        this->num_batches = num_samples/batch_size;
        num_samples = num_batches * batch_size;

        // For each epoch, perform gradient descent and update weights and biases
        for (int epoch = 1; epoch <= epochs; epoch++) {
            cout << endl << "EPOCH " << epoch << "/" << epochs << endl;

            // Get start time
            auto start = chrono::high_resolution_clock::now();

            // Store number of correct predictions
            int train_correct = gradient_descent(X, num_features, num_samples, Y, num_classes, batch_size, learning_rate);

            // Get end time
            auto end = chrono::high_resolution_clock::now();

            // Calculate duration of time passed
            double duration = (double) chrono::duration_cast<chrono::microseconds>(end - start).count()/1000000.0;

            // Calculate remaining time
            int seconds = (int) duration*(epochs - epoch);
            int minutes= seconds/60;
            int hours= minutes/60;
            minutes %= 60;
            seconds %= 60;

            // Print the results of the epoch
            stringstream output;
            output << "TRAIN ACCURACY: " << train_correct << "/" << num_samples;
            output << " (" << fixed << setprecision(2) << 100.0 * train_correct / num_samples << "%)";
            
            // Find performance on validation set if provided
            if (validation_X != nullptr && validation_Y != nullptr && num_validation > 0) {
                
                num_validation = (num_validation / batch_size) * batch_size;

                int validation_correct = 0;

                // Create array of offsets each associated with a label/image pair
                int *data_offsets = new int[num_validation];

                // Fill with numbers 0 to num_validation-1 in increasing order
                iota(data_offsets, data_offsets + num_validation, 0);

                float *gpu_valid_X;
                float *gpu_valid_Y;

                CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_valid_X, num_features * batch_size * sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_valid_Y, num_classes * batch_size * sizeof(float)));
                
                for (int i = 0; i < num_validation; i += batch_size) {
                
                    // Load features and labels
                    for (int sample_num = 0; sample_num < batch_size; sample_num++) {
                        CHECK_CUDA_ERROR(cudaMemcpy(gpu_valid_X + sample_num * num_features, validation_X + data_offsets[sample_num + i] * num_features, num_features * 1 * sizeof(float), cudaMemcpyHostToDevice));
                        CHECK_CUDA_ERROR(cudaMemcpy(gpu_valid_Y + sample_num * num_classes, validation_Y + data_offsets[sample_num + i] * num_classes, num_classes * 1 * sizeof(float), cudaMemcpyHostToDevice));
                    }

                    // Forward propagate each layer
                    float *gpu_A_prev = gpu_valid_X;
                    for (const auto &layer : layers) {
                        gpu_A_prev = layer->forward_prop(gpu_A_prev, batch_size);
                    }

                    // Add the number of correct predictions from the mini-batch
                    int batch_correct = get_num_correct(gpu_A_prev, gpu_valid_Y, num_classes, batch_size);
                    validation_correct += batch_correct;
                }

                output << " - VALIDATION ACCURACY: " << validation_correct << "/" << num_validation;
                output << " (" << fixed << setprecision(2) << 100.0 * validation_correct / num_validation << "%)";

                CHECK_CUDA_ERROR(cudaFree(gpu_valid_X));
                CHECK_CUDA_ERROR(cudaFree(gpu_valid_Y));
                delete[] data_offsets;
            }

            output << " - TIME ELAPSED: " << fixed << setprecision(2) << duration << "s";
            output << " - ETA: " << setfill('0') << setw(2) << hours << ":"
            << setfill('0') << setw(2) << minutes << ":"
            << setfill('0') << setw(2) << seconds;

            // Print all at once
            cout << output.str() << endl;
        }

        cout << ">>> TRAINING COMPLETE." << endl << endl;

        // Destroy layers dependent on batch size
        for (const auto &layer : layers) {
            layer->destroy_batch();
        }
    }
    void evaluate(float *test_X, int num_features, int num_test, float *test_Y, int num_classes, int batch_size = 50) {

        // Initialize layers dependent on batch size
        int prev_num_nodes = -1;
        for (const auto &layer : layers) {
            prev_num_nodes = (*layer).init_batch(prev_num_nodes, batch_size);
        }

        int num_test_batches = (num_test / batch_size);
        num_test = num_test_batches * batch_size;

        int test_correct = 0;

        // Create array of offsets each associated with a label/image pair
        int *data_offsets = new int[num_test];

        // Fill with numbers 0 to num_test-1 in increasing order
        iota(data_offsets, data_offsets + num_test, 0);

        float *gpu_test_X;
        float *gpu_test_Y;

        CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_test_X, num_features * batch_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_test_Y, num_classes * batch_size * sizeof(float)));
        
        for (int i = 0; i < num_test; i += batch_size) {
        
            // Load features and labels
            for (int sample_num = 0; sample_num < batch_size; sample_num++) {
                CHECK_CUDA_ERROR(cudaMemcpy(gpu_test_X + sample_num * num_features, test_X + data_offsets[sample_num + i] * num_features, num_features * 1 * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(gpu_test_Y + sample_num * num_classes, test_Y + data_offsets[sample_num + i] * num_classes, num_classes * 1 * sizeof(float), cudaMemcpyHostToDevice));
            }

            // Forward propagate each layer
            float *gpu_A_prev = gpu_test_X;
            for (const auto &layer : layers) {
                gpu_A_prev = layer->forward_prop(gpu_A_prev, batch_size);
            }

            // Add the number of correct predictions from the mini-batch
            int batch_correct = get_num_correct(gpu_A_prev, gpu_test_Y, num_classes, batch_size);
            test_correct += batch_correct;

            // Update console
            stringstream output;

            // Update console
            output << "\r";
            output << "BATCH " << (i / batch_size) + 1 << "/" << num_test_batches << " ";
            output << "[";

            float percentage_completion = (((float)i / batch_size) + 1) / num_test_batches;
            bool arrow = true;

            for (int j = 1; j <= 32; j++) {
                if (percentage_completion >= (float)j / 32) {
                    output << "=";
                } else {
                    if (arrow) {
                        output << ">";
                        arrow = false;
                    } else {
                        output << ".";
                    }
                }
            }

            output << "] - TEST ACCURACY: " << test_correct << "/" << num_test;
            output << " (" << fixed << setprecision(2) << 100.0 * test_correct / num_test << "%)";

            // Print all at once
            cout << output.str() << flush;
        }
        cout << endl;
        cout << ">>> TESTING COMPLETE." << endl << endl;

        CHECK_CUDA_ERROR(cudaFree(gpu_test_X));
        CHECK_CUDA_ERROR(cudaFree(gpu_test_Y));

        // Destroy layers dependent on batch size
        for (const auto &layer : layers) {
            layer->destroy_batch();
        }

        delete[] data_offsets;
    }
    void save(string path) {
        // Declare and open file
        ofstream file = ofstream(path, ios::out | ios::binary);

        cout << "SAVING MODEL TO " << path << endl;

        if (file.is_open()) {
            // Save each layer to file
            for (const auto &layer : layers) {
                layer->save(file);
            }

            // Denotes end of the model
            LayerType end_of_file = NO_LAYER;
            file.write((char *) &end_of_file, sizeof(int)); // first 4 bytes

            cout << ">>> SAVING COMPLETE." << endl << endl;
            // Close the file
            file.close();
        } else {
            cerr << "ERROR: FAILED TO OPEN FILE " << path;
            exit(1);
        }
    }
    void read(string path) {
        // Declare and open file
        ifstream file(path, ios::in | ios::binary);

        cout << "READING MODEL FROM " << path << endl;

        if (file.is_open()) {
            LayerType layer_type;
            file.read((char *) &layer_type, sizeof(LayerType));
            while (layer_type != NO_LAYER) {
                if (layer_type == FLATTEN) {
                    this->add(make_unique<DeepCore::Flatten>(file));
                } else if (layer_type == DENSE) {
                    this->add(make_unique<DeepCore::Dense>(file));
                }
                file.read((char *) &layer_type, sizeof(LayerType));
            }
            cout << ">>> READING COMPLETE." << endl << endl;
            file.close();
        } else {
            cerr << "ERROR: FAILED TO OPEN FILE " << path;
            exit(1);
        }
    }
private:
    vector<unique_ptr<DeepCore::Layer>> layers;
    Loss loss_func;
    int num_batches;
    
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

    int gradient_descent(float *X, int num_features, int num_samples, float *Y, int num_classes, int batch_size, float learning_rate) {
        // Number of correct predictions
        int total_correct = 0;

        // Create array of offsets each associated with a label/image pair
        int *data_offsets = new int[num_samples];

        // Fill with numbers 0 to num_samples-1 in increasing order
        iota(data_offsets, data_offsets + num_samples, 0);

        // Randomly shuffle array of offsets to randomize image selection in mini-batches
        shuffle(data_offsets, data_offsets + num_samples, default_random_engine(chrono::system_clock::now().time_since_epoch().count()));

        float *gpu_X;
        float *gpu_Y;

        CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_X, num_features * batch_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_Y, num_classes * batch_size * sizeof(float)));
        
        // Perform gradient descent for each mini-batch
        for (int i = 0; i < num_samples; i += batch_size) {
            
            // Load features and labels
            for (int sample_num = 0; sample_num < batch_size; sample_num++) {
                CHECK_CUDA_ERROR(cudaMemcpy(gpu_X + sample_num * num_features, X + data_offsets[sample_num + i] * num_features, num_features * 1 * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(gpu_Y + sample_num * num_classes, Y + data_offsets[sample_num + i] * num_classes, num_classes * 1 * sizeof(float), cudaMemcpyHostToDevice));
            }

            // Debug: print the features and labels
            // print_batch(gpu_X, num_features, batch_size, gpu_Y, num_classes);

            // Forward propagate each layer
            float *gpu_A_prev = gpu_X;
            for (const auto &layer : layers) {
                gpu_A_prev = layer->forward_prop(gpu_A_prev, batch_size);
            }

            // Add the number of correct predictions from the mini-batch
            int batch_correct = get_num_correct(gpu_A_prev, gpu_Y, num_classes, batch_size);
            total_correct += batch_correct;
            
            // Back propagate
            float *gpu_dC_dA_prev;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_dC_dA_prev, batch_size * 1 * num_classes * sizeof(float)));
            compute_dC_dAL(gpu_dC_dA_prev, gpu_A_prev, gpu_Y, batch_size, 1, num_classes, loss_func);
            float *gpu_dC_dA;
            for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) {
                gpu_dC_dA = (*layer)->back_prop(gpu_dC_dA_prev, batch_size);
                CHECK_CUDA_ERROR(cudaFree(gpu_dC_dA_prev));
                gpu_dC_dA_prev = gpu_dC_dA;
            }

            // Update parameters
            for (const auto &layer : layers) {
                layer->update_params(learning_rate, batch_size);
            }

            // Update console
            stringstream output;

            output << "\r";
            output << "BATCH " << (i / batch_size) + 1 << "/" << num_batches << " ";
            output << "[";

            float percentage_completion = (((float) i / batch_size) + 1) / num_batches;
            bool arrow = true;
            
            for (int j = 1; j <= 32; j++) {
                if (percentage_completion >= (float) j / 32) {
                    output << "=";
                } else {
                    if (arrow) {
                        output << ">";
                        arrow = false;
                    } else {
                        output << ".";
                    }
                }
            }

            output << "] - BATCH ACCURACY: ";
            output << fixed << setprecision(3) << (float)batch_correct / batch_size;
            output << " - TOTAL ACCURACY: ";
            output << fixed << setprecision(3) << (float)total_correct / (i + batch_size);
            cout << output.str() << flush;;
        }
        cout << endl;

        CHECK_CUDA_ERROR(cudaFree(gpu_X));
        CHECK_CUDA_ERROR(cudaFree(gpu_Y));
        delete[] data_offsets;

        return total_correct;
    }
};

void get_labels(float *Y, int num_classes, int num_samples, string path, int start_offset) {
    ifstream labels_file(path, ios::in | ios::binary);
    if (labels_file.is_open()) {
        labels_file.seekg(start_offset);
        for (int i = 0; i < num_samples; i++) {
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
        cerr << "ERROR: FAILED TO OPEN FILE " << path << endl;
        exit(1);
    }
}

void get_images(float *X, int num_features, int num_samples, string path, int start_offset) {
    ifstream images_file(path, ios::in | ios::binary);
    if (images_file.is_open()) {
        images_file.seekg(start_offset);
        for (int i = 0; i < num_samples; i++) {
            for (int j= 0; j < 784; j++) {
                int value = 0;
                images_file.read((char *) &value, 1);
                X[j + i * 784] = value/255.0; // Transform value from range [0, 255] to range [0, 1]
            }
        }
        images_file.close();
    } else {
        cerr << "ERROR: FAILED TO OPEN FILE " << path << endl;
        exit(1);
    }
}

void print_batch(float *X, float *Y, int size) {
    // For size number of labels/images, print them
    for (int i = 0; i < size; i++) {
        // Print label
        cout << "The following number is: ";
        for (int j = 0; j < 10; j++) {
            if (Y[i*10 + j] == 1) {
                cout << j << "\n";
                break;
            }
        }
        // Print image
        for (int j = 0; j < 784; j++) {
            if (j != 0 && j % 28 == 0) {
                cout << "\n";
            }
            if (X[i*784 + j] < 0.5) {
                cout << "@.@"; // Represents dark pixel
            } else {
                cout << " . "; // Represents light pixel
            }
        }
        cout << "\n";
    }
}

#define TRAIN_LABELS_FILE_PATH R"(.\data\train-labels.idx1-ubyte)"
#define TRAIN_IMAGES_FILE_PATH R"(.\data\train-images.idx3-ubyte)"
#define TEST_LABELS_FILE_PATH R"(.\data\t10k-labels.idx1-ubyte)"
#define TEST_IMAGES_FILE_PATH R"(.\data\t10k-images.idx3-ubyte)"

#define LABEL_START 8
#define IMAGE_START 16

#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000

float X[784 * NUM_TRAIN_IMAGES];
float Y[10 * NUM_TRAIN_IMAGES];

float test_X[784 * NUM_TEST_IMAGES];
float test_Y[10 * NUM_TEST_IMAGES];

int main() {
    // Initialize cuBLAS handle
    cublasCreate(&HANDLE);

    get_images(X, 784, NUM_TRAIN_IMAGES, TRAIN_IMAGES_FILE_PATH, IMAGE_START);
    get_labels(Y, 10, NUM_TRAIN_IMAGES, TRAIN_LABELS_FILE_PATH, LABEL_START);

    get_images(test_X, 784, NUM_TEST_IMAGES, TEST_IMAGES_FILE_PATH, IMAGE_START);
    get_labels(test_Y, 10, NUM_TEST_IMAGES, TEST_LABELS_FILE_PATH, LABEL_START);

    DeepCore model;
    model.add(make_unique<DeepCore::Flatten>(784));
    model.add(make_unique<DeepCore::Dense>(400, RELU));
    model.add(make_unique<DeepCore::Dense>(100, RELU));
    model.add(make_unique<DeepCore::Dense>(10, SOFTMAX));
    model.compile(CROSS_ENTROPY);
    model.fit(X, 784, NUM_TRAIN_IMAGES, Y, 10, 50, 3, 0.1, test_X, NUM_TEST_IMAGES, test_Y);
    model.evaluate(test_X, 784, NUM_TEST_IMAGES, test_Y, 10, 100);
    model.save(R"(.\models\784-400-100-10.bin)");
    model.destroy();

    DeepCore model1;
    model1.read(R"(.\models\784-400-100-10.bin)");
    model1.evaluate(test_X, 784, NUM_TEST_IMAGES, test_Y, 10, 200);
    model1.destroy();
    return 0;
}