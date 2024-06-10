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
        virtual int init(int prev_num_nodes, int batch_size) = 0;
        virtual void destroy() = 0;
        virtual float *forward_prop(float *gpu_prev_A, int batch_size) = 0;
    };

    class Dense : public Layer {
    public:
        Dense(int num_nodes, Activation activation_func)
            : num_nodes(num_nodes), activation_func(activation_func) {}

        int init(int prev_num_nodes, int batch_size) override {
            this->prev_num_nodes = prev_num_nodes;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_W, num_nodes * prev_num_nodes * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_B, num_nodes * 1 * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A, num_nodes * batch_size * sizeof(float)));

            if (activation_func == RELU) { // initialize weights
                he_init(gpu_W, num_nodes, prev_num_nodes);
            } else if (activation_func == SOFTMAX) {
                xavier_init(gpu_W, num_nodes, prev_num_nodes);
            }

            // initialized biases
            CHECK_CUDA_ERROR(cudaMemset(gpu_B, 0, num_nodes * 1 * sizeof(float)));
            return num_nodes;
        }
        void destroy() override {
            CHECK_CUDA_ERROR(cudaFree(gpu_W));
            CHECK_CUDA_ERROR(cudaFree(gpu_B));
            CHECK_CUDA_ERROR(cudaFree(gpu_A));
        }
        float *forward_prop(float *gpu_prev_A, int batch_size) override {
            // Perform A = W*prev_A
            multiply_tensor(gpu_A, gpu_W, gpu_prev_A, 1, num_nodes, batch_size, prev_num_nodes);

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
    private:
        int prev_num_nodes;
        int num_nodes;
        Activation activation_func;
        float *gpu_A;
        float *gpu_W;
        float *gpu_B;
    };

    class Flatten : public Layer {
    public:
        Flatten(int num_nodes) : num_nodes(num_nodes) {}
        int init(int prev_num_nodes, int batch_size) override {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A, num_nodes * batch_size * sizeof(float)));
            return num_nodes;
        }
        void destroy() override {
            CHECK_CUDA_ERROR(cudaFree(gpu_A));
        }
        float *forward_prop(float *gpu_prev_A, int batch_size) override {
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_A, gpu_prev_A, num_nodes * batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
            return gpu_A;
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
    }
    void destroy() {
        for (const auto &layer : layers) {
            layer->destroy();
        }
    }
    void fit(float *X, int num_features, int num_samples, float *Y, int num_classes, int batch_size = 50, int epochs = 10) {
        // Note: X must be of dimension num_features x num_samples
        // Y is of dimension num_classes x num_samples

        // Initialize layers
        this->batch_size = batch_size;
        int prev_num_nodes = -1;
        for (const auto &layer : layers) {
            prev_num_nodes = (*layer).init(prev_num_nodes, batch_size);
        }

        // Adjust num_samples for batch_size
        num_samples = (num_samples/batch_size) * batch_size;

        // For each epoch, perform gradient descent and update weights and biases
        for (int epoch = 1; epoch <= epochs; epoch++) {
            cout << "EPOCH " << epoch << "/" << epochs << endl;

            // Get start time
            auto start = chrono::high_resolution_clock::now();

            // Store number of correct predictions
            int train_correct = gradient_descent(X, num_features, num_samples, Y, num_classes, batch_size);

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
            cout << "TRAIN ACCURACY: " << train_correct << "/" << num_samples;
            printf(" (%.2f%%)", 100.0 * train_correct / num_samples);
            cout << " - TIME ELAPSED: ";
            printf("%.2fs", duration);
            cout << " - ETA: ";
            printf("%02d:%02d:%02d\n", hours, minutes, seconds);
            cout << endl;
        }

        cout << "FINISHED TRAINING." << endl;
    }
    void evaluate() {

    }
private:
    vector<unique_ptr<DeepCore::Layer>> layers;
    Loss loss_func;
    int batch_size;

    int gradient_descent(float *X, int num_features, int num_samples, float *Y, int num_classes, int batch_size) {
        // Number of correct predictions
        int total_correct = 0;

        // Create array of offsets each associated with a label/image pair
        int *data_offsets = new int[num_samples];

        // Fill with numbers 0 to num_samples-1 in increasing order
        iota(data_offsets, data_offsets + num_samples, 0);

        // Randomly shuffle array of offsets to randomize image selection in mini-batches
        shuffle(data_offsets, data_offsets + num_samples, default_random_engine(SEED));

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
            float *gpu_prev_A = gpu_X;
            for (const auto &layer : layers) {
                gpu_prev_A = layer->forward_prop(gpu_prev_A, batch_size);
            }

            cout << "done forward prop" << endl;
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
        cout << "Error: Failed to open file " << path << endl;
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
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }
}

#define TRAIN_LABELS_FILE_PATH R"(.\data\train-labels.idx1-ubyte)"
#define TRAIN_IMAGES_FILE_PATH R"(.\data\train-images.idx3-ubyte)"
#define TEST_LABELS_FILE_PATH R"(.\data\t10k-labels.idx1-ubyte)"
#define TEST_IMAGES_FILE_PATH R"(.\data\t10k-images.idx3-ubyte)"
#define LABEL_START 8
#define IMAGE_START 16
#define NUM_TRAIN_IMAGES 100


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

int main() {
    // Initialize cuBLAS handle
    cublasCreate(&HANDLE);

    float X[784 * NUM_TRAIN_IMAGES];
    float Y[10 * NUM_TRAIN_IMAGES];
    get_images(X, 784, NUM_TRAIN_IMAGES, TRAIN_IMAGES_FILE_PATH, IMAGE_START);
    get_labels(Y, 10, NUM_TRAIN_IMAGES, TRAIN_LABELS_FILE_PATH, LABEL_START);

    DeepCore model;
    model.add(make_unique<DeepCore::Flatten>(784)); // input layer
    model.add(make_unique<DeepCore::Dense>(100, RELU)); // first layer
    model.add(make_unique<DeepCore::Dense>(10, SOFTMAX)); // second (output) layer
    model.compile(CROSS_ENTROPY);
    model.fit(X, 784, NUM_TRAIN_IMAGES, Y, 10, 50, 10);
    model.destroy();

    return 0;
}