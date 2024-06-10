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
const unsigned SEED = chrono::system_clock::now().time_since_epoch().count();

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

void get_batch(float *gpu_X, float *X, int num_features, int *data_offsets, int i) {

}

class DeepCore {
public:
    DeepCore() : layers() { // call vector constructor
        // Initialize cuBLAS handle
        cublasCreate(&handle);
    } 

    class Layer { // let's just make this a dense layer for now
    public:
        virtual int init(int prev_num_nodes, int batch_size) const = 0;
        virtual void destroy() const = 0;
    };

    class Dense : public Layer {
    public:
        Dense(int num_nodes, Activation activation_func)
            : num_nodes(num_nodes), activation_func(activation_func) {}
        int init(int prev_num_nodes, int batch_size) const override {
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
        void destroy() const override {
            CHECK_CUDA_ERROR(cudaFree(gpu_W));
            CHECK_CUDA_ERROR(cudaFree(gpu_B));
            CHECK_CUDA_ERROR(cudaFree(gpu_A));
        }
    private:
        int num_nodes;
        Activation activation_func;
        float *gpu_A;
        float *gpu_W;
        float *gpu_B;
    };

    class Flatten : public Layer {
    public:
        Flatten(int num_nodes) : num_nodes(num_nodes) {}
        int init(int prev_num_nodes, int batch_size) const override {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_A, num_nodes * batch_size * sizeof(float)));
            return num_nodes;
        }
        void destroy() const override {
            CHECK_CUDA_ERROR(cudaFree(gpu_A));
        }
        void set_batch() {

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
            (*layer).destroy();
        }
        // Free handle
        cublasDestroy(handle);
    }
    void fit(float *X, int num_features, int num_samples, float *Y, int num_classes, int batch_size = 32, int epochs = 10) {
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
    cublasHandle_t handle;
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

        // Perform gradient descent for each mini-batch
        for (int i = 0; i < num_samples; i += batch_size) {
        
            // Get image and label batch
            batch_init(X, num_features, data_offsets, i);
            batch_init(Y, num_classes, data_offsets, i);

            // Copy from CPU to GPU
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_X, X, INPUT_SIZE * batch_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_Y, Y, OUTPUT_SIZE * batch_size * sizeof(float), cudaMemcpyHostToDevice));

            // Forward propagate to get activations A1 and A2
            forward_prop(gpu_X, gpu_A1, gpu_A2, gpu_W1, gpu_B1, gpu_W2, gpu_B2);

            // Print batch and preductions
            if (PRINT_BATCH_AND_PREDICTIONS) {
                print_batch_and_predictions(gpu_X, gpu_Y, gpu_A2);
            }

            // Add the number of correct predictions from the mini-batch
            int batch_correct = get_num_correct(gpu_A2, gpu_Y, OUTPUT_SIZE, batch_size);
            total_correct += batch_correct;

            // Back propagate to get dC/W1, dC/dB1, dC/dW2, dC/dB2
            back_prop(gpu_dC_dW1, gpu_dC_dB1, gpu_dC_dW2, gpu_dC_dB2, gpu_X, gpu_Y, gpu_A1, gpu_A2, gpu_W1, gpu_W2);

            // Update parameters
            update_params(gpu_W1, gpu_B1, gpu_W2, gpu_B2, gpu_dC_dW1, gpu_dC_dB1, gpu_dC_dW2, gpu_dC_dB2);

            // Update console
            cout << "\r";
            cout << "BATCH " << (i / batch_size) + 1 << "/" << NUM_BATCHES << " ";
            cout << "[";
            float percentage_completion = (((float) i / batch_size) + 1) / NUM_BATCHES;
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
            printf("%.3f", (float) batch_correct / batch_size);
            cout << " - TOTAL ACCURACY: ";
            printf("%.3f", (float) total_correct / (i + batch_size));
            cout << flush;;
        }
        cout << endl;

        delete[] data_offsets;
    }
};

int main() {
    DeepCore model;
    model.add(make_unique<DeepCore::Flatten>(784)); // input layer
    model.add(make_unique<DeepCore::Dense>(100, RELU)); // first layer
    model.add(make_unique<DeepCore::Dense>(10, SOFTMAX)); // second (output) layer
    model.compile(CROSS_ENTROPY);
    // model.fit();
    model.destroy();
    cout << "hello";
}