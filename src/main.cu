#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <numeric>

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

using namespace std;

void forward_prop() {
    
}

void back_prop() {
    
}

void update_params() {

}

void get_label_batch(float (&Y)[OUTPUT_SIZE*BATCH_SIZE], const int offsets[NUM_TRAIN_IMAGES], int index) {
    ifstream labels_file(TRAIN_LABELS_FILE_PATH, ios::in | ios::binary);
    if (labels_file.is_open()) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            labels_file.seekg(LABEL_START + offsets[index + i]);
            int label;
            labels_file.read((char *) &label, 1);
            Y[label + i*10] = 1;
        }
        labels_file.close();
    } else {
        cout << "Error: Failed to open file " << TRAIN_LABELS_FILE_PATH << endl;
        exit(1);
    }
}

void get_image_batch(float (&X)[INPUT_SIZE*BATCH_SIZE], const int offsets[NUM_TRAIN_IMAGES], int index) {
    ifstream images_file(TRAIN_IMAGES_FILE_PATH, ios::in | ios::binary);
    if (images_file.is_open()) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            images_file.seekg(IMAGE_START + 784 * offsets[index + i]);
            for (int j= 0; j < 784; j++) {
                int value = 0;
                images_file.read((char *) &value, 1);
                X[j + i*784] = value/255.0; // Transform value from range [0, 255] to range [0, 1]
            }
        }
        images_file.close();
    } else {
        cout << "Error: Failed to open file " << TRAIN_IMAGES_FILE_PATH << endl;
        exit(1);
    }
}

void print_batch(float (&X)[INPUT_SIZE*BATCH_SIZE], float (&Y)[OUTPUT_SIZE*BATCH_SIZE]) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Print label
        cout << "The following number is: ";
        for (int label = 0; label < 10; label++) {
            if (Y[label + i*10] == 1) {
                cout << label << "\n";
                break;
            }
        }
        // Print image
        for (int value = 0; value < 784; value++) {
            if (value != 0 && value % 28 == 0) {
                cout << "\n";
            }
            if (X[value + i*784] < 0.5) {
                cout << "@.@"; // Represents dark pixel
            } else {
                cout << " . "; // Represents light pixel
            }
        }
        cout << "\n";
    }
}

void gradient_descent(float (&W1)[L1_SIZE*INPUT_SIZE], float (&B1)[L1_SIZE], float (&W2)[OUTPUT_SIZE*L1_SIZE], float (&B2)[OUTPUT_SIZE]) {
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

        float dW1[L1_SIZE*INPUT_SIZE];
        float dB1[L1_SIZE];
        float dW2[OUTPUT_SIZE*L1_SIZE];
        float dB2[OUTPUT_SIZE];

        float X[INPUT_SIZE*BATCH_SIZE];
        float Y[OUTPUT_SIZE*BATCH_SIZE];

        // Get image and label batch
        get_image_batch(X, data_offsets, i);
        get_label_batch(Y, data_offsets, i);

        // Optionally print out the training labels and images
        print_batch(X, Y);

        // // Forward propagate to get Z1, A1, Z2, and A2
        // states_and_activations fp = forward_prop(X, *W1, *B1, *W2, *B2);

        // // Back propagate to get dW1, dB1, dW2, dB2
        // derivatives bp = back_prop(X, Y, fp.Z1, fp.A1, fp.Z2, fp.A2, *W2);

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

void he_init(float *weights, int m, int n) {
    random_device rd; // Random device for seeding
    mt19937 gen(rd()); // Mersenne Twister generator
    normal_distribution<> d(0, sqrt(2.0 / n)); // He normal distribution
    for (int i = 0; i < m*n; i++) {
        weights[i] = d(gen);
    }
}

int main() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    float W1[L1_SIZE*INPUT_SIZE];
    float B1[L1_SIZE]{0};
    float W2[OUTPUT_SIZE*L1_SIZE];
    float B2[OUTPUT_SIZE]{0};

    // Initialize weights with He initialization method
    he_init(W1, L1_SIZE, INPUT_SIZE);
    he_init(W2, OUTPUT_SIZE, L1_SIZE);
    
    gradient_descent(W1, B1, W2, B2);

    return 0;
}