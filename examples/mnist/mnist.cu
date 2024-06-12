#include <fstream>
#include "../../src/deepcore.cu"

#define TRAIN_LABELS_FILE_PATH R"(.\data\train-labels.idx1-ubyte)"
#define TRAIN_IMAGES_FILE_PATH R"(.\data\train-images.idx3-ubyte)"
#define TEST_LABELS_FILE_PATH R"(.\data\t10k-labels.idx1-ubyte)"
#define TEST_IMAGES_FILE_PATH R"(.\data\t10k-images.idx3-ubyte)"

#define LABEL_START 8
#define IMAGE_START 16

#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define NUM_PREDICT_IMAGES 10
 
using namespace std;

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

void print_batch_and_predictions(float *X, float *actual_Y, float *pred_Y, int size) {
    // For size number of labels/images, print them
    for (int i = 0; i < size; i++) {
        // Print image
        for (int j = 0; j < 784; j++) {
            if (j != 0 && j % 28 == 0) {
                cout << endl;
            }
            if (X[i*784 + j] < 0.5) {
                cout << "@.@"; // Represents dark pixel
            } else {
                cout << " . "; // Represents light pixel
            }
        }
        cout << endl;
        int predicted = -1, actual = -1;
        float predicted_max_prob = 0;
        // Get predicted label
        for (int j = 0; j < 10; j++) {
            if (pred_Y[i*10 + j] > predicted_max_prob) {
                predicted = j;
                predicted_max_prob = pred_Y[i*10 + j];
            }
        }
        // Get actual label
        for (int j = 0; j < 10 && actual == -1; j++) {
            if (actual_Y[i*10 + j] == 1) {
                actual = j; 
                break;
            }
        }
        cout << "PREDICTED LABEL: " << predicted << " - ACTUAL LABEL: " << actual << endl << endl;;

    }
}

float X[784 * NUM_TRAIN_IMAGES];
float Y[10 * NUM_TRAIN_IMAGES];

float test_X[784 * NUM_TEST_IMAGES];
float test_Y[10 * NUM_TEST_IMAGES];

float predict_X[784 * NUM_PREDICT_IMAGES];
float actual_Y[10 * NUM_PREDICT_IMAGES];
float predict_Y[10 * NUM_PREDICT_IMAGES];

int main() {
    // load the data
    get_images(X, 784, NUM_TRAIN_IMAGES, TRAIN_IMAGES_FILE_PATH, IMAGE_START);
    get_labels(Y, 10, NUM_TRAIN_IMAGES, TRAIN_LABELS_FILE_PATH, LABEL_START);

    get_images(test_X, 784, NUM_TEST_IMAGES, TEST_IMAGES_FILE_PATH, IMAGE_START);
    get_labels(test_Y, 10, NUM_TEST_IMAGES, TEST_LABELS_FILE_PATH, LABEL_START);

    get_images(predict_X, 784, NUM_PREDICT_IMAGES, TEST_IMAGES_FILE_PATH, IMAGE_START);
    get_labels(actual_Y, 10, NUM_PREDICT_IMAGES, TEST_LABELS_FILE_PATH, LABEL_START);

    // define hyperparameters
    int num_features = 784;
    int num_classes = 10;
    int batch_size = 50;
    int num_epochs = 20;
    float learning_rate = 0.1;

    // DeepCore model_1;
    // model_1.add(make_unique<DeepCore::Flatten>(784));
    // model_1.add(make_unique<DeepCore::Dense>(400, RELU));
    // model_1.add(make_unique<DeepCore::Dense>(10, SOFTMAX));
    // model_1.compile(CROSS_ENTROPY);
    // model_1.fit(X, num_features, NUM_TRAIN_IMAGES, Y, num_classes, batch_size, num_epochs, learning_rate, test_X, NUM_TEST_IMAGES, test_Y);
    // model_1.evaluate(test_X, num_features, NUM_TEST_IMAGES, test_Y, num_classes, batch_size);
    // model_1.save(R"(.\models\784-400-10.bin)");
    // model_1.destroy();

    // DeepCore model_2;
    // model_2.read(R"(.\models\784-400-10.bin)");
    // model_2.evaluate(test_X, num_features, NUM_TEST_IMAGES, test_Y, num_classes, batch_size);
    // model_2.predict(predict_X, num_features, NUM_PREDICT_IMAGES, predict_Y, num_classes);
    // print_batch_and_predictions(predict_X, actual_Y, predict_Y, NUM_PREDICT_IMAGES);
    // model_2.destroy();

    DeepCore model_3;
    model_3.add(make_unique<DeepCore::Flatten>(784));
    model_3.add(make_unique<DeepCore::Dense>(300, RELU));
    model_3.add(make_unique<DeepCore::Dense>(100, RELU));
    model_3.add(make_unique<DeepCore::Dense>(10, SOFTMAX));
    model_3.compile(CROSS_ENTROPY);
    model_3.fit(X, num_features, NUM_TRAIN_IMAGES, Y, num_classes, batch_size, num_epochs, learning_rate, test_X, NUM_TEST_IMAGES, test_Y);
    model_3.evaluate(test_X, num_features, NUM_TEST_IMAGES, test_Y, num_classes, batch_size);
    model_3.save(R"(.\models\784-300-100-10.bin)");
    model_3.destroy();

    DeepCore model_4;
    model_4.read(R"(.\models\784-300-100-10.bin)");
    model_4.evaluate(test_X, num_features, NUM_TEST_IMAGES, test_Y, num_classes, batch_size);
    model_4.predict(predict_X, num_features, NUM_PREDICT_IMAGES, predict_Y, num_classes);
    print_batch_and_predictions(predict_X, actual_Y, predict_Y, NUM_PREDICT_IMAGES);
    model_4.destroy();
    return 0;
}