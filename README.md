# DeepCore
DeepCore is a C++ neural network library that leverages CUDA for accelerated tensor operations. Its user-friendly and intuitive API draws inspiration from Keras's Tensorflow API. DeepCore is built from the ground up, from the fundamental tensor operations that power artificial neural networks. The source code aims to aid in understanding the mathematics behind neural networks, implementing concepts such as forward propagation, backpropagation, gradient descent, Jacobian computation, and the chain rule.

## Installation and Usage
Clone the repository into a local directory. Ensure you have the [CUDA Toolkit 12.5](https://developer.nvidia.com/cuda-toolkit), the [Microsoft Visual Studio Compiler](https://visualstudio.microsoft.com/) (cl.exe), and other relevant dependencies installed. Ensure your `PATH` is configured correctly.

To use the DeepCore library, simply include the `./src/deepcore.cu` file into your project source code. You will then be able to access all of the functions provided by the library. Compile your project with the `nvcc` compiler. Refer to the `./examples/` directory for practical examples of ultilizing DeepCore. 

## DeepCore API Documentation

### The `Layer` class
An abstract base class for neural network layers.
**Usage**
- Use the derived classes `Dense` and `Flatten` as DeepCore neural network layers.
#### The `Dense` subclass
A dense (fully connected) neural network layer. Trainable parameters consist of a weights and biases matrix. Non-trainable parameters consist of the size of the dense layer (number of nodes) and its activation function.
##### `Dense(int num_nodes, Activation activation_func)` constructor
Creates a dense layer consisting of `num_nodes` nodes and `activation_func` activation function.
**Arguments**
- `int num_nodes`: The size of number of nodes of the Dense layer.
- `Activation activation_func`: The activation function of the Dense layer. Currently implemented activation functions include `RELU` and `SOFTMAX`

**Usage**
```cpp
model.add(make_unique<DeepCore::Dense>(300, RELU));
model.add(make_unique<DeepCore::Dense>(10, SOFTMAX));
```
#### The `Flatten` subclass
A flattening neural network layer. Serves as an input layer into a dense neural network. Does not contain trainable parameters, with its only non-trainable parameter being its size (number of nodes).
##### `Flatten(int num_nodes)` constructor
Creates a flattening layer consisting of `num_nodes` nodes.
**Arguments**
- `int num_nodes`: The size of number of nodes of the Dense layer.

**Usage**
```cpp
model.add(make_unique<DeepCore::Flatten>(num_features));
model.add(make_unique<DeepCore::Flatten>(784));
```

### The `DeepCore` class
A model grouping several `Layer` objects into an object with training/prediction functionalities.
A DeepCore model is instantiated by simply declaring an object its type:
```cpp
DeepCore model;
```
Once you've declared the model, you can add layers using `add()`, configure it with `compile()`, and then train the model using `fit()`. Alternatively, if you have a model saved in a file, you can load it using `read()`. Once loaded, you can use the model for predictions with `predict()`, evaluate its performance on test data with `evaluate()`, or save it back to a file with `save()`. When you're done using the model, remember to free up program resources by calling `destroy()`. An snippet of using DeepCore is below; refer to `./examples/mnist` for the entire example.
```cpp
// Training, evaluating, and saving a model to file
DeepCore model;
model.add(make_unique<DeepCore::Flatten>(784));
model.add(make_unique<DeepCore::Dense>(300, RELU));
model.add(make_unique<DeepCore::Dense>(100, RELU));
model.add(make_unique<DeepCore::Dense>(10, SOFTMAX));
model.compile(CROSS_ENTROPY);
model.fit(X, num_features, NUM_TRAIN_IMAGES, Y, num_classes, batch_size, num_epochs, learning_rate, test_X, NUM_TEST_IMAGES, test_Y);
model.evaluate(test_X, num_features, NUM_TEST_IMAGES, test_Y, num_classes, batch_size);
model.save(R"(.\models\784-300-100-10.bin)");
model.destroy();

// Reading a model from file, evaluting it, and using it to make predictions
DeepCore model;
model.read(R"(.\models\784-300-100-10.bin)");
model.evaluate(test_X, num_features, NUM_TEST_IMAGES, test_Y, num_classes, batch_size);
model.predict(predict_X, num_features, NUM_PREDICT_IMAGES, predict_Y, num_classes);
print_batch_and_predictions(predict_X, actual_Y, predict_Y, NUM_PREDICT_IMAGES);
model.destroy();
```
#### `void add()` method
Adds a layer to the model.
**Arguments**
- `std::unique_ptr<DeepCore::Layer> layer`: A unique pointer managing the Layer to be added to the model.

**Usage**
```cpp
model.add(make_unique<DeepCore::Flatten>(784));
model.add(make_unique<DeepCore::Dense>(300, RELU));
```
**Output**
```
None
```
#### `void compile()` method
Configures and initializes the model with the provided loss function. 
**Arguments**
- `Loss loss_func`: The specific loss function the model should use. Currently implemented loss functions include `CROSS_ENTROPY` and `MSE`.

**Usage**
```cpp
model.compile(MSE);
model.compile(CROSS_ENTROPY);
```
**Output**
```
None
```
#### `void fit()` method
Fits the model to the input data (features) `X` and target data (labels) `Y` using stochastic gradient descent. Optionally evaluates performance on a validation set after each epoch. 
Note: The model must have already been compiled with `compile()`.
**Arguments**
- `float *X`: The input data matrix to train the model. Should be of dimension `num_features`×`num_samples`.
- `int num_features`: Number of features (input dimensions) of the dataset.
- `int num_samples`: Number of samples (data points) in the dataset.
- `float *Y`: The target data matrix to train the model. Should be of dimension `num_classes`×`num_samples`.
- `int num_classes`: Number of classes (output dimensions) of the dataset.
- `int batch_size = 50`: Number of samples per gradient update. Defaults to `50`.
- `int epochs = 10`: Number of epochs (iterations over the dataset) to train the model. Defaults to `10`.
- `float learning_rate = 0.1`: Learning rate of the model for gradient descent. Defaults to `0.1`.
- `float *validation_X = nullptr`: The validation input data matrix (optional).
- `int num_validation = -1`: Number of samples in the validation set (optional).
- `float *validation_Y = nullptr`: The validation target data (optional).

**Usage**
```cpp
model.fit(X, num_features, NUM_TRAIN_IMAGES, Y, num_classes, batch_size, num_epochs, learning_rate, test_X, NUM_TEST_IMAGES, test_Y);
model.fit(X, 784, 60000, Y, 10, 50, 5, 0.1, test_X, 10000, test_Y);
```
**Output**
```
COMPILED MODEL:
______________________________________________________________________
 Layer (type)                 Output Shape                  Param #   
======================================================================
 Flatten                      (50, 784, 1)                  0         
 Dense                        (50, 300, 1)                  235500    
 Dense                        (50, 100, 1)                  30100     
 Dense                        (50, 10, 1)                   1010      
======================================================================
Total trainable params: 266610
______________________________________________________________________

EPOCH 1/20
BATCH 1200/1200 [================================] - BATCH ACCURACY: 0.940 - TOTAL ACCURACY: 0.922
TRAIN ACCURACY: 55313/60000 (92.19%) - VALIDATION ACCURACY: 9588/10000 (95.88%) - TIME ELAPSED: 28.15s - ETA: 00:01:52

EPOCH 2/20
BATCH 1200/1200 [================================] - BATCH ACCURACY: 1.000 - TOTAL ACCURACY: 0.968
TRAIN ACCURACY: 58102/60000 (96.84%) - VALIDATION ACCURACY: 9674/10000 (96.74%) - TIME ELAPSED: 34.58s - ETA: 00:01:42

EPOCH 3/20
BATCH 1200/1200 [================================] - BATCH ACCURACY: 0.980 - TOTAL ACCURACY: 0.978
TRAIN ACCURACY: 58700/60000 (97.83%) - VALIDATION ACCURACY: 9715/10000 (97.15%) - TIME ELAPSED: 29.57s - ETA: 00:00:58

EPOCH 4/20
BATCH 1200/1200 [================================] - BATCH ACCURACY: 1.000 - TOTAL ACCURACY: 0.985
TRAIN ACCURACY: 59073/60000 (98.45%) - VALIDATION ACCURACY: 9715/10000 (97.15%) - TIME ELAPSED: 30.85s - ETA: 00:00:30

EPOCH 5/20
BATCH 1200/1200 [================================] - BATCH ACCURACY: 0.980 - TOTAL ACCURACY: 0.989
TRAIN ACCURACY: 59353/60000 (98.92%) - VALIDATION ACCURACY: 9756/10000 (97.56%) - TIME ELAPSED: 28.64s - ETA: 00:00:00
>>> TRAINING COMPLETE.
```
#### `void evaluate()` method
Evaluates a trained model on the input data (features) `X` and target data (labels) `Y`. 
Note: The model must have been trained with `fit()` or read from file with `read()`.
**Arguments**
- `float *test_X`: The input data matrix to test the model. Should be of dimension `num_features`×`num_samples`.
- `int num_features`: Number of features (input dimensions) of the dataset.
- `int num_test`: Number of samples (data points) in the dataset.
- `float *test_Y`: The target data matrix to test the model. Should be of dimension `num_classes`×`num_samples`.
- `int num_classes`: Number of classes (output dimensions) of the dataset.
- `int batch_size = 50`: Number of samples to process at a time. Defaults to `50`.

**Usage**
```cpp
model.evaluate(test_X, num_features, NUM_TEST_IMAGES, test_Y, num_classes, batch_size);
model.evaluate(test_X, 784, 10000, test_Y, 10, 50);
```
**Output**
```
BATCH 200/200 [================================] - TEST ACCURACY: 9756/10000 (97.56%)
>>> TESTING COMPLETE.
```
#### `void predict()` method
Predicts target data (labels) `predict_Y` of the input data (features) `predict_X` using a trained model. 
Note: The memory for `predict_Y` must be allocated and managed by the caller. The model must have been trained with `fit()` or read from file with `read()`.
**Arguments**
- `float *predict_X`: The input data matrix to test the model. Should be of dimension `num_features`×`num_samples`.
- `int num_features`: Number of features (input dimensions) of the dataset.
- `int num_samples`: Number of samples (data points) in the dataset.
- `float *predict_Y`: The target data matrix to test the model. Should be of dimension `num_classes`×`num_samples`.
- `int num_classes`: Number of classes (output dimensions) of the dataset.

**Usage**
```cpp
model.predict(predict_X, num_features, NUM_PREDICT_IMAGES, predict_Y, num_classes);
model.predict(predict_X, 784, 50, predict_Y, 10);
```
**Output**
```
>>> PREDICTION COMPLETE.
```
#### `void save()` method
Saves all the information (layer information, weights, biases) about the model to file `path`. The model can be recovered with `read()`.
**Arguments**
- `string path`: Path of the file for the model to be saved into.

**Usage**
```cpp
model.save(R"(.\models\784-300-100-10.bin)");
```
**Output**
```
SAVING MODEL TO .\models\784-300-100-10.bin
>>> SAVING COMPLETE.
```
#### `void read()` method
Reads and loads all the information (layer information, weights, biases) about the model from file `path`. 
Note: After `read()` is called the model does not need to be compiled with `compile()`.
**Arguments**
- `string path`: Path of the file for the model to be loaded from.

**Usage**
```cpp
model.read(R"(.\models\784-300-100-10.bin)");
```
**Output**
```
READING MODEL FROM .\models\784-300-100-10.bin
MODEL SPECIFICATIONS:
______________________________________________________________________
 Layer (type)                 Output Shape                  Param #   
======================================================================
 Flatten                      (n, 784, 1)                   0         
 Dense                        (n, 300, 1)                   235500    
 Dense                        (n, 100, 1)                   30100     
 Dense                        (n, 10, 1)                    1010      
======================================================================
Total trainable params: 266610
______________________________________________________________________
>>> READING COMPLETE.
```
#### `void destroy()` method
Frees the rest of the memory associated with the model that was allocated from `compile()` or `read()`.
**Arguments**
- `None`

**Usage**
```cpp
model.destroy();
```
**Output**
```
None
```
## Stochastic Gradient Descent Algorithm
Given a set of input data $X$, a set of labels $Y$, a set of parameters $\Theta$, and a model $\hat{Y}(X,\Theta)$, we want to find a set of parameters $\hat{\Theta}$ that minimizes a cost function $C(Y, \hat{Y})$. 

To do this, we must find the gradient of the cost function $\nabla C$, which is a vector of all the partial derivatives of $C$ relative to every parameter $\theta \in \Theta$. The gradient tells us the sensitivity of the cost $C$ relative to each parameter $\theta \in \Theta$. 

In simple words, it tells us the factor by which the cost $C$ changes with respect to a change in a given parameter $\theta$. For example, if for a given parameter $\theta$, the partial derivative of $C$ with respect to $\theta$ is $\frac{dC}{d\theta} = 2$, then a change of parameter $\theta$ by $-0.1$ would result in a change in the cost $C$ by $-0.1*2 = -0.2$. This provides us with a direct method of minimizing the cost $C$, by updating each parameter as follows: $\theta_{new} = \theta_{old}-\eta\frac{dC}{d\theta}$, where $\eta$ is the learning rate hyperparameter and $\frac{dC}{d\theta}$ comes from $\nabla C$. 

All that's left is to compute the gradient vector $\nabla C$, and we'll be able to update our parameters to minimize $C$. But how exactly do we compute the gradient?

Consider a simplified example. Suppose we had a single data sample, represented by a vector $\vec{x}$, and its label, represented by a vector $\vec{y}$. Suppose our model consists of a `Flatten` layer, a hidden `Dense (ReLU)` layer, and an output `Dense (Softmax)` layer we will call $L^{(0)}$, $L^{(1)}$, and $L^{(2)}$ respectively. Each layer in a neural network consists of activation vectors we will call $\vec{a^{(0)}}$, $\vec{a^{(1)}}$, and $\vec{a^{(2)}}$ respectively. The activation of our `Flatten` layer will simply be our input data $\vec{a^{(0)}} = \vec{x}$. The activations of `Dense` layers are defined as $\vec{a^{(L)}}=f(\vec{z^{(L)}})$, where $f$ is some activation function and $\vec{z^{(L)}} = W^{(L)}\vec{a^{(L-1)}}+\vec{b^{(L)}})$, where $L$ is the current layer, $L-1$ is the previous layer, $W^{(L)}$ is the matrix of weights associated with layer $L$, and $\vec{b^{(L)}}$ is the vector of biases associated with layer $L$. These weights and biases are the parameters we want to "tune" in order to minimize the cost $C$. Thus we are interested in $\frac{dC}{dW^{(L)}}$ and $\frac{dC}{d\vec{b^{(L)}}}$ for each layer $L$. 

Since the output layer undergoes `Softmax`, our model outputs probabilites of a given output class $i \in classes$ in the vector $\vec{a^{(2)}}$. Then, our cost function $C$ will be in terms of the vector $\vec{a^{(2)}}$ and $\vec{y}$. Suppose that our cost function $C$ is defined as $`C(\vec{a^{(2)}},\vec{y})=-\sum_{i \in classes}{y_i*log({a^{(2)}_i})}`$, known as `Cross-Entropy Loss`. Then, the derivative of the cost function $C$ with respect to activations in the second layer is a Jacobian $\frac{dC}{d\vec{a^{(2)}}}$, where $`\frac{dC}{da^{(2)}_i}=-y_i*\frac{1}{a^{(2)}_i}`$. Similarly, $`\frac{d\vec{a^{(2)}}}{d\vec{z^{(2)}}}`$ is a Jacobian where $`\frac{d\vec{a^{(2)}_i}}{d\vec{z^{(2)}_j}}=a^{(2)}_i(\delta_{ij}-a^{(2)}_j)`$, where 
$`\delta_{ij} = \begin{cases}
1 & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}`$. This comes from the activations of the second layer $`\vec{a^{(2)}}`$ being defined as $`\vec{a^{(2)}}=f(\vec{z^{(2)}})`$ where $`f(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}`$ is the `softmax` function.

## Resources
#### CUDA Runtime
DeepCore uses NVIDIA's [CUDA](https://developer.nvidia.com/cuda-toolkit) parallel computing platform, allowing for parallelization of tasks such as computing Jacobians or applying activation functions to matrices.

#### cuBLAS
DeepCore uses NVIDIA's [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html), a lightweight library built on top of NVIDIA's CUDA runtime, dedicated to performing basic linear algebra operations. DeepCore extends cuBLAS's functionality by implementing a tensor multiplication function with cuBLAS's batched matrix multiplication function.
#### MNIST Dataset
DeepCore was evaluated using the [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset, a popular benchmark dataset for digit recognition tasks. The [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset consists of 60,000 training images and 10,000 testing images of handwritten digits. Each image is a grayscale image of size 28x28 pixels.
## Personal word
As a rising junior undergrad pursuing Computer Science and Applied Mathematics, I created this project out of personal interest to gain a deeper understanding of the underlying mathematics behind artificial neural networks, and to achieve my long-awaited goal of learning GPU/CUDA programming. Throughout the journey, I learned C++ OOP, CUDA programming, cuBLAS, and solidified my understanding of backpropagation. Other than using cuBLAS for basic matrix operations, everything was implemented from scratch - starting literally from a scratch sheet of paper containing mathematical constructs to a realized program.  Huge thanks to the [Stanford CS224N NLP with Deep Learning](https://youtu.be/X0Jw4kgaFlg?si=O9N0UqGuZ3VsixyQ) course and [3Blue1Brown's Deep Learning Series](https://youtu.be/tIeHLnjs5U8?si=jFUHxcMr3w0KXxM2) for being incredible free and online resources. If you have any suggestions/comments, feel free to reach out!