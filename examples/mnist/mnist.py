"""
PyTorch equivalent of examples/mnist/mnist.cu, for a head-to-head efficiency
comparison against DeepCore's hand-written CUDA/cuBLAS implementation.

Architecture, hyperparameters, and training regime are matched as closely as
possible to model_3 in mnist.cu:
    Flatten(784) -> Dense(300, ReLU) -> Dense(100, ReLU) -> Dense(10, Softmax)
    loss = cross entropy, optimizer = plain (non-momentum) mini-batch SGD
    batch_size = 50, epochs = 5, learning_rate = 0.1

Reads the same raw IDX files DeepCore uses (./data) instead of torchvision's
dataset downloader, so both implementations train on identical bytes.
"""

import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA_DIR = Path(__file__).resolve().parent / "data"

BATCH_SIZE = 50
NUM_EPOCHS = 5
LEARNING_RATE = 0.1


def read_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic number for images file {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic number for labels file {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        # He init to match DeepCore's he_init() on its ReLU layers
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits; nn.CrossEntropyLoss applies softmax internally


@torch.no_grad()
def evaluate(model, X, Y, device, batch_size=BATCH_SIZE):
    model.eval()
    correct = 0
    for i in range(0, len(X) - len(X) % batch_size, batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = Y[i:i + batch_size].to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
    total = len(X) - len(X) % batch_size
    return correct, total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_X = torch.from_numpy(read_idx_images(DATA_DIR / "train-images.idx3-ubyte"))
    train_Y = torch.from_numpy(read_idx_labels(DATA_DIR / "train-labels.idx1-ubyte"))
    test_X = torch.from_numpy(read_idx_images(DATA_DIR / "t10k-images.idx3-ubyte"))
    test_Y = torch.from_numpy(read_idx_labels(DATA_DIR / "t10k-labels.idx1-ubyte"))

    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    num_train = len(train_X)
    num_batches = num_train // BATCH_SIZE

    print("COMPILED MODEL: 784 -> Dense(300, ReLU) -> Dense(100, ReLU) -> Dense(10, Softmax)")
    total_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        perm = torch.randperm(num_train)
        epoch_correct = 0

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        for b in range(num_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            xb = train_X[idx].to(device, non_blocking=True)
            yb = train_Y[idx].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_correct += (logits.argmax(dim=1) == yb).sum().item()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        train_acc = epoch_correct / (num_batches * BATCH_SIZE)
        val_correct, val_total = evaluate(model, test_X, test_Y, device)
        print(
            f"EPOCH {epoch}/{NUM_EPOCHS} - TRAIN ACCURACY: {epoch_correct}/{num_batches * BATCH_SIZE} "
            f"({train_acc * 100:.2f}%) - VALIDATION ACCURACY: {val_correct}/{val_total} "
            f"({val_correct / val_total * 100:.2f}%) - TIME ELAPSED: {epoch_time:.2f}s"
        )

    total_time = time.perf_counter() - total_start
    print(f">>> TRAINING COMPLETE. TOTAL TIME: {total_time:.2f}s")

    test_correct, test_total = evaluate(model, test_X, test_Y, device)
    print(f"TEST ACCURACY: {test_correct}/{test_total} ({test_correct / test_total * 100:.2f}%)")
    print(">>> TESTING COMPLETE.")


if __name__ == "__main__":
    main()
