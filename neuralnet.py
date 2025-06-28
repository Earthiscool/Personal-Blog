import numpy as np
import matplotlib.pyplot as plt

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Feed Forward
def feed_forward(x, w1, w2):
    z1 = x.dot(w1)        # Input to hidden layer
    a1 = sigmoid(z1)      # Output of hidden layer
    z2 = a1.dot(w2)       # Input to output layer
    a2 = sigmoid(z2)      # Output of output layer
    return a2

# Loss Function (Mean Squared Error)
def loss(predicted, actual):
    mse = np.square(predicted - actual)
    return np.sum(mse) / len(y)

# Generate random weights
def generate_weights(rows, cols):
    return np.random.randn(rows, cols)

# Backpropagation
def back_propagation(x, y_true, w1, w2, alpha):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)

    # Calculate error
    d2 = a2 - y_true
    d1 = np.multiply(w2.dot(d2.T).T, a1 * (1 - a1))

    # Gradients
    w1_adjustment = x.T.dot(d1)
    w2_adjustment = a1.T.dot(d2)

    # Update weights
    w1 -= alpha * w1_adjustment
    w2 -= alpha * w2_adjustment

    return w1, w2

# Training function
def train(x, y, w1, w2, alpha=0.01, epochs=10):
    accuracy = []
    loss_values = []

    for epoch in range(epochs):
        epoch_loss = []

        for i in range(len(x)):
            output = feed_forward(x[i], w1, w2)
            epoch_loss.append(loss(output, y[i]))
            w1, w2 = back_propagation(x[i], y[i], w1, w2, alpha)

        avg_loss = sum(epoch_loss) / len(x)
        epoch_accuracy = (1 - avg_loss) * 100

        print(f"Epoch {epoch + 1}: Accuracy = {epoch_accuracy:.2f}%")

        accuracy.append(epoch_accuracy)
        loss_values.append(avg_loss)

    return accuracy, loss_values, w1, w2

# Prediction function
def predict(x, w1, w2):
    output = feed_forward(x, w1, w2)
    predicted_label = np.argmax(output)

    if predicted_label == 0:
        print("Image is of letter A.")
    elif predicted_label == 1:
        print("Image is of letter B.")
    else:
        print("Image is of letter C.")

    plt.imshow(x.reshape(5, 6))
    plt.show()

# --- Main Program ---

# Dataset
a = [
    0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1
]

b = [
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0
]

c = [
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0
]

# Labels (One-hot encoded)
y = np.array([
    [1, 0, 0],  # A
    [0, 1, 0],  # B
    [0, 0, 1]   # C
])

# Preparing input data
x = np.array([
    np.array(a).reshape(1, 30),
    np.array(b).reshape(1, 30),
    np.array(c).reshape(1, 30)
])

# Initialize weights
w1 = generate_weights(30, 5)
w2 = generate_weights(5, 3)

# Train the model
accuracy, loss_values, w1, w2 = train(x, y, w1, w2, alpha=0.1, epochs=100)

# Predict on a sample
predict(x[1], w1, w2)  # Predicting the second image (B)

# Plotting accuracy
plt.plot(accuracy)
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.show()

# Plotting loss
plt.plot(loss_values)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Visualize the letter 'A'
plt.imshow(np.array(a).reshape(5, 6))
plt.title('Letter A')
plt.show()

# Show input data and labels
print("Input Data (x):\n", x)
print("\nLabels (y):\n", y)
