import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Plot the data
plt.scatter(X, y)
plt.title('Synthetic Data for Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Function to compute the mean squared error (cost)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

# Gradient Descent function
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)

    for iteration in range(num_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient
        cost_history[iteration] = compute_cost(X, y, theta)

    return theta, cost_history

# Add a bias term to X (intercept term)
X_b = np.c_[np.ones((100, 1)), X]

# Initial parameters
theta_initial = np.random.randn(2, 1)

# Hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Run gradient descent
theta, cost_history = gradient_descent(X_b, y, theta_initial, learning_rate, num_iterations)

# Plot the cost history
plt.plot(range(1, num_iterations + 1), cost_history, color='blue')
plt.title('Cost History over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Plot the data and the best-fitting line
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red', label='Linear Regression')
plt.title('Linear Regression with Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
