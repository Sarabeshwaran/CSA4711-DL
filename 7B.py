import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data[:, 5:6]  # Selecting a single feature (average number of rooms per dwelling)
y = boston.target

# Standardize the feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot the data
plt.scatter(X_scaled, y)
plt.title('Boston Housing Data - Average Number of Rooms vs. Median Value')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Value')
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

# Add a bias term to X
X_b = np.c_[np.ones((len(X_scaled), 1)), X_scaled]

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
plt.scatter(X_scaled, y)
plt.plot(X_scaled, X_b.dot(theta), color='red', label='Linear Regression')
plt.title('Linear Regression with Gradient Descent - Modified Data')
plt.xlabel('Average Number of Rooms (Standardized)')
plt.ylabel('Median Value')
plt.legend()
plt.show()
