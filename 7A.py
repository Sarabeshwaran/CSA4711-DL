import numpy as np
import matplotlib.pyplot as plt

# Function to minimize (you can replace this with any function)
def objective_function(x):
    return x**2 + 2*x + 1

# Derivative of the function (gradient)
def gradient(x):
    return 2*x + 2

def gradient_descent(learning_rate, num_iterations, initial_guess):
    history = []
    x = initial_guess

    for _ in range(num_iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        history.append(x)

    return history

# Parameters
learning_rate = 0.1
num_iterations = 20
initial_guess = -5

# Run gradient descent
path = gradient_descent(learning_rate, num_iterations, initial_guess)

# Plot the results
x_vals = np.linspace(-6, 2, 100)
y_vals = objective_function(x_vals)

plt.plot(x_vals, y_vals, label='Objective Function')
plt.scatter(path, [objective_function(x) for x in path], color='red', label='Gradient Descent Path')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
