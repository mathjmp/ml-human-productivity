import openpyxl
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    wb = openpyxl.load_workbook("dataset.xlsx")
    sheet = wb.active
    x = []
    y = []
    iterations = 0
    for row in sheet.iter_rows(values_only = True):
        
        if iterations == 0:
            iterations += 1
            continue
        
        x.append(row[0:5])
        y.append(row[5])
    return x, y

def hypothesis(x, w, b):
    m = len(x)
    total = 0
    for i in range(m):
        total += x[i] * w[i]
    result = total + b
    return result

def cost(x, y, w, b):
    m = len(x)
    error_ac = 0
    for i in range(m):
        prediction = hypothesis(x[i], w, b)
        error = (prediction - y[i]) ** 2
        error_ac += error
    total_cost = (1 / (2 * m)) * error_ac
    return total_cost

def take_cost_averages(weight_costs, m):
    n = len(weight_costs)
    dd_weights = []
    for i in range(n):
        average_cost = weight_costs[i] / m
        dd_weights.append(average_cost)
    return dd_weights

def take_parameters_derivative(x, y, w, b):
    m = len(x)
    n = len(x[0])
    bias_cost = 0
    weight_costs = [0] * n
    for i in range(m):
        prediction = hypothesis(x[i], w, b)
        error = prediction - y[i]
        bias_cost += error
        for j in range(n):
            weight_costs[j] += error * x[i][j]
    dd_bias = bias_cost / m
    dd_weights = take_cost_averages(weight_costs, m)
    return dd_weights, dd_bias

def update_model_parameters(w, b, alpha, dd_weights, dd_bias):
    m = len(dd_weights)
    new_weights = []
    for i in range(m):
        new_weight = w[i] - alpha * dd_weights[i]
        new_weights.append(new_weight)
    new_bias = b - alpha * dd_bias
    return new_weights, new_bias

def compute_gradient(x, y, w, b, alpha, iterations = 2000):
    cost_hist = []
    for _ in range(iterations):
        c = cost(x, y, w, b)
        cost_hist.append(c)
        dd_weights, dd_bias = take_parameters_derivative(x, y, w, b)
        new_weights, new_bias = update_model_parameters(w, b, alpha, dd_weights, dd_bias)
        w = new_weights
        b = new_bias
    plt.plot(np.arange(len(cost_hist)), cost_hist)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.show()
    return w, b   

def mean(x):
    m = len(x)
    total = 0
    for i in range(m): 
        total += x[i]
    average = total / m
    return average

def std(x):
    m = len(x)
    total = 0
    average = mean(x)
    for i in range(m):
        total += (x[i] - average) ** 2
    deviation = (total / m) ** (1/2)
    return deviation

def matrix_mean(x):
    m = len(x)
    n = len(x[0])
    averages = []
    for j in range(n):
        total = 0
        for i in range(m):
            total += x[i][j]
        average = total / m
        averages.append(average)
    return averages

def matrix_std(x):
    m = len(x)
    n = len(x[0])
    averages = matrix_mean(x)
    deviations = []
    for j in range(n):
        total = 0
        for i in range(m):
            total += (x[i][j] - averages[j]) ** 2
        deviation = (total / m) ** (1/2)
        deviations.append(deviation)
    return deviations

def vector_zscore_normalization(x):
    average = mean(x)
    st_deviation = std(x)
    rescaled_vector = []
    m = len(x)
    for i in range(m):
        if st_deviation == 0:
            rescaled_value = 0
        else:
            rescaled_value = (x[i] - average) / st_deviation
        rescaled_vector.append(rescaled_value)
    return rescaled_vector

def matrix_zscore_normalization(x):
    averages = matrix_mean(x)
    st_deviations = matrix_std(x)
    rescaled_matrix = []
    m = len(x)
    n = len(x[0])
    for i in range(m):
        row = []
        for j in range(n):
            if st_deviations[j] == 0:
                rescaled_value = 0
            else:
                rescaled_value = (x[i][j] - averages[j]) / st_deviations[j]
            row.append(rescaled_value)
        rescaled_matrix.append(row)
    return rescaled_matrix

train_x, train_y = load_dataset()
train_x_mean = matrix_mean(train_x)
train_x_std = matrix_std(train_x)
train_y_mean = mean(train_y)
train_y_std = std(train_y)
x = matrix_zscore_normalization(train_x)
y = vector_zscore_normalization(train_y)
weights = [0, 0, 0, 0, 0]
bias = 0
learning_rate = 0.01
weights, bias = compute_gradient(x, y, weights, bias, learning_rate)
index = 0
target = train_x[index]
target_normalized = []

for i in range(len(target)):
    value = (target[i] - train_x_mean[i]) / train_x_std[i]
    target_normalized.append(value)

prediction_normalized = hypothesis(target_normalized, weights, bias)
prediction = prediction_normalized * train_y_std + train_y_mean

print(f"features={train_x[index]}, output={train_y[index]}")
print(f"test features={target}, output={prediction}")
print(f"cost={cost(x, y, weights, bias)}")