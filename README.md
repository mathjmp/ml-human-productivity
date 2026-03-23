# 📈 Linear Regression from Scratch (Gradient Descent)

This project is a **from-scratch implementation of Linear Regression** using **Gradient Descent**, built without relying on machine learning libraries.

The goal is to deeply understand:

* How models learn
* How gradients are computed
* Why normalization is critical
* How training and inference pipelines work

---

## ⚠️ Performance Notice

This implementation is intentionally written **without NumPy vectorization** and relies on **pure Python loops** for all computations (hypothesis, cost, gradients, normalization).

### 🚨 What this means

* The algorithm is **significantly slower** than optimized implementations
* Time complexity is the same, but execution is slower due to:

  * Python loop overhead
  * Lack of low-level optimizations (C-based operations)
* Training on large datasets (like 10k+ rows) may take noticeable time

---

### 🧠 Why this was done

This project prioritizes **understanding over performance**.

By avoiding vectorization, every step of the algorithm is explicit:

* How predictions are computed
* How gradients are calculated
* How parameters are updated

This helps build strong intuition about how machine learning works internally.

---

### 🚀 Production note

In real-world applications, this would be replaced with:

* NumPy vectorized operations
* Or ML libraries that leverage optimized backends

This would result in:

* **10x–100x faster execution**
* Better scalability
* Cleaner code

---

### 💡 Recommendation

Once you fully understand this implementation, try rewriting it using NumPy vectorization to compare:

* Performance differences
* Code simplicity
* Numerical behavior

---

## 🧠 What This Project Covers

This implementation includes:

* ✅ Manual dataset loading from `.xlsx`
* ✅ Feature normalization (Z-score)
* ✅ Linear regression hypothesis function
* ✅ Cost function (Mean Squared Error)
* ✅ Gradient computation (derivatives)
* ✅ Gradient descent optimization
* ✅ Training visualization (cost vs iterations)
* ✅ Inference with proper denormalization

---

## 📂 Dataset Format

The dataset is expected to be an Excel file (`dataset.xlsx`) with the following structure:

| Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 | Target |
| --------- | --------- | --------- | --------- | --------- | ------ |
| x₁        | x₂        | x₃        | x₄        | x₅        | y      |

* First row is ignored (header)
* Features are columns `0–4`
* Target is column `5`

---

## ⚙️ How It Works

### 1. Data Loading

Reads the dataset using `openpyxl` and separates:

* `X` → input features
* `y` → target values

---

### 2. Feature Scaling (Z-score Normalization)

```python
# mean
mu = sum(x) / len(x)

# standard deviation
sigma = (sum((xi - mu) ** 2 for xi in x) / len(x)) ** 0.5

# normalization
z = [(xi - mu) / sigma for xi in x]
```

---

### 3. Hypothesis Function (Prediction)

```python
def hypothesis(x, w, b):
    total = 0
    for j in range(len(x)):
        total += x[j] * w[j]
    return total + b
```

Equivalent to:

```python
y_hat_i = sum(w[j] * x_i[j] for j in range(n)) + b
```

---

### 4. Cost Function (Mean Squared Error)

```python
def cost(X, y, w, b):
    m = len(X)
    total_error = 0

    for i in range(m):
        y_hat = hypothesis(X[i], w, b)
        total_error += (y_hat - y[i]) ** 2

    return (1 / (2 * m)) * total_error
```

---

### 5. Gradient Computation

```python
def compute_gradients(X, y, w, b):
    m = len(X)
    n = len(X[0])

    dw = [0] * n
    db = 0

    for i in range(m):
        y_hat = hypothesis(X[i], w, b)
        error = y_hat - y[i]

        db += error

        for j in range(n):
            dw[j] += error * X[i][j]

    # average gradients
    dw = [val / m for val in dw]
    db = db / m

    return dw, db
```

---

### 6. Gradient Descent Update Rule

```python
def update_parameters(w, b, dw, db, alpha):
    new_w = []
    for j in range(len(w)):
        new_w.append(w[j] - alpha * dw[j])

    new_b = b - alpha * db

    return new_w, new_b
```

---

### 7. Training Loop

```python
for iteration in range(num_iterations):
    dw, db = compute_gradients(X, y, w, b)
    w, b = update_parameters(w, b, dw, db, learning_rate)
```

---

### 8. Inference Pipeline

```python
# normalize input using training stats
x_norm = [(x[i] - mu_x[i]) / sigma_x[i] for i in range(len(x))]

# predict (normalized space)
y_norm = hypothesis(x_norm, w, b)

# denormalize output
y = y_norm * sigma_y + mu_y
```

---

## 🚀 How to Run

```bash
pip install openpyxl numpy matplotlib
python main.py
```

---

## 📊 Example Output

```bash
features=[...], output=...
test features=[...], output=...
cost=...
```

---

## 🔥 Key Learnings

* Gradient descent from scratch
* Feature scaling importance
* How models learn weights
* Difference between normalized and real values
* Internal mechanics of ML frameworks

---

## ⚠️ Notes

* Handles division-by-zero during normalization
* Learning rate must be tuned
* Works best with normalized features

---

## 💡 Future Improvements

* Vectorized implementation (NumPy)
* Mini-batch gradient descent
* Regularization (L1/L2)
* Polynomial regression
* Evaluation metrics (R², MAE)

---

## 🎯 Goal

This project focuses on **deep understanding over abstraction**.

Everything is implemented manually to build strong intuition
