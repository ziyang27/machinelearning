# Build Your Own Linear Regression

## Overview
This project demonstrates the implementation of linear regression from scratch using Python, without relying on machine learning libraries like scikit-learn. The goal is to understand the fundamental mathematics and algorithms behind linear regression by building it step-by-step using gradient descent optimization.

## Video Reference
This project is based on the tutorial: [How to implement Linear Regression from scratch with Python](https://www.youtube.com/watch?v=QlbyGPVaRSE&ab_channel=InfiniteCodes)

## Objectives
- Implement linear regression algorithm from scratch
- Understand the mathematical foundations of gradient descent
- Build intuition for cost functions and optimization
- Compare custom implementation with library versions
- Demonstrate mastery of fundamental ML concepts

## Mathematical Foundation
**Linear Regression Equation:**
```
y = w*x + b
```
Where:
- `y` = predicted output
- `w` = weight (slope)
- `b` = bias (y-intercept)
- `x` = input feature

**Cost Function (Mean Squared Error):**
```
J(w,b) = (1/2m) * Σ(y_pred - y_actual)²
```

**Gradient Descent Updates:**
```
w = w - α * (∂J/∂w)
b = b - α * (∂J/∂b)
```
## Implementation Methodology

### 1. Algorithm Structure
**Core Components:**
- Weight initialization
- Bias initialization
- Prediction function
- Cost calculation
- Gradient computation
- Parameter updates

### 2. Training Phase
**Step-by-Step Process:**

1. **Parameter Initialization**:
   ```python
   weight = 0.0
   bias = 0.0
   ```

2. **For Each Training Iteration**:
   - **Prediction**: Calculate `pred_y = w*x + b` for each data point
   - **Error Calculation**: Compute difference between predicted and actual values
   - **Gradient Descent**: Update weight and bias using calculated gradients
   - **Repeat**: Continue for n iterations until convergence

3. **Gradient Calculations**:
   - **Weight Gradient**: `dw = (1/m) * Σ(x * (pred_y - y_actual))`
   - **Bias Gradient**: `db = (1/m) * Σ(pred_y - y_actual)`

4. **Parameter Updates**:
   - **New Weight**: `w = w - learning_rate * dw`
   - **New Bias**: `b = b - learning_rate * db`

### 3. Testing Phase
**Prediction Process:**
- Given a new data point
- Apply the learned equation: `pred_y = w*x + b`
- Return the predicted value

## Technical Skills Demonstrated
- **Mathematical Implementation**: Converting equations to code
- **Gradient Descent**: Understanding and implementing optimization
- **Algorithm Design**: Building ML algorithms from first principles
- **Performance Analysis**: Tracking cost function during training
- **Vectorization**: Efficient numpy operations (if applicable)

## Key Learnings
- **Gradient Descent Mechanics**: How optimization algorithms work step-by-step
- **Mathematical Intuition**: Understanding the "why" behind linear regression
- **Parameter Sensitivity**: Impact of learning rate and iterations
- **Convergence Behavior**: How algorithms reach optimal solutions
- **Implementation Challenges**: Numerical stability and efficiency considerations

## Conclusion
This project successfully implements linear regression from scratch, providing deep insights into the mathematical foundations of machine learning. The hands-on approach to building gradient descent optimization demonstrates mastery of fundamental ML concepts that underpin more complex algorithms.

Understanding these basics is crucial for advancing to more sophisticated machine learning techniques and neural networks.

---

**Project Status**: ✅ Completed  
**Last Updated**: 15/7/2025 
**Time Invested**: 2 Hours  
