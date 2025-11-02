import numpy as np 

rng = np.random.default_rng(42)  
X = np.linspace(0, 10, 50)       
noise = rng.normal(0, 1.0, size=X.shape)  
y = 3.0 * X + 2.0 + noise       

X_mean, X_std = X.mean(), X.std() 
X_scaled = (X - X_mean) / X_std  

X_design = np.c_[np.ones_like(X_scaled), X_scaled] 

theta = np.zeros(2, dtype=float) 

alpha = 0.1       
epochs = 1000     

def predict(X_mat, theta_vec):
    return X_mat @ theta_vec 

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) 

def gradient(X_mat, y_true, theta_vec):
    n = X_mat.shape[0]
    y_pred = predict(X_mat, theta_vec)
    grad = -(2.0 / n) * (X_mat.T @ (y_true - y_pred)) 
    return grad 

loss_history = []
for epoch in range(epochs):
    y_pred = predict(X_design, theta)     
    loss = mse(y, y_pred)                 
    loss_history.append(loss)            
    grad = gradient(X_design, y, theta)   
    theta -= alpha * grad                 

theta0_scaled, theta1_scaled = theta 

intercept_original = theta0_scaled - theta1_scaled * (X_mean / X_std)  
slope_original = theta1_scaled / X_std                                 

print("Fitted parameters (original scale):")
print(f"Intercept ≈ {intercept_original:.3f}, Slope ≈ {slope_original:.3f}") 
print(f"Final MSE: {loss_history[-1]:.3f}")

x_new = np.array([[-1.0], [0.0], [5.0], [12.0]]).ravel()  
y_new = intercept_original + slope_original * x_new      
print("Predictions:", y_new)  

try:
    from sklearn.linear_model import LinearRegression 
    lr = LinearRegression().fit(X.reshape(-1, 1), y)  
    print("Sklearn Intercept, Slope:", lr.intercept_, lr.coef_[0]) 
except Exception as e:
    print("scikit-learn not available; skipping comparison.") 
