# Linear-Regression
# Full Python code for simple linear regression from scratch

# --- Helper Functions ---

def get_mean(values):
    """Calculates the mean of a list of numbers."""
    return sum(values) / float(len(values))

def get_variance(values, mean):
    """Calculates the variance of a list."""
    return sum([(x - mean)**2 for x in values])

def get_covariance(x_values, mean_x, y_values, mean_y):
    """Calculates the covariance between two lists."""
    covariance = 0.0
    for i in range(len(x_values)):
        covariance += (x_values[i] - mean_x) * (y_values[i] - mean_y)
    return covariance

# --- Main Regression Function ---

def simple_linear_regression(X, y):
    """
    Calculates the slope (m) and intercept (b) for a simple linear regression.
    
    The formula for the slope (m) is:
    m = covariance(x, y) / variance(x)
    
    The formula for the intercept (b) is:
    b = mean(y) - m * mean(x)
    """
    
    # Calculate means
    mean_x = get_mean(X)
    mean_y = get_mean(y)
    
    # Calculate variance and covariance
    var_x = get_variance(X, mean_x)
    covar_xy = get_covariance(X, mean_x, y, mean_y)
    
    # Calculate coefficients
    m = covar_xy / var_x
    b = mean_y - m * mean_x
    
    return m, b

# --- How to use the code ---

if __name__ == "__main__":
    
    # 1. Provide your sample data
    # Let's say we have hours studied (X) and test score (y)
    X_data = [1, 2, 4, 5, 5, 6, 7]
    y_data = [30, 45, 50, 65, 70, 75, 80]

    print("--- Simple Linear Regression (Pure Python) ---")
    print(f"X data: {X_data}")
    print(f"y data: {y_data}")
    print("-" * 30)

    # 2. Calculate the regression coefficients
    try:
        slope, intercept = simple_linear_regression(X_data, y_data)
        
        print(f"Calculated slope (m): {slope:.4f}")
        print(f"Calculated intercept (b): {intercept:.4f}")
        print("\nYour regression line equation is: y = {:.4f}x + {:.4f}".format(slope, intercept))
        print("-" * 30)

        # 3. Use the model to make predictions
        hours_to_predict = 3
        predicted_score = (slope * hours_to_predict) + intercept
        
        print(f"Predicting score for {hours_to_predict} hours of study:")
        print(f"Predicted score (y): {predicted_score:.2f}")

    except ZeroDivisionError:
        print("Error: Cannot perform regression. The variance of X is zero.")
    except Exception as e:
        print(f"An error occurred: {e}")
