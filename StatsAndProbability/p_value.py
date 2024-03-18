import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)  # Set random seed for reproducibility

r_squared_values = []
p_values = []
last_iteration_data = None

for i in range(555):
    # Generate two correlated random variables with a correlation of 0.2
    cov_matrix = [[1.0, 0.2], [0.2, 1.0]]
    x, y = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=100).T

    # Create a DataFrame from the generated variables
    df = pd.DataFrame({'X': x, 'Y': y})

    # Perform regression using statsmodels
    X = df['X']
    X = sm.add_constant(X)  # Add a constant term for the intercept
    model = sm.OLS(df['Y'], X)
    results = model.fit()

    # Retrieve R-squared and p-value
    r_squared = results.rsquared
    p_value = results.pvalues[1]  # p-value for the coefficient of X

    # Append values to the lists
    r_squared_values.append(r_squared)
    p_values.append(p_value)

    if i == 554:
        # Store data of the last iteration
        last_iteration_data = df

# Plot the scatter plot of the last iteration
plt.figure(figsize=(8, 6))
plt.scatter(last_iteration_data['X'], last_iteration_data['Y'])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of Last Iteration")
plt.grid(True)
plt.show()

# Plot the distribution of R-squared
plt.figure(figsize=(10, 6))
plt.hist(r_squared_values, bins=20, edgecolor='black')
plt.xlabel("R-squared")
plt.ylabel("Frequency")
plt.title("Distribution of R-squared (Correlation = 0.2)")
plt.grid(True)
plt.show()

# Plot the distribution of p-values
plt.figure(figsize=(10, 6))
plt.hist(p_values, bins=20, edgecolor='black')
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.title("Distribution of p-values (Correlation = 0.2)")
plt.grid(True)
plt.show()

# Calculate the survival function of R-squared
r_squared_values.sort(reverse=True)
survival = np.arange(1, len(r_squared_values) + 1) / len(r_squared_values)

# Plot the survival function of R-squared
plt.figure(figsize=(10, 6))
plt.plot(r_squared_values, survival, marker='o', linestyle='-', color='blue')
plt.xlabel("R-squared")
plt.ylabel("Survival Function")
plt.title("Survival Function of R-squared (Correlation = 0.2)")
plt.grid(True)
plt.show()
