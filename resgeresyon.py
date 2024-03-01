import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Generate some random data
x = np.random.rand(100)
y = x * 2 + np.random.rand(100)

# Create a scatter plot of the data
plt.scatter(x, y)

# Add a regression line
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
plt.plot(x, predictions, color='red')

# Set plot labels and title
plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.title('Regression plot')

# Show the plot
plt.show()