# ============================ Polynomial Regression ==============================

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#=================================================================================
'''
The data used composes of a lot of information. For this task the engine
size and fuel consumption data has been isolated for modelling.
'''
# Read information from .csv file and save it to their respective
# variables.

def readCSV():
    # Create empty lists to add information to
    engine_size = []
    consumption = []
    with open('fuel_consumption.csv') as csvfile:
        read = csv.reader(csvfile, delimiter = ',')
        temp = 0
        for row in read:
            # This if statement is to store the headings separately and the data separately
            if temp == 1:
                engine_size.append(float(row[4]))
                consumption.append(float(row[11]))
            else:
                temp = 1
    return engine_size, consumption

x_data, y_data = readCSV()

# The data has 1067 data points. Use the first 600 points to train the model
# and the rest for testing.
x_train = np.reshape(x_data[:601], (1, -1)).T
y_train = np.reshape(y_data[:601], (1, -1)).T

x_test = np.reshape(x_data[601:], (1, -1)).T
y_test = np.reshape(y_data[601:], (1, -1)).T

#=================================================================================

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Engine Size vs Fuel Consumption')
plt.xlabel('Engine Size (liters)')
plt.ylabel('Fuel Consumption (mpg)')
plt.axis([0, 15, 0, 70])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()

'''
If you execute the code, you will see that the simple linear regression model is plotted with
a solid line. The quadratic regression model is plotted with a dashed line and evidently the
quadratic regression model fits the training data better.

The line of best fit with a degree of 2 more accurately describes the fuel consumption up to
where the data points end. Realistically speaking, if the engine sizes were to increase beyond 8
liters the mpg would increase, which isn't realistic.


REFERENCES
1. I got the .csv data after googling random data from
https://codekarim.com/sites/default/files/ML0101EN-Reg-Polynomial-Regression-Co2-py-v1.html#download_data

'''
