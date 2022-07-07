from statistics import linear_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

"""_summary_
    Program train, predict and plot a Polynomial regression graph
    Dataset was created using fps values from tom's hardware and settings were 1080p ultra
    https://www.tomshardware.com/reviews/gpu-hierarchy,4388.html
    Pricing was the cheapest instock gpu's at Wootware
    https://www.wootware.co.za/computer-hardware/video-cards-video-devices
"""
 
# Reads the user selected csv file
def read_csv(file):
    return pd.read_csv(file)

def split_data(dataset):
    # Splits data into trainin and test data randomly
    training_data , test_data  = [i.to_numpy() for i in train_test_split
                                  (dataset, train_size=0.8)]
    train_x = training_data[:, 1:2]
    test_x = test_data[:, 1:2]
    train_y = training_data[:, 2]
    test_y = test_data[:, 2]
    return train_x, train_y, test_x, test_y


dataset = read_csv("GpuData.csv")
train_x, train_y, test_x, test_y = split_data(dataset)

# linear line prediction
linear_line = LinearRegression()
linear_line.fit(train_x, train_y)
line = np.linspace(0, 140)
line_prediction = linear_line.predict(line.reshape(line.shape[0], 1))

# polynomial line prediction
polynomial_line = PolynomialFeatures(degree=2)
train_x_quadratic = polynomial_line.fit_transform(train_x)
test_x_quadratic = polynomial_line.transform(train_x)
quadratic_regression = LinearRegression()
quadratic_regression.fit(train_x_quadratic, train_y)
quadratic_prediction = polynomial_line.transform(line.reshape(line.shape[0], 1))

print(dataset.set_index("GPU"))

# Plot the graph
plt.title('Graphics card price per fps')
plt.xlabel('1080p ultra settings fps')
plt.ylabel('Graphics card prices in ZAR')
plt.scatter(train_x, train_y, color = "grey")
plt.scatter(test_x, test_y, color = "green")
plt.plot(line, line_prediction, color = "violet")
plt.plot(line, quadratic_regression.predict(quadratic_prediction), color = 'red', linestyle='--')
plt.axis([0, 140, 0, 62000])
plt.legend(["Trained data", "Test data", "Linear regression line", "Polynomial regression line"])
plt.show()