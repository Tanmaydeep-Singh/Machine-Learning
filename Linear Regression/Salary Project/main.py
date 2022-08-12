### In this you will learn the basics of LR ###


## Importing libraries

import pandas as pd  #To read data


import matplotlib.pyplot as plt  #To plot graph


from sklearn.linear_model import LinearRegression #To imply linear regression


data = pd.read_csv('Salary_Data.csv') #Reading data file

data.describe()



# Assigning list data to variables
x = pd.DataFrame(data, columns=['YearsExperience'])
y = pd.DataFrame(data, columns=['Salary']) #Salary



regression = LinearRegression() #Getting linearregression function

regression.fit(x,y) #To give values

regression.coef_  # slope cofficient  theta-1  || Gives an array of weight estimated by Linear Regression.

regression.intercept_ # theta || Gives the  x value



regression.score(x,y) # Goodness of fit r^2 || Turns the difference to positive

print(regression.score(x,y))

#Plotting the Graph
plt.figure(figsize=(10,6 ))

plt.scatter(x,y,alpha=0.3) #Marking the dots

#to plot line
plt.plot(x.values,regression.predict(x.values), color='c')
plt.title('Job experience VS Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()