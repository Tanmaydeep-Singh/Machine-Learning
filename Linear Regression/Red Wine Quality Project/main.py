import pandas as pd
import matplotlib.pyplot as plotlib
from sklearn.linear_model import LinearRegression

data = pd.read_csv('winequality-red.csv')

data.describe()

# Assigning list data to variables
a  = pd.DataFrame(data, columns=['fixed acidity'])
b = pd.DataFrame(data, columns=['volatile acidity'])
c = pd.DataFrame(data, columns=['citric acid'])
d = pd.DataFrame(data, columns=['residual sugar'])
e = pd.DataFrame(data, columns=['chlorides'])
f = pd.DataFrame(data, columns=['free sulfur dioxide'])
g = pd.DataFrame(data, columns=['total sulfur dioxide'])
h = pd.DataFrame(data, columns=['density'])
i = pd.DataFrame(data, columns=['pH'])
j = pd.DataFrame(data, columns=['sulphates'])
k = pd.DataFrame(data, columns=['alcohol'])


plotList = [a,b,c,d,e,f,g,h,i,j,k]



y = pd.DataFrame(data, columns=['quality'])

for value in plotList:
    print()
    print(value)


    x = value

    regression = LinearRegression()

    regression.fit(x,y)

    regression.coef_  # slope cofficient  theta-1

    regression.intercept_ # theta

    print(f"{regression.coef_} , {regression.intercept_}" )


    regression.score(x,y) # Goodness of fit r^2
    print(regression.score(x,y))

    plotlib.figure(figsize=(10,6 ))
    plotlib.scatter(x,y,alpha=0.3)

    #to plot line
    plotlib.plot(x.values,regression.predict(x.values), color='red')
    plotlib.title('Red Wine Quality')
    plotlib.ylim(0,10)
    plotlib.xlabel(f'{value.columns[0]}')
    plotlib.ylabel('Quality')

    plotlib.show()





