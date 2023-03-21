import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from tabulate import tabulate

data=pd.read_csv("student_scores.csv")
data
#plt.scatter(data.Hours, data.Scores)
#plt.show()

def f(m,b,x):
    return m*x+b;

def loss_function(m,b,points):
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores
        total_error+=(y-m*x+b)**2
    return total_error/len(points)

def gradient_descent(m_now, b_now, points, L):
    m_dradient = 0
    b_grandient=0
    n=len(points)
    for i in range(n):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores
        m_dradient+=-(2/n) * x * (y-(m_now*x+b_now))
        b_grandient+=-(2/n) * (y-(m_now*x+b_now))
    m=m_now-m_dradient*L
    b=b_now-b_grandient*L
    return m,b
m=0
b=0
L=0.0001
epochs=1000
errors=[]
for i in range(epochs):
    m,b=gradient_descent(m,b,data,L)
    loss=loss_function(m,b,data)
    errors.append(loss);

print(f"regression={m}*x+{b}")
plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
plt.xlabel("Hours")
plt.ylabel("Score")
plt.scatter(data.Hours, data.Scores, color='black')
plt.plot(list(range(0,10)),[m*x+b for x in range(0,10)], color='red')
plt.subplot(2,1,2)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.plot(list(range(0,len(errors))), errors)
plt.show()
data['predicted']=f(m,b,data['Hours'])
#display(data)
print(tabulate(data, headers = 'keys', tablefmt = 'psql'))