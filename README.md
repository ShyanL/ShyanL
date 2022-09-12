智能201 2006500004 李叙言

```
#导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["font.sans-serif"]=["SimHei"] 
plt.rcParams["axes.unicode_minus"]=False

#读取房价数据集
points=np.genfromtxt('boston.csv',delimiter=',')
X1 = points[1:,0]
X2 = points[1:,1]
Y  = points[1:,2]

#特征缩放-归一化
m=len(X1)
for i in range(m):
    X1[i] = (X1[i] - min(X1)) / (max(X1) - min(X1))
    X2[i] = (X2[i] - min(X2)) / (max(X2) - min(X2))

#自定义参数    
alpha = 0.01       #学习率
init_theta0 = 0    #初始参数
init_theta1 = 0
init_theta2 = 0
grad_num = 1000   #迭代次数

#代价函数
def cost_func(X1, X2, Y, now_theta0, now_theta1, now_theta2):
    total_cost = 0
    m = len(X1)
    for i in range(m):
        x1 = X1[i]
        x2 = X2[i]
        y  =  Y[i]
        total_cost += (now_theta0 + now_theta1*x1 + now_theta2*x2 - y) ** 2
    return total_cost/2
```

<img src="https://github.com/ShyanL/ShyanL/blob/main/%E6%A2%AF%E5%BA%A6%E5%81%8F%E5%AF%BC%E5%85%AC%E5%BC%8F%E6%8E%A8%E5%AF%BC.jpg" width="450" height="450" alt="微信小程式"/><br/>

```
#梯度下降
def grad_desc(X1, X2, Y, init_theta0, init_theta1, init_theta2, alpha, grad_num):
    #定义当前参数取值
    now_theta0 = init_theta0
    now_theta1 = init_theta1
    now_theta2 = init_theta2
    #记录每迭代一次时的代价函数loss取值
    cost_list = []
    m=len(X1)
    
    #开始迭代
    for i in range(grad_num):
        #求偏导得出一个公式和，如附件图
        sum_grad_theta0 = 0
        sum_grad_theta1 = 0
        sum_grad_theta2 = 0
        #开始迭代
        for j in range(m):
            x1 = X1[j]
            x2 = X2[j]
            y  =  Y[j]
            sum_grad_theta0 += now_theta0 + now_theta1*x1 + now_theta2*x2 - y
            sum_grad_theta1 += (now_theta0 + now_theta1*x1 + now_theta2*x2 - y) * x1
            sum_grad_theta2 += (now_theta0 + now_theta1*x1 + now_theta2*x2 - y) * x2
        #更新参数
        now_theta0 = now_theta0 - alpha*sum_grad_theta0
        now_theta1 = now_theta1 - alpha*sum_grad_theta1
        now_theta2 = now_theta2 - alpha*sum_grad_theta2
        cost_list.append(cost_func(X1, X2, Y, now_theta0, now_theta1, now_theta2))
    
    return now_theta0, now_theta1, now_theta2, cost_list

#最终取值
theta0, theta1, theta2, cost_list = grad_desc(X1, X2, Y, init_theta0, init_theta1, init_theta2, alpha, grad_num)
print("theta0 is : ",theta0)
print("theta1 is : ",theta1)  
print("theta2 is : ",theta2)
```

<img src="https://github.com/ShyanL/ShyanL/blob/main/%E6%B1%82%E5%8F%82%E7%BB%93%E6%9E%9C.jpg" width="400" height="200" alt="微信小程式"/><br/>


```
#代价变化曲线图
plt.xlabel('迭代次数')
plt.ylabel('loss')
plt.plot(cost_list)
```
<img src="https://github.com/ShyanL/ShyanL/blob/main/%E4%BB%A3%E4%BB%B7%E5%87%BD%E6%95%B0%E6%B1%82%E5%80%BC%E5%8F%98%E5%8C%96%E8%BF%87%E7%A8%8B.jpg" width="400" height="300" alt="微信小程式"/><br/>

```
#建立三维视图
fig = plt.figure()
ax = Axes3D(fig,auto_add_to_figure=False)
fig.add_axes(ax)

#散点图
ax.scatter(X1, X2, Y, c='r', marker='o', s=50)

#拟合函数
X1, X2 = np.meshgrid(X1, X2)
func = theta0 + theta1*X1 + theta2*X2

ax.plot_surface(X1, X2, func, cmap='rainbow')
ax.set_xlabel('房子面积')
ax.set_ylabel('房间数量')
ax.set_zlabel('房子售价')
plt.show()   
```
