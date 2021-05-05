import numpy as np
import torch
from sklearn import datasets
from matplotlib import pyplot as plt

diabetes = datasets.load_diabetes()

#读取0-8行，生成张量，最后一列不读取
x_data=torch.from_numpy(diabetes.data).float()
#读取最后一列，[-1]是为了保证读取后是矩阵而不是向量
y_data = torch.from_numpy(diabetes.target).float()
y_data = y_data/y_data.max()
# y_data = y_data.unsqueeze(1)
class Model(torch.nn.Module):
    #初始化
    def __init__(self,activation_type):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(10,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        if(activation_type == 's'):
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()


    #前向传播
    def forward(self,x):
        #为了防止混淆，统一用x，这是惯例
        x=self.activation(self.linear1(x))
        x=self.activation(self.linear2(x))
        x=self.activation(self.linear3(x))
        return x


model_sig = Model("s")
loss_sig = []

model_relu = Model('r')
loss_relu = []


#损失函数和优化器
criterion = torch.nn.MSELoss()
sig_optimizer = torch.optim.Adam(model_sig.parameters(),lr=0.0001)
relu_optimizer = torch.optim.Adam(model_relu.parameters(),lr=0.0001)

n = 20000

#不使用mini-batch
for i in range(n):
    #Forward
    y_pred = model_sig(x_data).squeeze(1)
    y_pred2 = model_relu(x_data).squeeze(1)
    loss=criterion(y_pred,y_data)
    loss2=criterion(y_pred2,y_data)

    loss_sig.append(loss)
    loss_relu.append(loss2)
    #Backward
    #将梯度初始化为0
    sig_optimizer.zero_grad()
    relu_optimizer.zero_grad()
    loss.backward()
    loss2.backward()

    print("sig:",loss.item(),"   relu:",loss2.item())
    #Update
    sig_optimizer.step()
    relu_optimizer.step()

x = range(n)
plt.xlabel("epoch")
plt.ylabel("loss")



plt.plot(x,loss_sig, color="blue", linewidth=2.5, linestyle="-", label="sigmoid")
plt.plot(x,loss_relu, color="red",  linewidth=2.5, linestyle="-", label="relu")
plt.legend(loc='upper left')

plt.show()