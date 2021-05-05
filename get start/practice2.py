import torch

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    '''对forward()函数进行重写'''
    '''这是一个callable的函数'''
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
'''优化器'''
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

for epoch in range(1000):
    '''前向传播'''
    y_pred = model(x_data)
    '''计算损失'''
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    '''一定要将梯度归零'''
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#打印权重和偏置
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

#测试模型
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=',y_test.data)