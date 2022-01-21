import torch

tt1=torch.tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
tt2=torch.tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599, 0.1, 0.2])
result=tt1.view(3, 2)
tt2=tt2.view(4, 2)
print(result)
print(tt2)
result=result.expand(3,3)
print(result)
m = nn.ConstantPad2d(2, 3.5)