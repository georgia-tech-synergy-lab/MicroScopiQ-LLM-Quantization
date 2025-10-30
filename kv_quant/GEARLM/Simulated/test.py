from compress_function import gears_channelQ
import torch
input = torch.randn([4,32,1000,128])
result = gears_channelQ(input,4,100,0.02)
# print(torch.norm(input - result,p=1))
