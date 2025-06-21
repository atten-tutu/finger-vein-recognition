# pip install thop
# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
import time
from models import get_model

# Model
print('==> Building model..')
model = get_model()
input = torch.randn(1, 3, 224, 224)
dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
torch.cuda.synchronize()
start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result = model(input.to(device))
torch.cuda.synchronize()
end = time.time()
print('infer_time:', end-start)
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
