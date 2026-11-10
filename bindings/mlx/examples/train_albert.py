"""Compare native FullyFusedMLP with MLX matmuls, sharing native HashGrid."""
import argparse, json, time
from pathlib import Path
from functools import partial
import numpy as np
from PIL import Image
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tinymetalnn_mlx as tcnn

p = argparse.ArgumentParser()
p.add_argument('--batch-size', type=int, default=65536)
p.add_argument('--steps', type=int, default=1000)
p.add_argument('--mode', choices=['native', 'mlx'], required=True)
p.add_argument('--image', default='albert.jpg')
p.add_argument('--out', default='outputs/albert')
a = p.parse_args()
enc = dict(otype='HashGrid', n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, per_level_scale=2.0)
net = dict(otype='FullyFusedMLP', activation='ReLU', output_activation='None', n_neurons=64, n_hidden_layers=2)
image = np.asarray(Image.open(a.image).convert('RGB'), dtype=np.float32)/255
h,w,_ = image.shape
pixels = mx.array(image.reshape(-1,3))
reference = tcnn.NetworkWithInputEncoding(2,3,enc,net,seed=17)
class Plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding = tcnn.Encoding(2,enc,seed=17,dtype=reference.dtype)
        self.weights = []
        offset=0
        # Native output layer is padded to 16 channels.
        for ni,no in [(32,64),(64,64),(64,16)]:
            self.weights.append(reference.params[offset:offset+ni*no].reshape(no,ni))
            offset += ni*no
        assert reference.n_params == offset+self.encoding.n_params
        self.encoding.params = reference.params[offset:]
    def __call__(self,x):
        x=self.encoding(x)
        for i,weight in enumerate(self.weights):
            x=x @ weight.astype(reference.dtype).T
            if i<2: x=mx.maximum(x,0)
        return x[:,:3]
model = reference if a.mode=='native' else Plain()
optimizer=optim.Adam(learning_rate=1e-3)
def loss(m,x,y): return mx.mean(mx.square(m(x).astype(mx.float32)-y))
vg=nn.value_and_grad(model,loss)
def step(x,y):
    value,grads=vg(model,x,y)
    optimizer.update(model,grads)
    return value
state=[model.state,optimizer.state]
step=partial(mx.compile,inputs=state,outputs=state)(step)
mx.random.seed(23)
def sample():
    idx=mx.random.randint(0,h*w,(a.batch_size,))
    xy=mx.stack(((idx%w+0.5)/w,(idx//w+0.5)/h),axis=1)
    return xy,pixels[idx]
mx.eval(model.parameters(),pixels)
# The first update includes compilation and is reported separately.
x,y=sample()
if a.mode == 'mlx':
    assert mx.allclose(model(x[:128]),reference(x[:128]),atol=0.003,rtol=0.03).item()
start=time.perf_counter()
value=step(x,y); mx.eval(value,model.parameters(),optimizer.state)
first=time.perf_counter()-start
# Count warmup as iteration 1: exactly --steps parameter updates.
start=time.perf_counter()
for i in range(1,a.steps):
    x,y=sample()
    value=step(x,y)
    mx.eval(value,model.parameters(),optimizer.state)
    if (i+1)%100==0:
        print(json.dumps(dict(mode=a.mode,batch=a.batch_size,iteration=i+1,loss=value.item(),elapsed=time.perf_counter()-start)),flush=True)
elapsed=time.perf_counter()-start
pred=[]
for begin in range(0,h*w,65536):
    idx=mx.arange(begin,min(begin+65536,h*w))
    xy=mx.stack(((idx%w+0.5)/w,(idx//w+0.5)/h),axis=1)
    v=model(xy).astype(mx.float32); mx.eval(v); pred.append(np.array(v))
result=np.concatenate(pred).reshape(h,w,3)
mse=float(np.mean((result-image)**2))
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
stem=f'{a.mode}-{a.batch_size}'
Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)).save(out/f'{stem}.png')
record=dict(mode=a.mode,batch=a.batch_size,steps=a.steps,first_step_seconds=first,remaining_seconds=elapsed,ms_per_step=elapsed/(a.steps-1)*1000,mse=mse,psnr=-10*np.log10(mse),encoding=enc,network=net)
(out/f'{stem}.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
