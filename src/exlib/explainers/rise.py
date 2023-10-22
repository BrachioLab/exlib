"""
https://github.com/eclique/RISE/blob/master/explanations.py
"""


import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from .common import FeatureAttrMethod, FeatureAttrOutput


class TorchImageRISE(FeatureAttrMethod):
    def __init__(self, model, input_size, postprocess=None, \
                 gpu_batch=100, N=2000, \
                 s=8, p1=0.5, seed=42):
        super(TorchImageRISE, self).__init__(model, postprocess)
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.generate_masks(N, s, p1)

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], 
                                                              y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        # np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.to(device)
        self.N = N
        self.p1 = p1

    def forward(self, x, label=None):
        # Apply array of filters to the image]
        # print('RISE')
        self.model.eval()
        with torch.no_grad():
            N = self.N
            B, C, H, W = x.size()
            stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
            stack = stack.view(N * B, C, H, W)
            # stack = stack

            #p = nn.Softmax(dim=1)(model(stack)) in batches
            p = []
            for i in range(0, N*B, self.gpu_batch):
                pred = self.model(stack[i:min(i + self.gpu_batch, N*B)])
                if self.postprocess is not None:
                    pred = self.postprocess(pred)
                p.append(pred)
            p = torch.cat(p)
            if label is None:
                # if no label, then explain the top class
                pred_x = self.model(x)
                if self.postprocess is not None:
                    pred_x = self.postprocess(pred_x)
                label = torch.argmax(pred_x, dim=-1)
            CL = p.size(1)
            p = p.view(N, B, CL)
            sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
            sal = sal.view(B, CL, H, W)
        
        return FeatureAttrOutput(sal[range(B), label], sal)


class TorchTextRISE(FeatureAttrMethod):
    def __init__(self, model, input_size, postprocess=None, \
                 gpu_batch=100, N=500, \
                 s=8, p1=0.5, seed=42, mask_combine=None):
        super(TorchTextRISE, self).__init__(model, postprocess)
        self.input_size = input_size  # only one number for text
        self.gpu_batch = gpu_batch
        self.generate_masks(N, s, p1)
        self.mask_combine = mask_combine

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, self.input_size))

        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size)
            # Linear upsampling and cropping
            self.masks[i, :] = resize(grid[i], tuple([up_size]), order=1, mode='reflect', 
                                        anti_aliasing=False)[x:x + self.input_size]
        self.masks = self.masks.reshape(-1, self.input_size)
        # np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.to(device)
        self.N = N
        self.p1 = p1

    def forward(self, x, label=None, kwargs=None):
        # Apply array of filters to the image]
        # print('RISE')
        self.model.eval()
        with torch.no_grad():
            N = self.N
            B, L = x.size()
            if self.mask_combine:
                stack = self.mask_combine(x, self.masks)
                stack = stack.reshape(N * B, L, -1)
            else:
                stack = torch.mul(self.masks.view(N, 1, L), x.data.view(B, L))
                stack = stack.reshape(N * B, L)
            kwargs_new = {}
            for k, v in kwargs.items():
                if v is not None:
                    kwargs_new[k] = v.unsqueeze(0).expand(N, B, L).reshape(N * B, L)
            # stack = stack

            #p = nn.Softmax(dim=1)(model(stack)) in batches
            p = []
            for i in tqdm(range(0, N*B, self.gpu_batch)):
                kwargs_curr = {k: v[i:min(i + self.gpu_batch, N*B)] for k, v in kwargs_new.items()}
                if self.mask_combine:
                    pred = self.model(inputs_embeds=stack[i:min(i + self.gpu_batch, N*B)], **kwargs_curr)
                else:
                    pred = self.model(stack[i:min(i + self.gpu_batch, N*B)], **kwargs_curr)
                if self.postprocess is not None:
                    pred = self.postprocess(pred)
                p.append(pred)
            p = torch.cat(p)
            if label is None:
                # if no label, then explain the top class
                pred_x = self.model(x, **kwargs)
                if self.postprocess is not None:
                    pred_x = self.postprocess(pred_x)
                label = torch.argmax(pred_x, dim=-1)
            CL = p.size(1)
            p = p.view(N, B, CL)
            sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, L))
            sal = sal.view(B, CL, L)
        
        return FeatureAttrOutput(sal[range(B), label], sal)
