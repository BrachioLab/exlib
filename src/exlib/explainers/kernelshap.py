import torch
import torch.nn.functional as F
import shap
import math
import numpy as np
from lime import lime_image
from .common import *

def patch_segment(image, patch_height=8, patch_width=8,permute = None,dtype = "torch"):
    #Input: C,H,W
    if permute is not None:
        image = np.transpose(image,permute)

    channels, image_height, image_width = image.shape
    if image_height % patch_height != 0 or image_width % patch_width != 0:
        print("patch height and width need to perfectly divide image")
        raise ValueError

    row_indices = torch.arange(image_height)
    column_indices = torch.arange(image_width)

    row_factors = row_indices // patch_height
    column_factors = column_indices // patch_width

    row_factor_matrix = row_factors[:, None]
    column_factor_matrix = column_factors[None, :]

    segment = column_factor_matrix * (image_height // patch_height) + row_factor_matrix
    if dtype == "torch":
        return segment.int()
    elif dtype == "numpy":
        return segment.numpy()
    else:
        raise ValueError


def fill_segmentation(values, segmentation,dtype = "torch"):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    if dtype == "numpy":
        return out

    else:
        return torch.Tensor(out)

def mask_image_torch(zs, segmentation, image, model,background=None):
    permuted = False
    if image.shape[0] == 3:
        image = torch.permute(image,(1,2,0))
        permuted = True

    if background is None:
        background = image.mean((0,1))
    out = torch.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2])).to(next(model.parameters()).device)
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background

    if permuted:
        out = torch.permute(out,(0,3,1,2))


    return out

def f_torch(z,image,segments,model):
    x = mask_image_torch(z,segments,image,model,0) #setting background = 0 (black)
    x =  model(x).detach().cpu().numpy()
    return x


def explain_image_cls_with_kernelShap(model, x, ts,
                                kernelShapImageExplainerKwargs={},
                                # Gets FA for every label if top_labels == None
                                explain_instance_kwargs={},
                                get_image_and_mask_kwargs={},
                                patch_height=16,
                                patch_width=16,
                                nsamples=100,
                                ):
    """

    """
    segmentation_fn = explain_instance_kwargs.get('segmentation_fn')

    if segmentation_fn is None:
        segments = patch_segment(x,patch_height=patch_height,patch_width=patch_width,dtype="torch")

    else:
        segments = segmentation_fn(x) #Look into providing arguments to this

    # print("segments shape",segments.shape)
    f = lambda z : f_torch(z,x,segments,model)

    ## Images here are not batched
    C, H, W = x.shape
    x_np = x.cpu().permute(1,2,0).numpy()
    # print(len(segments.unique()), segments.max(), segments.min())
    explainer = shap.KernelExplainer(f, np.zeros((1,len(segments.unique()))), **kernelShapImageExplainerKwargs)
    
    # print('segments shape:', segments.shape)
    # print('np.unique(segments).shape:', np.unique(segments.cpu().numpy()).shape)
    # print('Input to shap_values:', np.ones((1, np.unique(segments.cpu().numpy()).shape[0])).shape)

    # import pdb; pdb.set_trace()
    shap_values = explainer.shap_values(np.ones((1,len(segments.unique()))),nsamples=nsamples)

    if isinstance(ts, torch.Tensor):
        todo_labels = ts.numpy()
    else:
        todo_labels = ts




    attrs = []
    for t in todo_labels:
        m = fill_segmentation(shap_values[t][0], segments)
        attrs.append(m)
    attrs = torch.stack(attrs)
    return FeatureAttrOutput(attrs, explainer)



class KernelShapImageCls(FeatureAttrMethod):
    def __init__(self, model,
                 KernelShapImageExplainerKwargs={},
                 explain_instance_kwargs={
                 },
                 get_image_and_mask_kwargs={},
                 patch_height=16,
                 patch_width=16,
                 nsamples=100,
                 ):
        super(KernelShapImageCls, self).__init__(model)
        self.KernelShapImageExplainerKwargs = KernelShapImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.nsamples = nsamples


    def forward(self, x, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        N = x.size(0)
        assert x.ndim == 4 and len(t) == N
        if t.ndim == 1:
            t = t.unsqueeze(1)
        assert t.ndim == 2

        attrs, kshap_exps = [], []
        for i in range(N):
            xi, ti = x[i], t[i]
            out = explain_image_cls_with_kernelShap(self.model, xi, ti.cpu().numpy().tolist(),
                    kernelShapImageExplainerKwargs=self.KernelShapImageExplainerKwargs,
                    explain_instance_kwargs=self.explain_instance_kwargs,
                    get_image_and_mask_kwargs=self.get_image_and_mask_kwargs, 
                    patch_height=self.patch_height,
                    patch_width=self.patch_width,
                    nsamples=self.nsamples,
                    )

            attrs.append(out.attributions.permute(1,2,0).unsqueeze(0)) #.repeat(3,1,1))
            kshap_exps.append(out.explainer_output)

        attrs = torch.stack(attrs, dim=0)
        if attrs.ndim == 5 and attrs.size(-1) == 1:
            attrs = attrs.squeeze(-1)

        return FeatureAttrOutput(attrs, kshap_exps)

