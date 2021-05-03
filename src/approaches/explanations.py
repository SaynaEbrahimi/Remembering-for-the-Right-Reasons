'''
Saliency maps implementations are inspired by https://github.com/kazuto1011/grad-cam-pytorch
'''

from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class _BaseWrapper(object):
    def __init__(self, model, args):
        super(_BaseWrapper, self).__init__()
        self.device = args.device.name
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image, task_id):
        # self.image_shape = image.shape[2:]
        self.model.zero_grad()
        self.logits = self.model(image, task_id)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        # self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):

    def __init__(self, args):
        # model, candidate_layers=None):
        self.model = None # model is set when module is called
        super(BackPropagation, self).__init__(self.model, args)
        self.args = args

    def forward(self, image, task_id):
        self.image_shape = image.shape[2:]
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image, task_id)

    # def forward(self, image):
    #     self.image = image.requires_grad_()
    #     return super(BackPropagation, self).forward(self.image)

    def __call__(self, inputs, model, task_id):
        self.model = model

        probs, ids = self.forward(inputs, task_id)
        ids_ = torch.LongTensor([ids[:, 0].tolist()]).T.to(device=self.args.device.name)
        self.backward(ids_)
        gradient  = self.generate()
        # gradient = gradient.squeeze(1)
        return gradient, self.model, probs[:,0], ids[:,0]


    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, args):
        self.model = None  # model is set when module is called
        super(GuidedBackPropagation, self).__init__(args)
        self.args = args

    # def __init__(self, model):
    #     super(GuidedBackPropagation, self).__init__(model)

    # def backward_hook(self, module, grad_in, grad_out):
    #     # Cut off negative gradients
    #     if isinstance(module, nn.ReLU):
    #         return (F.relu(grad_in[0]),)


    def __call__(self, inputs, model, task_id):
        self.model = model

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))

        probs, ids = self.forward(inputs, task_id)
        ids_ = torch.LongTensor([ids[:, 0].tolist()]).T.to(device=self.args.device.name)
        self.backward(ids_)
        gradient  = self.generate()
        # gradient = gradient.squeeze(1)
        return gradient, self.model, probs[:,0], ids[:,0]


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, args):
        self.model = None  # model is set when module is called
        super(Deconvnet, self).__init__(args)
        self.args = args

    def forward(self, image, task_id):
        self.image_shape = image.shape[2:]
        return super(Deconvnet, self).forward(image, task_id)

    def __call__(self, inputs, model, task_id):
        self.model = model

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_out[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))

        probs, ids = self.forward(inputs, task_id)
        ids_ = torch.LongTensor([ids[:, 0].tolist()]).T.to(device=self.args.device.name)
        self.backward(ids_)
        gradient  = self.generate()
        gradient = gradient.squeeze(1)
        return gradient, self.model, probs[:,0], ids[:,0]





class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """
    def __init__(self, args):
        # model, candidate_layers=None):
        self.model = None # model is set when module is called
        super(GradCAM, self).__init__(self.model, args)
        self.args = args


    # def __init__(self, args):
    #     self.model = None  # model is set when module is called
    #     super(GradCAM, self).__init__(self.model,args)
    #     self.args = args
        self.fmap_pool = {}
        self.grad_pool = {}
        # self.candidate_layers = candidate_layers  # list
        self.target_layer = args.architecture.target_layer  # string of layer name

        self.upsample = args.saliency.upsample





    def __call__(self, inputs, model, task_id):
        self.model = model
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output#.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0]#.detach()

            return backward_hook
        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.target_layer is None or name == self.target_layer:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

        probs, ids = self.forward(inputs, task_id)
        ids_ = torch.LongTensor([ids[:, 0].tolist()]).T.to(device=self.args.device.name)
        self.backward(ids_)
        gradcams, fmaps = self.generate(target_layer=self.target_layer)
        gradcams = gradcams.squeeze(1) # shape: [B, H, W]
        return gradcams, self.model, probs[:,0], ids[:,0]


    def forward(self, image, task_id):
        self.image_shape = image.shape[2:]

        return super(GradCAM, self).forward(image, task_id)

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)


        if self.upsample:
            gcam = F.interpolate(
                gcam, self.image_shape, mode="bilinear", align_corners=False
            )

            B, C, H, W = gcam.shape
            gcam = gcam.view(B, -1)
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam /= gcam.max(dim=1, keepdim=True)[0]
            gcam = gcam.view(B, C, H, W)
        return gcam, fmaps

        #     return gcam
        # else:
        #     return gcam.squeeze(1)




class SmoothGrad(_BaseWrapper):


    def __init__(self, args):
        self.model = None  # model is set when module is called
        super(SmoothGrad, self).__init__(self.model, args)
        self.args = args
        self.sigma = 0.2
        self.n_samples = 20


    def forward(self, image, task_id):
        self.image_shape = image.shape[2:]
        self.image = image.requires_grad_()
        return super(SmoothGrad, self).forward(self.image, task_id)


    def __call__(self, inputs, model, task_id):
        self.model = model

        if self.args.saliency.method == 'gsmooth':
            def backward_hook(module, grad_in, grad_out):
                # Cut off negative gradients
                if isinstance(module, nn.ReLU):
                    return (F.relu(grad_in[0]),)


            for module in self.model.named_modules():
                self.handlers.append(module[1].register_backward_hook(backward_hook))

        sigma = (inputs.max() - inputs.min()) * self.sigma
        gradients = []

        # for i in tqdm(range(self.n_samples)):
        for i in range(self.n_samples):

            noised_image = inputs + torch.randn(inputs.size(),device=self.device) * sigma
            # noised_image = noised_image.to(self.device)

            probs, ids = self.forward(noised_image, task_id)
            ids_ = torch.LongTensor([ids[:, 0].tolist()]).T.to(device=self.args.device.name)
            self.backward(ids_)
            gradient  = self.generate()
            gradients.append(gradient)

        gradient = torch.stack(gradients)


        return gradient, self.model, None, None


    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient

