# This code is adapted from the https://github.com/jacobgil/pytorch-grad-cam

import argparse
import os
import cv2
import inspect
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
# from torchvision import models

from option import Options
# import models.model_zoo as models
from models import model as M
from datasets.cassava import CassavaDataset, NUM_CLASS


# from torchvision import models


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        # for resnet
        for name, module in self.model._modules.items():
            if name == 'fc':
                continue

            if name != self.target_layers[0]:
                x = module(x)
            else:
                for sub_name, sub_module in module._modules.items():
                    if sub_name != self.target_layers[1]:
                        x = sub_module(x)
                    else:
                        for sub_sub_name, sub_sub_module in sub_module._modules.items():
                            x = sub_sub_module(x)
                            if sub_sub_name == self.target_layers[2]:
                                x.register_hook(self.save_gradient)
                                outputs += [x]

        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.pretrained, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.head(output)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img_name, img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(img_name, np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.pretrained.zero_grad()
        self.model.head.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use-cuda', action='store_true', default=False,
#                         help='Use NVIDIA GPU acceleration')
#     # parser.add_argument('--image-path', type=str, default='./examples/both.png',
#     #                     help='Input image path')
#     parser.add_argument('--image-path', type=str,
#                         default='/home/ace19/dl_data/kgc_multiview/validation',
#                         help='Input image path')
#     parser.add_argument('--output-path', type=str,
#                         default='/mnt/sda/dl_results/kgc_multiview/20200205_12view_training_for_2nd_official_test/grad_cam',
#                         help='Input image path')
#     parser.add_argument('--resume', type=str,
#                         # default=None,
#                         # default='pre-trained/RESNET50_CBAM_new_name_wrap.pth',
#                         default='pre-trained/multiview_encoding_model_best.pth.tar',
#                         # default='pre-trained/multiview_encoding_checkpoint95.pth.tar',
#                         help='put the path to resuming file if needed')
#     args = parser.parse_args()
#     args.use_cuda = args.use_cuda and torch.cuda.is_available()
#     if args.use_cuda:
#         print("Using GPU for acceleration")
#     else:
#         print("Using CPU for computation")
#
#     return args


def main():
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init the model
    # model = models.get_model(args.model)
    model = M.Model(NUM_CLASS, backbone=args.model)
    print(model)

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            # model.module.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no resume checkpoint found at '{}'". \
                               format(args.resume))
    else:
        raise RuntimeError("=> config \'args.resume\' is '{}'". \
                           format(args.resume))

    """ 
    1. Loads an image with opencv.
    2. Preprocesses it for model and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    
    Makes the visualization. 
    """
    grad_cam = GradCam(model=model,
                       # target_layer_names=['layer4', '2', 'relu'],
                       target_layer_names=['layer4', '2', 'bn2'],
                       use_cuda=args.cuda)
    # grad_cam = GradCam(model=models.vgg19(pretrained=True), \
    #                    target_layer_names=["35"], use_cuda=args.cuda)

    cls_lst = os.listdir(args.image_path)
    for cls in cls_lst:
        if not os.path.exists(os.path.join(args.output_path, cls)):
            os.makedirs(os.path.join(args.output_path, cls))

        cls_path = os.path.join(args.image_path, cls)
        img_lst = os.listdir(cls_path)
        for img in img_lst:
            img_path = os.path.join(cls_path, img)
            image = cv2.imread(img_path, 1)
            image = np.float32(cv2.resize(image, (224, 224))) / 255
            input = preprocess_image(image)

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested index.
            target_index = None
            mask = grad_cam(input, target_index)

            show_cam_on_image(os.path.join(args.output_path, cls, img), image, mask)

    # gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), use_cuda=args.use_cuda)
    # gb = gb_model(input, index=target_index)
    # gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask*gb)
    # gb = deprocess_image(gb)
    #
    # cv2.imwrite('gb.jpg', gb)
    # cv2.imwrite('cam_gb.jpg', cam_gb)


if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     """ python grad-cam.py <path_to_image>
#     1. Loads an image with opencv.
#     2. Preprocesses it for VGG19 and converts to a pytorch variable.
#     3. Makes a forward pass to find the category index with the highest score,
#     and computes intermediate activations.
#     Makes the visualization. """
#
#     args = get_args()
#
#     # Can work with any model, but it assumes that the model has a
#     # feature method, and a classifier method,
#     # as in the VGG models in torchvision.
#     grad_cam = GradCam(model=models.vgg19(pretrained=True), \
#                        target_layer_names=["35"], use_cuda=args.use_cuda)
#
#     img = cv2.imread(args.image_path, 1)
#     img = np.float32(cv2.resize(img, (224, 224))) / 255
#     input = preprocess_image(img)
#
#     # If None, returns the map for the highest scoring category.
#     # Otherwise, targets the requested index.
#     target_index = None
#     mask = grad_cam(input, target_index)
#
#     show_cam_on_image(img, mask)
#
#     gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), use_cuda=args.use_cuda)
#     gb = gb_model(input, index=target_index)
#     gb = gb.transpose((1, 2, 0))
#     cam_mask = cv2.merge([mask, mask, mask])
#     cam_gb = deprocess_image(cam_mask * gb)
#     gb = deprocess_image(gb)
#
#     cv2.imwrite('gb.jpg', gb)
#     cv2.imwrite('cam_gb.jpg', cam_gb)
