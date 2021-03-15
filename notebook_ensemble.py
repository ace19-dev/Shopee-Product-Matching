from __future__ import print_function

import inspect
import os
import random
import collections

import pandas as pd
from tqdm import tqdm
from scipy.special import softmax

############## TODO:delete
import transformer
from datasets.product import NUM_CLASS, ProductTestDataset
import model as M
# from models import model2 as M2
from training.loss import *

# ############################

global best_pred, acclist_train, acclist_val

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

__cwd__ = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

FOLD_MODELS = [
    'experiments/shopee-product-matching/tf_efficientnet_b4_ns/(2021-03-04_13:28:50)cassava_fold0_576x576_tf_efficientnet_b4_ns_acc(80.57661)_loss(0.07819)_checkpoint1.pth.tar',
    'experiments/shopee-product-matching/tf_efficientnet_b4_ns/(2021-03-04_13:28:50)cassava_fold1_576x576_tf_efficientnet_b4_ns_acc(83.36722)_loss(0.07957)_checkpoint1.pth.tar',
    'experiments/shopee-product-matching/tf_efficientnet_b4_ns/(2021-03-04_13:28:50)cassava_fold2_576x576_tf_efficientnet_b4_ns_acc(81.70393)_loss(0.0773)_checkpoint1.pth.tar',
    'experiments/shopee-product-matching/tf_efficientnet_b4_ns/(2021-03-04_13:28:50)cassava_fold3_576x576_tf_efficientnet_b4_ns_acc(82.95749)_loss(0.07689)_checkpoint1.pth.tar',
    'experiments/shopee-product-matching/tf_efficientnet_b4_ns/(2021-03-04_13:28:50)cassava_fold4_576x576_tf_efficientnet_b4_ns_acc(80.00000)_loss(0.07863)_checkpoint1.pth.tar',
]

FOLD_MODELS_2 = [
    'experiments/shopee-product-matching/tf_efficientnet_b5_ns/(2021-03-04_14:33:12)cassava_fold0_640x640_tf_efficientnet_b5_ns_acc(77.84143)_loss(0.09037)_checkpoint1.pth.tar',
    'experiments/shopee-product-matching/tf_efficientnet_b5_ns/(2021-03-04_14:33:12)cassava_fold1_640x640_tf_efficientnet_b5_ns_acc(79.98521)_loss(0.07774)_checkpoint1.pth.tar',
    'experiments/shopee-product-matching/tf_efficientnet_b5_ns/(2021-03-04_14:33:12)cassava_fold2_640x640_tf_efficientnet_b5_ns_acc(82.16595)_loss(0.07733)_checkpoint1.pth.tar',
    'experiments/shopee-product-matching/tf_efficientnet_b5_ns/(2021-03-04_14:33:12)cassava_fold3_640x640_tf_efficientnet_b5_ns_acc(82.66174)_loss(0.07624)_checkpoint1.pth.tar',
    'experiments/shopee-product-matching/tf_efficientnet_b5_ns/(2021-03-04_14:33:12)cassava_fold4_640x640_tf_efficientnet_b5_ns_acc(82.73567)_loss(0.0776)_checkpoint1.pth.tar',
]

# dataset_root = '/kaggle/input/shopee-product-matching'
dataset_root = '/home/ace19/dl_data/shopee-product-matching'
dataset_name = 'cassava'
basemodel = 'tf_efficientnet_b4_ns'
# basemodel = 'resnet50'
test_batch_size = 1
workers = 4
no_cuda = False
# TODO: set random seed per ensemble
seed = 8

cuda = not no_cuda and torch.cuda.is_available()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# init dataloader
testset = ProductTestDataset(dataset_root,
                             transform=transformer.test_augmentation())
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=workers,
                                          pin_memory=True)

testset2 = ProductTestDataset(dataset_root,
                             transform=transformer.test_augmentation2())
test_loader2 = torch.utils.data.DataLoader(testset2,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=workers,
                                          pin_memory=True)


def inference_func(test_loader):
    model.eval()
    bar = tqdm(test_loader)

    PREDS = []
    PREDS2 = []

    with torch.no_grad():
        for batch_idx, (images, fnames) in enumerate(bar):
            x = images.cuda()
            logits = model(x)
            PREDS2 += [torch.softmax(logits, dim=1).detach().cpu()]
            PREDS += [logits.data.max(1)[1]]
        PREDS = torch.cat(PREDS).cpu().numpy()
        PREDS2 = torch.cat(PREDS2).cpu().numpy()
    return PREDS, PREDS2


def inference_func(test_loader, model, image_size, num_tta):
    model.eval()
    bar = tqdm(test_loader)

    PREDS = []

    with torch.no_grad():
        for batch_idx, (images, fnames) in enumerate(bar):
            x = images.cuda()

            if num_tta == 0:
                logits = model(x)
            elif num_tta == 4:
                x = torch.stack([x, x.flip(-1), x.flip(-2), x.flip(-1, -2)], 0)
                x = x.view(-1, 3, image_size, image_size)
                logits = model(x)
                logits = logits.view(test_batch_size, num_tta, -1).mean(1)
            elif num_tta == 8:
                x = torch.stack([x, x.flip(-1), x.flip(-2), x.flip(-1, -2),
                                 x.transpose(-1, -2), x.transpose(-1, -2).flip(-1),
                                 x.transpose(-1, -2).flip(-2), x.transpose(-1, -2).flip(-1, -2)], 0)
                x = x.view(-1, 3, image_size, image_size)
                logits = model(x)
                logits = logits.view(test_batch_size, num_tta, -1).mean(1)

            # PREDS += [logits.data.max(1)[1]]
            PREDS += [torch.softmax(logits, 1).detach().cpu()]

        PREDS = torch.cat(PREDS).cpu().numpy()

    return PREDS


# test_data = pd.read_csv("/kaggle/input/shopee-product-matching/sample_submission.csv")
test_data = pd.read_csv("/home/ace19/dl_data/shopee-product-matching/sample_submission.csv")
test_images = test_data['image_id'].values

# test_preds = []
# tmp = []
# for i in range(len(FOLD_MODELS)):
#     #     model = enet_v2(enet_type[i], out_dim=5)
#     # init the model
#     model = M.Model(NUM_CLASS, backbone=basemodel)
#     # print(model)
#     model = model.cuda()
#     # load pre-trained model
#     # https://tutorials.pytorch.kr/beginner/saving_loading_models.html
#     # 1. 몇몇 키를 제외하고 state_dict 의 일부를 불러오거나, 적재하려는 모델보다 더 많은 키를 갖고 있는 state_dict 를 불러올 때에는
#     # load_state_dict() 함수에서 strict 인자를 False 로 설정하여 일치하지 않는 키들을 무시하도록 해야 합니다.
#     # 2. 한 계층에서 다른 계층으로 매개변수를 불러오고 싶지만, 일부 키가 일치하지 않을 때에는 적재하려는 모델의 키와 일치하도록 state_dict 의
#     # 매개변수 키의 이름을 변경하면 됩니다.
#     # https: // jangjy.tistory.com / 318
#     pretrained_dict = torch.load(FOLD_MODELS[i])['state_dict']
#     new_model_dict = model.state_dict()
#     pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()
#                        if k[7:] in new_model_dict}
#     new_model_dict.update(pretrained_dict)
#     model.load_state_dict(new_model_dict)
#     # inference
#     # pred, pred2 = inference_func(test_loader)
#     pred = inference_func(test_loader)
#     test_preds += [pred]
#     # tmp += [pred2]
#
# test_preds = np.asarray(test_preds).transpose().tolist()

test_preds = []
for i in range(len(FOLD_MODELS)):
    # init the model
    model = M.Model(NUM_CLASS, backbone='tf_efficientnet_b4_ns')
    # print(model)
    model = model.cuda()
    # load pre-trained model - https://jangjy.tistory.com/318
    # model.load_state_dict(torch.load(FOLD_MODELS[i])['state_dict'], strict=True)
    pretrained_dict = torch.load(FOLD_MODELS[i])['state_dict']
    new_model_dict = model.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()
                       if k[7:] in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)
    # inference
    test_preds += [inference_func(test_loader, model, 576, 8)]

# test_preds = np.asarray(test_preds).transpose().tolist()
test_preds = np.asarray(test_preds)
# print(test_preds.shape)


test_preds2 = []
for i in range(len(FOLD_MODELS_2)):
    # init the model
    model2 = M.Model(NUM_CLASS, backbone='tf_efficientnet_b5_ns')
    # print(model)
    model2 = model2.cuda()
    # load pre-trained model - https://jangjy.tistory.com/318
    # model.load_state_dict(torch.load(FOLD_MODELS[i])['state_dict'], strict=True)
    pretrained_dict = torch.load(FOLD_MODELS_2[i])['state_dict']
    new_model_dict = model2.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()
                       if k[7:] in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model2.load_state_dict(new_model_dict)
    # inference
    test_preds2 += [inference_func(test_loader2, model2, 640, 8)]

# test_preds2 = np.asarray(test_preds2).transpose().tolist()
test_preds2 = np.asarray(test_preds2)

# # https://stackoverflow.com/questions/25815377/sort-list-by-frequency
# ensemble = []
# # ensemble2 = []
# for lst in test_preds:
#     counts = collections.Counter(lst)
#     new_list = sorted(lst, key=counts.get, reverse=True)
#     ensemble.append(new_list[0])
#     # ensemble2.append(int(np.mean(lst)))

total_pred = test_preds + test_preds2
total_pred = np.mean(total_pred, axis=0).tolist()
ensemble = []
for prob in total_pred:
    cls = np.argmax(prob)
    ensemble.append(cls)

# TODO
# pred = 0.5*predictions + 0.5*np.mean(test_preds, axis=0)
# submission
test_data['label'] = ensemble
# test_data['label2'] = ensemble2
test_data[['image_id', 'label']].to_csv('submit/submission.csv', index=False)
