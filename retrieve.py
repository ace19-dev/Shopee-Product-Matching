'''
TODO:
'''

import os
import random
import numpy as np
from tqdm import tqdm
import time
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

from datasets.product import ProductTestDataset
import model as M
import transformer
from option import Options

import matching as matching

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

TOP_N = 50
RESULT_PATH = 'results'


def _print_distances(distance_matrix, top_n_indice):
    distances = []
    num_row, num_col = top_n_indice.shape
    for r in range(num_row):
        col = []
        for c in range(num_col):
            col.append(distance_matrix[r, top_n_indice[r, c]])
        distances.append(col)

    return distances


# TODO: 특정 거리를 설정해서 top 50 top
def match_n(top_n, galleries, queries):
    # The distance metric used for measurement to query.
    metric = matching.NearestNeighborDistanceMetric("cosine")
    start = time.time()
    distance_matrix = metric.distance(queries, galleries)
    end = time.time()
    print("distance measure time: {}".format(end - start))

    # top_indice = np.argmin(distance_matrix, axis=1)
    # top_n_indice = np.argpartition(distance_matrix, top_n, axis=1)[:, :top_n]
    # top_n_dist = _print_distances(distance_matrix, top_n_indice)
    # top_n_indice2 = np.argsort(top_n_dist, axis=1)
    # dist2 = _print_distances(distance_matrix, top_n_indice2)

    # TODO: need improvement.
    top_n_indice = np.argsort(distance_matrix, axis=1)[:, :top_n]
    top_n_distance = _print_distances(distance_matrix, top_n_indice)

    return top_n_indice, top_n_distance


# TODO: use the top_n_distance to select a similar items
# def show_retrieval_result(top_n_indice, top_n_distance,
#                           gallery_path_list, gallery_posid_list, gallery_label_list,
#                           query_path_list, query_posid_list, query_label_list):
#     # for real submit
#     # test_data = pd.read_csv("/home/ace19/dl_data/shopee-product-matching/sample_submission.csv")
#     # test_images = test_data['posting_id'].values.tolist()
#
#     matches = []
#     query_posids = []
#     query_paths = []
#     query_labels = []
#
#     gallery_posids = []
#     gallery_paths = []
#     gallery_labels = []
#
#     col = top_n_indice.shape[1]
#     for row_idx, query_img_path in enumerate(query_path_list):
#         # fig, axes = plt.subplots(ncols=51, figsize=(15, 4))
#         # # fig.suptitle(query_img_path.split('/')[-1], fontsize=12, fontweight='bold')
#         # axes[0].set_title(query_img_path.split('/')[-1] + '_' + query_label_list[row_idx],
#         #                   color='r', fontweight='bold')
#         # axes[0].imshow(Image.open(query_img_path))
#         query_posids.append(query_posid_list[row_idx])
#         query_paths.append(query_img_path.split('/')[-1])
#         query_labels.append(query_label_list[row_idx])
#
#         posids = []
#         paths = []
#         labels = []
#         for i in range(col):
#             # TODO: fix
#             # if top_n_distance[row_idx, i] < 1.0:
#             #     continue
#
#             posids.append(gallery_posid_list[top_n_indice[row_idx, i]])
#             # posids.append(gallery_posid_list[top_n_indice[row_idx, i]])
#             paths.append(gallery_path_list[top_n_indice[row_idx, i]])
#             labels.append(gallery_label_list[top_n_indice[row_idx, i]])
#
#             # img_path = gallery_path_list[top_n_indice[row_idx, i]]
#             # axes[i + 1].set_title(img_path.split('/')[-1])
#             # axes[i + 1].set_title(gallery_label_list[row_idx])
#             # axes[i + 1].imshow(Image.open(img_path))
#
#         gallery_posids.append(' '.join(posids))
#         gallery_paths.append(paths)
#         gallery_labels.append(labels)
#
#         print(" Retrieval result {} create.".format(row_idx + 1))
#         # fig.savefig(os.path.join(RESULT_PATH, query_img_path.split('/')[-1]))
#         # plt.close()
#
#     # # TODO: for real submit
#     # test_df['query_posting_id'] = query_posids
#     # test_df['query_path'] = query_paths
#     # test_df['query_label'] = query_labels
#     # test_df['gallery_posting_id'] = ' '.join(gallery_posids)
#     # test_df['gallery_path'] = ' '.join(gallery_paths)
#     # test_df['gallery_label'] = ' '.join(gallery_labels)
#     # # test_df['matches'] = matches
#     # test_df[['posting_id', 'query_posting_id', 'query_path', 'query_label',
#     #            'gallery_posting_id', 'gallery_path', 'gallery_label']].to_csv('submit/submission.csv', index=False)
#
#     # test_df = pd.DataFrame({'query_posting_id': query_posids, 'query_path': query_paths, 'query_label': query_labels,
#     #                         'gallery_posting_id': gallery_posids, 'gallery_path': gallery_paths,
#     #                         'gallery_label': gallery_labels})
#     test_df = pd.DataFrame({'posting_id': query_posids, 'matches': gallery_posids})
#     test_df.to_csv('submit/submission.csv', index=False)


def show_retrieval_result(top_n_indice, top_n_distance,
                          gallery_path_list, gallery_posid_list,
                          query_path_list, query_posid_list):
    test_df = pd.read_csv("/home/ace19/dl_data/shopee-product-matching/sample_submission.csv")
    test_images = test_df['posting_id'].values.tolist()

    query_posids = []
    gallery_posids = []

    col = top_n_indice.shape[1]
    for row_idx, query_img_path in enumerate(query_path_list):
        # fig, axes = plt.subplots(ncols=51, figsize=(15, 4))
        # # fig.suptitle(query_img_path.split('/')[-1], fontsize=12, fontweight='bold')
        # axes[0].set_title(query_img_path.split('/')[-1] + '_' + query_label_list[row_idx],
        #                   color='r', fontweight='bold')
        # axes[0].imshow(Image.open(query_img_path))
        query_posids.append(query_posid_list[row_idx])
        # query_paths.append(query_img_path.split('/')[-1])
        # query_labels.append(query_label_list[row_idx])

        posids = []
        # paths = []
        # labels = []
        for i in range(col):
            # TODO: fix
            # if top_n_distance[row_idx, i] < 1.0:
            #     continue

            posids.append(gallery_posid_list[top_n_indice[row_idx, i]])
            # paths.append(gallery_path_list[top_n_indice[row_idx, i]])
            # labels.append(gallery_label_list[top_n_indice[row_idx, i]])

            # img_path = gallery_path_list[top_n_indice[row_idx, i]]
            # axes[i + 1].set_title(img_path.split('/')[-1])
            # axes[i + 1].set_title(gallery_label_list[row_idx])
            # axes[i + 1].imshow(Image.open(img_path))

        gallery_posids.append(' '.join(posids))
        # gallery_paths.append(paths)
        # gallery_labels.append(labels)

        # print(" Retrieval result {} create.".format(row_idx + 1))
        # fig.savefig(os.path.join(RESULT_PATH, query_img_path.split('/')[-1]))
        # plt.close()

    test_df['matches'] = gallery_posids
    test_df[['posting_id', 'matches']].to_csv('submit/submission.csv', index=False)

    # test_df = pd.DataFrame({'query_posting_id': query_posids, 'query_path': query_paths, 'query_label': query_labels,
    #                         'gallery_posting_id': gallery_posids, 'gallery_path': gallery_paths,
    #                         'gallery_label': gallery_labels})
    # test_df = pd.DataFrame({'posting_id': query_posids, 'matches': gallery_posids})
    # test_df.to_csv('submit/submission.csv', index=False)


def main():
    global best_pred, acclist_train, acclist_val

    args, _ = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    galleryset = ProductTestDataset(data_dir=args.dataset_root,
                                    csv=['test.csv'],
                                    transform=transformer.test_augmentation())
    # queryset = ProductTestDataset(data_dir=args.dataset_root,
    #                               csv=['test.csv'],
    #                               transform=transformer.test_augmentation())
    gallery_loader = torch.utils.data.DataLoader(galleryset, batch_size=1, num_workers=args.workers)
    # query_loader = torch.utils.data.DataLoader(queryset, batch_size=1, num_workers=args.workers)

    # init the model
    model = M.Model(backbone=args.model)
    # model.half()  # to save space.
    print('\n-------------- model details --------------')
    print(model)

    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acc_lst_train']
            acclist_val = checkpoint['acc_lst_val']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no infer checkpoint found at '{}'". \
                               format(args.resume))
    else:
        raise RuntimeError("=> config \'args.resume\' is '{}'". \
                           format(args.resume))

    gallery_features_list = []
    gallery_path_list = []
    gallery_posid_list = []
    # gallery_label_list = []
    query_features_list = []
    query_path_list = []
    query_posid_list = []

    # query_label_list = []

    from sklearn.neighbors import NearestNeighbors

    def retrieval():
        model.eval()

        print(" ==> Loading gallery ... ")
        tbar = tqdm(gallery_loader, desc='\r')
        for batch_idx, (data, pos_id, img_path) in enumerate(tbar):
            if args.cuda:
                data = data.cuda()

            with torch.no_grad():
                features, output = model(data)

                # TTA
                # batch_size, n_crops, c, h, w = data.size()
                # # fuse batch size and ncrops
                # features, _ = model(data.view(-1, c, h, w))
                # # avg over crops
                # features = features.view(batch_size, n_crops, -1).mean(1)
                gallery_features_list.extend(features)
                gallery_path_list.extend(img_path)
                gallery_posid_list.extend(pos_id)
                # gallery_label_list.extend(gt)
        # end of for

        # print("\n ==> Loading query ... ")
        # tbar = tqdm(query_loader, desc='\r')
        # for batch_idx, (data, pos_id, img_path) in enumerate(tbar):
        #     if args.cuda:
        #         data = data.cuda()
        #
        #     with torch.no_grad():
        #         features, output = model(data)
        #         # # TTA
        #         # batch_size, n_crops, c, h, w = data.size()
        #         # # fuse batch size and ncrops
        #         # features, _ = model(data.view(-1, c, h, w))
        #         # # avg over crops
        #         # features = features.view(batch_size, n_crops, -1).mean(1)
        #         query_features_list.extend(features)
        #         query_path_list.extend(img_path)
        #         query_posid_list.extend(pos_id)
        #         # query_label_list.extend(gt)
        # # end of for

        print("\n ==> Copy query ... ")
        query_features_list = gallery_features_list.copy()
        query_path_list = gallery_path_list.copy()
        query_posid_list = gallery_posid_list.copy()

        if len(query_features_list) == 0:
            print('No query data!!')
            return

        # matching
        top_n_indice, top_n_distance = \
            match_n(TOP_N,
                    torch.stack(gallery_features_list).cpu(),
                    torch.stack(query_features_list).cpu())

        # Show n images from the gallery similar to the query image.
        show_retrieval_result(top_n_indice, top_n_distance,
                              gallery_path_list, gallery_posid_list,
                              query_path_list, query_posid_list)

    retrieval()


if __name__ == "__main__":
    main()
