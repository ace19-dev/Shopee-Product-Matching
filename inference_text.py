import random
import numpy as np, pandas as pd, gc
import cv2, matplotlib.pyplot as plt
from tqdm import tqdm
import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer, CountVectorizer
from cuml.neighbors import NearestNeighbors

print('RAPIDS', cuml.__version__)

import torch

from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

from option import Options
from datasets.product_text import ProductTextDataset, NUM_CLASS
from models import text_model as M


# TODO: https://github.com/fyang93/diffusion

# Function to get 50 nearest neighbors of each image and apply a distance threshold to maximize cv
def get_neighbors(df, embeddings, KNN=50, threshold=0.0):
    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    #     print('distances: ', distances)

    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = cupy.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = df['posting_id'].iloc[cupy.asnumpy(ids)].values
        predictions.append(posting_ids)

    del model, distances, indices
    gc.collect()
    return predictions


def main():
    global best_pred, acc_lst_train, acc_lst_val

    args, args_desc = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    args.seed = random.randrange(999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.model == 'DistilBERT':
        model_name = 'cahya/distilbert-base-indonesian'
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        bert_model = DistilBertModel.from_pretrained(model_name)
    else:
        model_name = 'cahya/bert-base-indonesian-522M'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)
    print(tokenizer)

    inferset = ProductTextDataset(data_dir=args.dataset_root,
                                  csv=['train.csv'],
                                  tokenizer=tokenizer)
    infer_loader = torch.utils.data.DataLoader(inferset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    model = M.Model(bert_model, num_classes=NUM_CLASS)
    print(model)

    model.eval()

    features_lst = []
    tbar = tqdm(infer_loader, desc='\r')
    for batch in tbar:
        batch = {k: v.cuda() for k, v in batch.items()}
        # if args.cuda:
        #     input_ids, input_mask, labels = \
        #         input_ids.cuda(), input_mask.cuda(), labels.cuda()

        with torch.no_grad():
            # ArcFace
            features = model(batch)
            features_lst.append(features.cpu().numpy())

    text_embeddings = np.concatenate(features_lst)
    _ = gc.collect()

    print('text embeddings shape', text_embeddings.shape)
    text_embeddings = cupy.array(text_embeddings)

    test = pd.read_csv('../input/shopee-product-matching/test.csv')
    text_predictions = get_neighbors(test, text_embeddings, KNN=100, threshold=4.5)
    test['text_embeddings'] = text_predictions


if __name__ == '__main__':
    main()
