# Shopee-Product-Matching
Kaggle Competition: Determine if two products are the same by their images

# EDA (Exploratory Data Analysis)
- 'label_group' 항목이 GT?, 11014 개가 class 개수?
- 'image' 항목은 같은데 'label_group' 항목이 다르다.(multi label 문제?) -> 일부 소수의 이미지만, label noisy 문제.
  - total:  34250
  - train shape:  (34250, 5) 
  - unique posting id:  34250
  - unique image:  32412
  - unique image phash:  28735
  - unique title:  33117
  - unique label group:  11014
  
# Things to think about.
- big loss -> 클래스가 많아서? 상대적으로 낮은 validation loss -> 왜?
- how to solve big label classification?
- CV 리뷰, make fold 재 확인 필요.
- LR Scheduler 변경.

# Trial
- try image retrieval model. -> notebook timeout, Notebook Exceeded Allowed Compute
  - class num: 11014
  - use train data for validate above model because test data was hidden.
- kaggle notebook 샘플 + 학습된 model (effib4 with class num: 11014)
  - https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700
  - ViT (??)
- 기존 대회 데이터로 pretrained model 획득.
  - https://www.kaggle.com/c/shopee-product-matching/discussion/227671
- ArcFace 적용 - 테스트 진행 중.
- find other model through paper
  - use ocr

- lessoned learn
  - 왜 infer 할때 마다 embedding 값이 다른가에 대한 이유 -> 잘못 사용한 TTA
  : 올바른 TTA 사용법.
  with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.cuda()
            label = label.cuda()
            batch_size = img.shape[0]
            
            # TTA 5
            TTA = [img, img.flip(-1), img.flip(-2),
                   img.transpose(-1,-2), img.transpose(-1,-2).flip(-1)]
            img = torch.stack(TTA, 0)
            img = img.view(-1, 3, CFG.img_size, CFG.img_size)
            
            feat = model(img,label)
            feat = feat.view(len(TTA), batch_size, -1).mean(0)
#             feat = feat.view(len(TTA), batch_size, -1).max(0)[0]
            
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

# Wrapup
- https://www.kaggle.com/c/shopee-product-matching/discussion/238033?sort=published 참고할 것.
- The best trick in any competition is to analyze your OOF from your CV. 
After doing lots of OOF EDA, i discovered that my simple boundary wasn't getting as many matches as it could.
Ask yourself questions like "what probability does a product need to be in order to include and increase 
F1 metric?", next ask yourself, "can i compute probabilities of matches from my OOF?". 
Afterward you have a matrix of probabilities and it is clear that your decision boundary can be improved.
