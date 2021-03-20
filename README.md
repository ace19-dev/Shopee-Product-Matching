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

# Trial
- try image retrieval model. -> notebook timeout, Notebook Exceeded Allowed Compute
  - class num: 11014
  - use train data for validate above model because test data was hidden.
- kaggle notebook 샘플 + 학습된 model (effib4 with class num: 11014)
  - https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700
  - ViT (??)
- TODO: kaggle notebook 샘플
- find other model through some paper
  - use ocr

- 
