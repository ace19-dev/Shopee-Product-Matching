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
- 

# Trial
- try image retrieval method.
  - class num: 11014
