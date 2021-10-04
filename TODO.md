# 09/27 ~ 09/29 : 강의 듣기 및  

# 09/30

## 1. Augmentation for Image detection 공부 및 적용할 Aug 찾아서 정리하기 
###  Q. 막상 적용된거 보면 성능 향상이 될 건지는 잘 모르겠다. 향상이 될까? 
(back bone이 학습 되나?)

** Color Augmentation **

** Geometric Augmentation ** 

    참고 자료 : 
    ㄴ https://github.com/Jasonlee1995/AutoAugment_Detection
<br>

## 2. BaseCode로 epoch_26_tf_efficientdet_d1, epoch_50_tf_efficientdet_d3 모델을 추츨했다. 이를 이용해서 앙상블 for Image detection을 공부 및 적용해보자. 
<br>

## 3. Pytorch_Template를 수정해서 위를 적용할 수 있도록 고쳐 보자.

<br>

## Q. inference할때 image Dection, classfication에서 shuffle에 따라 차이가 심하다. False 순서대로 가져오는것, True 섞어서 가죠오는건데  이것으로 인해서 결과가 바꿔진다?

<br>

## Q. mmdetection에서 True 변경하고 하면 AssertionError 발생 


<br>
<br>

# 10/01
## 1. mmdetection 사용해 보기
## 2. test_set, valude_set으로 만들어 보자!
## 3. mmdetection - model을 수정해보도록 하자!