1. 경로 설정을 다음과 같이 합니다.  
.  
├── inference.py  
├── make_submission.sh  
├── model  
│   ├── Base-RCNN-FPN.yaml  
│   ├── Mask_RCNN_Res50_FPN_3x.yaml  
│   └── model_0040999.pth  
├── readme.md  
├── requirements.txt  
└── sample_data  
    ├── EndoCV_DATA1  
    │   ├── Endocv2021_test_data1_0.jpg  
    │   ├── Endocv2021_test_data1_1.jpg  
    │   ├── Endocv2021_test_data1_2.jpg  
    │   └── Endocv2021_test_data1_3.jpg  
    ├── EndoCV_DATA2  
    │   ├── Endocv2021_test_data2_0.jpg  
    │   ├── Endocv2021_test_data2_1.jpg  
    │   ├── Endocv2021_test_data2_2.jpg  
    │   └── Endocv2021_test_data2_3.jpg  
    └── EndoCV_DATA3  
        ├── Endocv2021_test_data3_0.jpg  
        ├── Endocv2021_test_data3_1.jpg  
        ├── Endocv2021_test_data3_2.jpg  
        └── Endocv2021_test_data3_3.jpg  
1. shell 스크립트 실행
```
bash ./make_submission.sh
```
1. Endocv2021 폴더에 결과가 생성되었는지 확인


