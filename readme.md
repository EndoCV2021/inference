1. 훈련했던 CONFIG과 모델 파일을 준비합니다. 샘플로 제가 했던것 올려 놓았는데 바꿔주세요.
2. make_submission.sh을 열어서 각자 환경에 맞게 인자들을 수정해주세요. 
3. 경로 설정을 다음과 같이 합니다.  
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
4. shell 스크립트 실행
```
bash ./make_submission.sh
```
5. Endocv2021 폴더에 결과가 생성되었는지 확인

TODO.
1. detection 추가
2. 자료 UP/DOWNLOAD 
