# IRIM FoundationPose Project

이 저장소는 [FoundationPose (NVlabs)](https://github.com/NVlabs/FoundationPose)를 기반으로 한 **project_IRIM** 확장 프로젝트입니다.  
데이터 습득 및 제어 연구에 도움이 되길 바라는 마음으로 작성했습니다..   
   
GPT와 함께 열심히 진행해 보았습니다.   

더 진행할 사안들이 많이 남아 있습니다.

---

## 📂 구조 정리

project_IRIM/       
├── data/ # 3D 객체 모델 (예: galaxy.obj)    
├── FoundationPose.py # Foundation Pose 추정     
├── get_K # 초기 K값 추출 및 Realsense 카메라 확인  
└── README.md # 문서


---

## 🚀 시작하기

### 1. PRE-SETUP
```bash
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
git clone https://github.com/HS-P/IRIM_FoundationPose.git
cd IRIM_FoundationPose

2. Realsense Camera 기종별 K값 얻기
python get_K.py

3. Foundation Pose 찾기
python FoundationPose.py
```
---
    
📖 참고 자료

FoundationPose (https://github.com/NVlabs/FoundationPose.git)

