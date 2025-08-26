# IRIM FoundationPose Project

이 저장소는 [FoundationPose (NVlabs)](https://github.com/NVlabs/FoundationPose)를 기반으로 한 **project_IRIM** 확장 프로젝트입니다.  
제어 및 강화학습 연구를 목적으로 하며, IRIM 연구실 실험 환경에 맞추어 커스터마이징되었습니다.

---

## 📂 저장소 구조

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

