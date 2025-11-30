# Experimento MMPose(descartado)

En este notebook intento procesar los landmarks de videos de LSA64 usando MMPose para comparar su performance con mediapipe
## Estado
- Descartado por lentitud, corre a menos de 1fps en una quadro t1000
## Como correrlo
```bash
conda env create -f environment.yml
conda activate mmpose-env
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
jupyter notebook