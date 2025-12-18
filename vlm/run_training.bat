@echo off
REM GPU 메모리 최적화를 위한 환경 변수 설정
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set CUDA_LAUNCH_BLOCKING=0

REM 학습 실행
python train_gyeranmari.py

pause
