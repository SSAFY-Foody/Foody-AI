# 모델 테스트 기준 문서 (testref.md)

## 테스트 개요

이 문서는 베이스 Qwen2.5-VL-3B-Instruct 모델과 한국 음식 데이터셋으로 파인튜닝한 모델의 성능을 비교하는 테스트 기준을 정의합니다.

## 테스트 데이터셋

- **파일**: `test.csv`
- **샘플 수**: 150개 (각 음식 라벨당 1개)
- **음식 라벨 수**: 150개
- **데이터 분할**: 학습에 사용되지 않은 이미지만 포함

## 평가 모델

### 1. 베이스 모델 (Base Model)
- **모델명**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **설명**: 파인튜닝 이전의 사전학습된 모델
- **용도**: 파인튜닝 효과를 측정하기 위한 기준선(baseline)

### 2. 파인튜닝 모델 (Fine-tuned Model)
- **모델 경로**: `./qwen25_kfood_output/final/`
- **설명**: 한국 음식 데이터셋으로 LoRA 파인튜닝한 모델
- **학습 데이터**: `train_2.csv` (3,000개 이미지)

## 평가 기준

### 정확도 계산 방식

**기본 방식 (Contains Match)**:
```python
is_correct = (true_label in predicted_text)
```
- 모델의 응답에 정답 라벨이 포함되어 있으면 정답으로 간주
- 예시: 
  - 정답: "김치찌개"
  - 예측: "이 이미지는 김치찌개입니다" → ✓ 정답
  - 예측: "김치찌개" → ✓ 정답
  - 예측: "된장찌개" → ✗ 오답

**엄격한 방식 (Exact Match)** (참고용):
```python
is_correct_exact = (predicted_text == true_label)
```
- 모델의 응답이 정답 라벨과 정확히 일치해야 정답
- 결과 파일에 함께 기록되지만 주요 평가 지표는 아님

### 평가 지표

1. **전체 정확도 (Overall Accuracy)**
   ```
   Accuracy = (정답 수 / 전체 샘플 수) × 100%
   ```

2. **라벨별 정확도 (Per-Label Accuracy)**
   - 각 음식 종류별 정확도
   - 테스트셋에서는 각 라벨당 1개씩만 있으므로 0% 또는 100%

3. **개선도 (Improvement)**
   ```
   Improvement = 파인튜닝 모델 정확도 - 베이스 모델 정확도
   ```

## 테스트 프로토콜

### 테스트 프롬프트
```
"이 이미지에 있는 한국 음식의 이름을 정확히 답하세요."
```
- 모든 모델에 동일한 프롬프트 사용
- 한국어로 질문하여 한국 음식 이름을 한국어로 답변하도록 유도

### 추론 설정
```python
max_new_tokens = 50
do_sample = False
temperature = None  # deterministic
```
- 결정론적(deterministic) 추론으로 재현성 보장
- 샘플링 비활성화

### 테스트 실행 순서
1. 베이스 모델 평가
2. GPU 메모리 정리
3. 파인튜닝 모델 평가
4. 결과 비교 및 저장

## 성공 기준

### 최소 목표
- **파인튜닝 모델 정확도**: ≥ 60%
- **개선도**: ≥ +20%p (베이스 모델 대비)

### 우수 목표
- **파인튜닝 모델 정확도**: ≥ 80%
- **개선도**: ≥ +40%p (베이스 모델 대비)

### 최상 목표
- **파인튜닝 모델 정확도**: ≥ 90%
- **개선도**: ≥ +50%p (베이스 모델 대비)

## 출력 파일

테스트 완료 후 `./test_results/` 디렉토리에 다음 파일이 생성됩니다:

### 1. `base_results.json`
베이스 모델의 상세 결과:
```json
{
  "model_name": "Base Model",
  "accuracy": 35.5,
  "correct": 53,
  "total": 150,
  "results": [
    {
      "image_path": "dataset/kfood/.../...",
      "true_label": "김치찌개",
      "predicted": "이것은 김치찌개입니다",
      "correct_exact": false,
      "correct_contains": true
    },
    ...
  ],
  "label_accuracies": {
    "김치찌개": 100.0,
    "된장찌개": 0.0,
    ...
  }
}
```

### 2. `finetuned_results.json`
파인튜닝 모델의 상세 결과 (동일한 형식)

### 3. `comparison.json`
두 모델의 비교 요약:
```json
{
  "base_accuracy": 35.5,
  "finetuned_accuracy": 82.3,
  "improvement": 46.8,
  "test_samples": 150
}
```

## 테스트 실행 방법

### 기본 실행
```bash
python test.py
```
- 베이스 모델과 파인튜닝 모델 모두 평가

### 파인튜닝 모델만 테스트
```bash
python test.py --skip_base
```

### 베이스 모델만 테스트
```bash
python test.py --skip_finetuned
```

### 커스텀 경로 설정
```bash
python test.py \
  --finetuned_path ./my_model/final \
  --test_csv ./my_test.csv \
  --output_dir ./my_results
```

## 결과 분석 가이드

### 1. 전체 정확도 확인
```bash
cat test_results/comparison.json
```
- 개선도가 양수인지 확인
- 목표 정확도 달성 여부 확인

### 2. 오분류 샘플 분석
```python
import json

# 파인튜닝 모델 결과 로드
with open('test_results/finetuned_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 틀린 샘플 찾기
errors = [r for r in results['results'] if not r['correct_contains']]

# 오분류 패턴 분석
for err in errors:
    print(f"True: {err['true_label']}")
    print(f"Predicted: {err['predicted']}")
    print(f"Image: {err['image_path']}")
    print("-" * 40)
```

### 3. 라벨별 성능 확인
```python
# 성능이 낮은 라벨 찾기
label_accs = results['label_accuracies']
poor_labels = {k: v for k, v in label_accs.items() if v < 50}
print("성능이 낮은 음식 종류:", poor_labels)
```

## 주의사항

### GPU 메모리
- 테스트 시 약 8-10GB GPU 메모리 필요
- 메모리 부족 시 한 번에 하나의 모델만 테스트

### 재현성
- `do_sample=False`로 설정하여 동일한 결과 보장
- 동일한 test.csv 사용 시 결과가 일관됨

### 평가 지표 해석
- Contains Match는 관대한 기준이지만 실용적
- 모델이 추가 설명과 함께 정답을 제공하는 경우 포착
- 엄격한 평가가 필요한 경우 `correct_exact` 참조

## 문제 해결

### 테스트가 너무 느림
```bash
# 샘플 수를 줄인 테스트 CSV 생성
head -21 test.csv > test_small.csv  # 헤더 + 20 샘플
python test.py --test_csv test_small.csv
```

### GPU OOM 에러
```bash
# 한 번에 하나씩만 테스트
python test.py --skip_finetuned  # 베이스만
python test.py --skip_base       # 파인튜닝만
```

### 정확도가 예상보다 낮음
1. 학습이 충분히 진행되었는지 확인 (loss 수렴 여부)
2. 테스트 이미지 품질 확인
3. 라벨별 성능 분석하여 특정 음식 종류에 문제가 있는지 확인
4. 학습 데이터에 해당 음식이 충분히 포함되어 있는지 확인

### 사용 명령어
예시 : python test.py --finetuned_path ./qwen25_v7/final