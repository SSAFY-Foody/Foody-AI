# 테스트 실행 가이드

## 빠른 시작

### 1. 베이스 모델과 파인튜닝 모델 모두 테스트
```bash
python test.py
```

### 2. 파인튜닝 모델만 테스트 (더 빠름)
```bash
python test.py --skip_base
```

### 3. 결과 확인
```bash
cat test_results/comparison.json
```

## 테스트 결과 예시

### comparison.json
```json
{
  "base_accuracy": 35.5,
  "finetuned_accuracy": 82.3,
  "improvement": 46.8,
  "test_samples": 150
}
```

### 결과 해석
- **베이스 모델**: 35.5% (150개 중 53개 정답)
- **파인튜닝 모델**: 82.3% (150개 중 123개 정답)
- **개선도**: +46.8%p 향상 ✓

## 상세 분석

### 틀린 샘플 확인
```python
import json

with open('test_results/finetuned_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 오답만 필터링
errors = [r for r in results['results'] if not r['correct_contains']]

for err in errors[:5]:  # 처음 5개만
    print(f"이미지: {err['image_path']}")
    print(f"정답: {err['true_label']}")
    print(f"예측: {err['predicted']}")
    print("-" * 50)
```

### 라벨별 정확도
```python
# 성능이 낮은 음식 찾기
label_accs = results['label_accuracies']
for label, acc in sorted(label_accs.items(), key=lambda x: x[1]):
    if acc < 100:
        print(f"{label}: {acc}%")
```

## 필요한 리소스

- **GPU 메모리**: ~8-10GB
- **실행 시간**: 
  - 베이스 모델만: ~5-10분
  - 파인튜닝 모델만: ~5-10분
  - 둘 다: ~10-20분

## 문제 해결

### GPU 메모리 부족
```bash
# 한 번에 하나씩
python test.py --skip_finetuned  # 베이스만
python test.py --skip_base       # 파인튜닝만
```

### 빠른 테스트 (샘플 줄이기)
```bash
head -21 test.csv > test_small.csv
python test.py --test_csv test_small.csv
```
