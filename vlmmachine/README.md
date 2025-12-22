# VLM 로직

- 목차

# 전체 추론 파이프라인

```jsx
사용자 이미지 업로드
      ↓
[1] VLM 음식명 예측 (Qwen2.5-VL)
      ↓
[2] RDB Exact Match (MySQL foods 테이블)
      ↓ (실패 시)
[3] RAG 유사 음식 검색 (ChromaDB + Sentence Transformer)
      ↓ (실패 시)
[4] LLM 기반 영양 추정 (Qwen VLM, Fallback)
      ↓
영양 정보 JSON 응답
```

### Response Json

```json
{
  "name": "김밥",
  "standard": "100g",
  "kcal": 154.2,
  "carb": 32.5,
  "protein": 4.8,
  "fat": 2.1,
  "sugar": 3.2,
  "natrium": 587.0
}
```

---

# Prompting

- VLM의 불안정한 출력 방지
- 사용자에게 항상 유효한 응답 보장

### 1. 2단계 재시도 전략 (Two-Stage Fallback)

> VLM이 가끔 “-”, “?”, “없음” 같은 무효한 출력을 생성하기 위해 2단계 프롬프트 진행
1차가 실패하면 더 짧고 간결한 프롬프트로 재시도
> 

**1차 프롬프트**

```jsx
prompt1 = (
            "너는 한국 음식 이미지 분류기다.\n"
            "이미지에서 '가장 중심이 되는 음식 1개'의 이름만 한국어로 출력해라.\n\n"
            "출력 규칙(매우 중요):\n"
            "1) 한국어 음식명만 출력\n"
            "2) 조사/문장/설명 금지 (예: '입니다', '같아요', '.' 금지)\n"
            "3) 따옴표/괄호/슬래시/이모지 금지\n"
            "4) 공백 없이 음식명만 출력 (최대 12자)\n\n"
            "예시(정답 형식):\n"
            "김밥\n"
            "계란말이\n"
            "떡볶이\n\n"
            "혼동 주의 규칙:\n"
            "- 계란말이: 달걀을 말아 네모/원통 형태, 단면이 말린 층\n"
            "- 오믈렛: 접힌 형태, 둥글고 납작함\n\n"
            "중요: 출력할 음식명이 없다고 판단되더라도 '-', '없음', '?'를 출력하지 말고\n"
            "가장 유사한 한국 음식명 1개를 반드시 출력하라.\n\n"
            "정답(음식명만):"
        )
```

**2차 프롬프트**

```jsx
prompt2 = (
            "너는 한국 음식 이미지 분류기다.\n"
            "모르겠어도 '-', '?', '없음'을 출력하지 마라.\n"
            "가장 유사한 한국 음식명 1개를 반드시 출력하라.\n"
            "음식명만 출력(설명 금지).\n"
            "정답:"
        )
```

**검증 로직**

```python
def _is_valid_food_name(self, s: str) -> bool:
        if not s:
            return False
        s = s.strip()

        # 흔한 실패 토큰들
        if s in {"-", "—", "_", "?", "없음", "모름", "알수없음", "알 수 없음", "unknown"}:
            return False

        # 한글이 1글자 이상은 있어야 음식명으로 취급
        if not re.search(r"[가-힣]", s):
            return False

        return True
```

### 2. 영어 한글 변환

> VLM 이 가끔 영어로 답변하는 경우를 대비하여 후처리 방식의 프롬프팅 채택
> 

```python
translations = {
            "omelette": "오믈렛",
            "omelet": "오믈렛",
            "eggroll": "계란말이",
            "egg roll": "계란말이",
            "koreaneggroll": "계란말이",
            "korean egg roll": "계란말이",
            "rolledegg": "계란말이",
            "rolled egg": "계란말이",
            "salad": "샐러드",
            "rice": "밥",
            "kimchi": "김치",
            "kimbap": "김밥",
            "ramen": "라면",
        }
```

---

# 영양 추론

> 음식명을 얻은 후, 영양정보를 3단계 영양 정보 추론
> 

### 1. RDB Match

- foods 테이블에서 정확히 일치하는 음식 검색
- 공백 무시를 통해 계란 말이, 계란말이와 같은 단어 동일하게 처리

```python
food = db.query(Foods).filter(
    func.replace(Foods.name, " ", "") == normalized_name
).first()

if food:
    # DB 데이터 직접 사용 (가장 정확)
    return food.kcal, food.carb, food.protein, ...
```

- MySQL `foods` 테이블에서 **정확히 일치하는 음식** 검색
- 공백 무시 비교: "계란 말이" vs "계란말이" 동일하게 처리
- **정확도 100%** (DB에 있는 음식은 공식 영양 데이터 사용)

### 2. RAG

- DB 에 없는 음식은 벡터 유사도 검색으로 유사한 음식 찾기
- LLM 모델의 환각 현상을 방지하기 위한 기능

- **RAG 동작 과정**
1. 음식 키워드와 벡터 유사도가 높은 Top-3 가져옴

```python
def rag_estimate_nutrition(
    food_name: str,
    top_k: int = 3,
    hard_threshold: float = 1.0,     
    soft_threshold: float = 0.75,
    eps: float = 1e-6
) -> Optional[dict]:
    """
    - Top-K 검색
    """
    result = food_collection.query(query_texts=[food_name], n_results=top_k)
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    documents = result.get("documents", [[]])[0]
```

1. 벡터 거리가 가까운 단어끼리 가중평균화하여 안정화 작업
- 이때 가장 가까운 1등도 너무 멀면 가중평균하지말고 LLM Fallback 진행
- `hard_threshold=1.0` : 이 이상이면 “아예 엉뚱한 매치” 가능성이 커서 LLM로 
(RAG 를 쓸지 말지 강한 기준)
- `soft_threshold=0.7~0.8` : “비슷한 후보들”만 평균에 참여시키는 필터
(RAG 를 쓸때 어디까지 믿고 사용할지 전하는 기준)

```python
    # 1번: hard threshold 넘으면 RAG 폐기
    if best_distance > hard_threshold:
        print(f"[WARN] Best match distance ({best_distance:.4f}) > hard_threshold ({hard_threshold}).")
        print("[WARN] Discarding RAG result -> fallback to LLM.")
        return None

    # 2번: soft_threshold 안쪽만 평균에 참여 (없으면 Top-1)
    candidates = []
    for meta, dist in zip(metadatas, distances):
        if dist <= soft_threshold:
            candidates.append((meta, dist))

    # 후보가 너무 없으면 Top-1만 사용
    if not candidates:
        print(f"[INFO] No candidates within soft_threshold ({soft_threshold}). Using Top-1 only.")
        return {
            "standard": best_meta.get("standard", "100g"),
            "kcal": float(best_meta.get("kcal", 0)),
            "carb": float(best_meta.get("carb", 0)),
            "protein": float(best_meta.get("protein", 0)),
            "fat": float(best_meta.get("fat", 0)),
            "sugar": float(best_meta.get("sugar", 0)),
            "natrium": float(best_meta.get("natrium", 0)),
        }
```

- RAG 구축 과정
1. 서버 시작 시 `foods` 테이블 전체를 Chroma에 인덱싱
2. 한국어 임베딩 모델: `jhgan/ko-sroberta-multitask`
3. 음식명(`name`) + 기준량(`standard`)를 문서로 저장
4. 영양 정보를 메타데이터로 저장

### 임베딩 모델 "jhgan/ko-sroberta-multitask”

```python
CHROMA_DB_DIR = os.getenv("FOODY_CHROMA_DIR", "./chroma_foods")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=os.getenv("FOODY_EMBED_MODEL", "jhgan/ko-sroberta-multitask")
)

food_collection = chroma_client.get_or_create_collection(
    name="food_nutrition",
    embedding_function=ko_embedding,
)
```

- 한국어 임베딩 모델로 텍스트를 벡터로 바꿔서 검색 가능하게 함

### LLM Fallback

- RAG 검색 실패시 Qwen VLM 에게 직접 영양 정보 생성 요청

```python
prompt = f"""
당신은 전문 영양학자입니다.
"{food_name}" 음식의 100g 기준 영양성분을 추정하세요.

❗ 반드시 아래 JSON 형식으로만 출력하세요.
❗ 설명, 문장, 코드블록, 여분의 텍스트는 절대로 넣지 마세요.
❗ 모든 수치는 number 로 출력하세요 (따옴표 금지)

예시 출력:
{{
  "standard": "100g",
  "kcal": 154,
  "carb": 3.2,
  "protein": 11.2,
  "fat": 10.1,
  "sugar": 1.1,
  "natrium": 250
}}
"""
```

- **정규 표현식**
- LLM 답변이 코드블록 (```json)으로 감싸는 경향이 있음
- 부가적인 설명이 추가하는 경향을 막기 위해 어떤 형태로 오든 JSON 부분만 추출하도록 Robust Parsing 기법 도입

```python
        try:
            json_block = re.search(r"\{.*\}", text, flags=re.S).group(0)
            data = json.loads(json_block)
        except Exception:
            data = {
                "standard": "100g",
                "kcal": 0,
                "carb": 0,
                "protein": 0,
                "fat": 0,
                "sugar": 0,
                "natrium": 0,
            }
```

### RAG 시나리오

### **시나리오 1: DB에 있는 음식**

```
사용자 업로드: "김밥.jpg"
  ↓
VLM 예측: "김밥"
  ↓
DB 검색: FOUND ("김밥", code="F12345")
  ↓
응답: DB의 정확한 영양 정보 (kcal=154, carb=32.5, ...)

```

### **시나리오 2: DB에 없는 유사 음식**

```
VLM: "엄마표계란말이"
  ↓
DB exact match 실패
  ↓
RAG query "엄마표계란말이"로 top_k=3 검색
  ↓
best_distance=0.48 (<=1.0) → 통과
  ↓
후보 중 dist<=0.75만 골라 가중평균
  ↓
결과 반환

```

### **시나리오 3: 완전히 새로운 음식**

```
사용자 업로드: "신메뉴_샐러드.jpg"
  ↓
VLM 예측: "샐러드"
  ↓
DB 검색: NOT FOUND
  ↓
RAG 검색: distance \u003e 1.0 (신뢰도 낮음)
  ↓
LLM 추정: Qwen VLM이 "샐러드" 영양 정보 생성
  ↓
응답: LLM이 생성한 영양 정보
```