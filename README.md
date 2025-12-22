# Foody (푸디) - AI 기반 식단 분석 및 커뮤니티 플랫폼

### 프로젝트 기간
2025.12.01 ~ 2025.12.25

### 팀원
- 박정훈
- 유주경

## 📖 프로젝트 소개
**Foody**는 사용자의 식단을 AI로 분석하여 영양 정보를 제공하고, 이를 기반으로 건강 관리를 돕는 **AI 식단 분석 서비스**입니다.

사용자는 총 세 가지의 방법(DB 음식, 직접 입력, 이미지 분석)을 통해 식단을 등록할 수 있습니다.
Vision-Language Model (VLM)을 활용하여 음식 이미지를 자동 인식하고, 칼로리 및 영양소를 분석합니다. 사용자는 분석된 레포트를 바탕으로 식단을 기록하고, 커뮤니티를 통해 다른 유저들과 식단 정보를 공유할 수 있습니다.

---

## 🏗️ 전체 아키텍처 (Architecture)

Foody는 크게 세 가지의 서버로 구성되어 있습니다.
각 서버의 상세 설정 및 구조는 각 레포지토리의 README를 참고하세요.

### 1. [🧠 Foody-AI (AI 분석 서버)](https://github.com/SSAFY-Foody/Foody-AI)
- **역할**: 음식 이미지 인식 및 식단 분석
- **핵심 기술**: Vision-Language Model (VLM), PyTorch

### 2. [🛡️ Foody-Backend (API 서버)](https://github.com/SSAFY-Foody/Foody-Backend)
- **역할**: 사용자 인증/인가, 데이터 관리(CRUD), 커뮤니티 기능, AI 서버 통신, 데이터베이스 통신
- **핵심 기술**: Spring Boot 3, MyBatis, MySQL, JWT, REST API, WebSocket, OAuth2.0

### 3. [🎨 Foody-Frontend (웹 클라이언트)](https://github.com/SSAFY-Foody/Foody-Frontend)
- **역할**: 사용자 인터페이스(UI), 식단 시각화, 커뮤니티 상호작용
- **핵심 기술**: Vue.js 3, Vite, TypeScript, TailwindCSS, Pinia

---

## ✨ 주요 기능
1.  **📸 AI 음식 이미지 분석**: 음식 이미지를 등록하고 분석 요청을 하면 AI가 음식 종류와 영양소를 자동으로 분석합니다.
2.  **📝 AI 식단 분석**: 식단을 등록하고 분석 요청을 하면 AI가 식단에 대한 분석 결과를 생성합니다.
3.  **📊 영양 리포트**: 섭취한 칼로리, 탄단지 비율을 시각적인 그래프로 제공합니다.
4.  **🤝 식단 커뮤니티**: 나의 식단 리포트를 공유하고, 다른 유저들과 소통(댓글)할 수 있습니다.
5.  **📚 푸디 도감**: 다양한 음식 캐릭터를 수집하고 도감을 채워나가는 재미 요소를 제공합니다.
6.  **🔐 회원 관리**: JWT 기반의 안전한 로그인 및 회원가입, 마이페이지 기능을 제공합니다.

---

## 시작하기 (Getting Started)

프로젝트를 실행하려면 각 모듈별 설정을 완료해야 합니다. 상세 내용은 아래 링크를 확인하세요.

- **Backend 설정 및 실행**: [Foody-Backend README](https://github.com/SSAFY-Foody/Foody-Backend)
- **Frontend 설정 및 실행**: [Foody-Frontend README](https://github.com/SSAFY-Foody/Foody-Frontend)
- **AI 서버 설정 및 실행**: [Foody-AI README](https://github.com/SSAFY-Foody/Foody-AI)

# Foody-AI

## 🧠 AI Architecture
**Foody-AI**는 음식 이미지를 분석하고 영양 정보를 추론하는 Vision-Language Model(VLM) 기반 서버입니다.

### 핵심 기술
- **Python**
- **PyTorch**
- **Vision-Language Model (VLM)**

---

## 🚀 실행 방법 (Run)

* required packages</br>transformers>=4.40.0
</br>peft>=0.10.0
accelerate>=0.27.0</br>
bitsandbytes>=0.43.0</br>
datasets>=2.18.0</br>
torch>=2.0.0</br>
pillow>=10.0.0 

```bash
# vlmmachine 서버 구동 예시 명령어
uvicorn app_v2:app --host 0.0.0.0 --port 8000

# vlm analysis 서버 구동 예시 명령어
uvicorn app_v2:app --host 0.0.0.0 --port 7000

# Qwen 모델 학습 예시 명령어
python continuedModel_train.py \
  --load_dir ./trained_models/qwen25_v8 \
  --output_dir ./trained_models/qwen25_v9 \
  --train_csv ./train_csvs/train_4.csv \
  --epochs 6

# Qwen 모델 테스트 명령어
python test.py --finetuned_path ../trained_models/qwen25_v5 --test_cxv ./test.csv --output_dir ../test_results/v5
```

---

## 📂 디렉토리 구조 (Directory Structure)
```
Foody-AI/
├── analysis/          # AI 분석 및 추론 로직 (Flask/FastAPI 등)
├── vlmmachine/        # VLM 추론 엔진 코어
└── vlmtraining/       # VLM 학습 데이터 전처리 및 학습 스크립트
```
---
# Foody AI Analysis

    
    


# 시스템 아키텍처

```
POST /api/analysis/report
    ↓
ai_request.userIsDiaBetes 확인
    ↓
    ┌─────────────────┴──────────────────┐
    ↓                                    ↓
userIsDiaBetes = True          userIsDiaBetes = False
    ↓                                    ↓
build_diabetes_context()           build_prompt()
    ↓
쿼리 생성:
"당뇨병 환자의 하루 식단 요약.
총 칼로리 2000 kcal, 탄수화물 250g, ..."
    ↓
DIABETES_RETRIEVER.invoke(query_text)
    ↓
ChromaDB 벡터 검색
    ↓
상위 4개 Document 반환
    ↓
page_content 추출 및 결합
    ↓
diabetes_context 생성
    ↓
build_prompt_diabetes(ai_request, characters, diabetes_context)
    ↓
    └─────────────────┬──────────────────┘
                      ↓
              call_gemini(prompt)
                      ↓
              Gemini API 호출
                      ↓
              JSON 응답 파싱
                      ↓
              AiReportResponse 반환
```

---

# 전략

BaseModel : GMS ( Gemini 2.5 Pro)

### 1. RAG (Retrieval-Augmented Generation) for Diabetes

- 당뇨병 환자에게 검증된 의학 지침 조언 제공
- 근거있는 답변으로 환각 방지

[자료실, 당뇨병 진료지침, KDA, 대한당뇨병학회 Korean Diabetes Association](https://www.diabetes.or.kr/bbs/?code=guide&mode=view&number=2099&page=1&code=guide)

- 당뇨병 관련 지침 → Chunking → Embedding → 문서 검색

### 임베딩

- **임베딩 모델 : sentence-transformers/all-MiniLM-L6-v2**

[sentence-transformers/all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

경량화된 다국어 임베딩 모델

한국어도 지원하여 속도와 성능의 균형이 좋음

```python
# PDF 로드
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 🔥 여기서 청킹(chunking) 수행
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 각 청크는 최대 1000자
    chunk_overlap=150     # 인접 청크 간 150자 겹침
)
split_docs = splitter.split_documents(docs)  # 여러 개의 작은 문서로 분할

# 분할된 문서들을 임베딩
vectordb = Chroma.from_documents(
    split_docs,  # ← 분할된 여러 문서
    embedding=embeddings,
    ...
)
```

- 청킹
- 1000자씩 잘라서 여러개의 문서로 분할
- 앞뒤 청크가 150자씩 겹치게 하여 문맥 유지
- 각 청크 변환 후 ChromaDB 에 저장

```python
# k=4: 상위 4개 문서를 검색
DIABETES_RETRIEVER = vectordb.as_retriever(search_kwargs={"k": 4})
```

- 사용자 쿼리와 유사한 상위 4개의 청크에서 검색

### 쿼리 생성

```python
def build_diabetes_context(ai_request):
    query_text = (
        f"당뇨병 환자의 하루 식단 요약. "
        f"총 칼로리 {ai_request.dayTotalKcal} kcal, "
        f"탄수화물 {ai_request.dayTotalCarb} g, "
        f"단백질 {ai_request.dayTotalProtein} g, "
        f"지방 {ai_request.dayTotalFat} g, "
        f"당류 {ai_request.dayTotalSugar} g, "
        f"나트륨 {ai_request.dayTotalNatrium} mg. "
        "당뇨병 환자의 식사요법, 혈당 관리, 탄수화물 조절, "
        "나트륨 제한에 대한 진료지침."
    )
```

- 사용자의 실제 섭취량을 쿼리에 포함시켜 맥락적으로 관련된 지침 (당뇨병) 검색

### 이원화 프롬프트 엔지니어링

**1) 역할 정의**

```
너는 식단 관리 서비스 '푸디(Foody)'의 캐릭터 추천 AI야.
```

**2) 입력 데이터 설명**

- 5가지 정보 블록으로 구성:
    1. 전체 요청 JSON
    2. stdInfo (권장 섭취량)
    3. dayTotal (실제 섭취량)
    4. meals (끼니별 상세)
    5. 캐릭터 목록 (DB에서 로드)

**3) 임무 명시**

```
1) 오늘 하루 섭취 성향을 가장 잘 표현하는 캐릭터 1명 선택
2) 평가 점수 (0~100) 산출
3) 한국어 맞춤 추천 멘트 작성

```

**4) 출력 형식 엄격 제한**

```json
{
  "characterId": <정수>,
  "score": <실수 또는 정수>,
  "comment": "<한국어 멘트>"
}

그 외 어떤 텍스트, 설명, 주석, 마크다운도 출력 금지

```

### **당뇨 사용자 프롬프트 추가 요소**

> 당뇨병환자의 식단은 GI 지수에 영향을 끼치므로 당뇨병 관련지침 기반 점수 산정 기준 진행
> 

**1) 전문가 페르소나**

```
너는 당뇨병 환자 식사요법에 익숙한 전문가야.

```

**2) RAG 컨텍스트 주입**

```
[3] 당뇨병 진료지침 RAG 결과

아래 내용은 당뇨병 진료지침/전문 문헌에서 검색한 관련 문단이다.
이 내용을 근거로 혈당 관리, 식사요법, 탄수화물/당류/나트륨 조절에 대해
사용자에게 전문성을 느낄 수 있는 조언을 해라.

{diabetes_block}

이 진료지침과 모순되는 조언을 하지 말고,
가능하면 진료지침의 요지를 사용자가 이해하기 쉬운 말로 풀어 써라.

```

**3) 안전 제약**

```
약물 용량 조절이나 인슐린 조절과 같은 구체적인 의료 행위는 절대로 지시하지 말고,
필요한 경우 "주치의와 상의해야 한다"는 표현으로만 안내한다.

```

**4) 엄격한 평가**

```
당뇨병 환자라는 점을 고려하여, 고당분/고GI/고탄수화물/고나트륨 섭취에 대해
일반 사용자보다 더 큰 감점을 줄 수 있다.
```

### 점수 산정 알고리즘

LLM에게 "점수 계산 공식"을 프롬프트에 직접 명시하여, 주관적 판단이 아닌 **객관적 수치 계산** 유도

### **점수 계산 프로세스**

**1) 영양소별 비율 계산**

```
R = (하루 총 섭취량) / (하루 권장량)

예시:
- R_칼로리 = dayTotalKcal / stdKcal
- R_탄수화물 = dayTotalCarb / stdCarb
- R_단백질 = dayTotalProtein / stdProtein
- R_지방 = dayTotalFat / stdFat
- R_당류 = dayTotalSugar / stdSugar

```

**2) 영양소별 점수 함수**

**일반 영양소 (칼로리/탄수화물/단백질/지방)**:

```
점수(R) =
  100                         if 0.9 ≤ R ≤ 1.1   (최적 범위)
  90 + (R - 0.8) × 100        if 0.8 ≤ R < 0.9   (약간 부족)
  90 + (1.2 - R) × 100        if 1.1 < R ≤ 1.2   (약간 과다)
  80 + (R - 0.7) × 100        if 0.7 ≤ R < 0.8   (많이 부족)
  80 + (1.3 - R) × 100        if 1.2 < R ≤ 1.3   (많이 과다)
  70 + (R - 0.6) × 100        if 0.6 ≤ R < 0.7
  70 + (1.4 - R) × 100        if 1.3 < R ≤ 1.4
  max(0, 70 - (R - 1.4) × 100) if R > 1.4        (극도 과다)
  max(0, 70 - (0.6 - R) × 100) if R < 0.6        (극도 부족)

```

**당류 (낮을수록 좋음)**:

```
점수_당류(R) =
  100                          if R ≤ 1.0        (권장량 이하)
  90 + (1.2 - R) × 50          if 1.0 < R ≤ 1.2
  80 + (1.4 - R) × 50          if 1.2 < R ≤ 1.4
  max(0, 80 - (R - 1.4) × 100) if R > 1.4        (과도)

```

**3) 가중치 적용**

```
최종 점수 = (kcalScore × 0.30) + (carbScore × 0.20)
          + (proteinScore × 0.25) + (fatScore × 0.15)
          + (sugarScore × 0.10)

※ 나트륨은 점수 계산에서 제외 (멘트에만 반영)

```

**4) 절대 규칙 (Hard Constraints)**

```
1) 결식 1회 → 최종 점수 ≤ 60점 (CAP)
2) 결식 2회 이상 → 최종 점수 ≤ 40점 (CAP)
3) 정보 부족 시 보수적 평가 (낮게)
4) 직감/추측으로 점수 조정 금지

```

### 푸디 캐릭터 프롬프트

```
1순위: 문제 캐릭터 (급하게 개선 필요)
  - 짜구리 (나트륨 과다)
  - 달다구리 (당류 과다)
  - 주전부엉 (간식 위주)
  - 왕마니 (과식)
  - 잠마니 (아침 결식)

2순위: 특수 식단 캐릭터
  - 슬리만더 (저칼로리 + 고단백)
  - 요마니 (매우 적은 섭취)

3순위: 이상적 캐릭터
  - 탄단지오 (영양 균형 우수)

```

**선정 로직**:

```
if dayTotalNatrium / stdNatrium > 1.5:
    → 짜구리 (1순위)
elif dayTotalSugar / stdSugar > 1.3:
    → 달다구리 (1순위)
elif 아침 총칼로리 == 0 or 아침 결식:
    → 잠마니 (1순위)
elif dayTotalKcal / stdKcal < 0.7 and dayTotalProtein / stdProtein > 1.0:
    → 슬리만더 (2순위)
elif 모든 영양소 0.9~1.1 범위:
    → 탄단지오 (3순위)
else:
    LLM이 description 참고하여 최종 결정
```
---
# VLM 로직


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
---
# VLM Training



BaseModel : Qwen 2.5 VL 3B (Hugging Face)

Data set : AI - Hub한국 이미지 (음식), 150 종 (약 15만개의 이미지 데이터 사용)

---

# 모델 선정 이유

> 음식 인식은 단순 탐지가 아니라 사용자 입력 사진의 맥락 (혼합 메뉴 / 가정식 / 브랜드 조립법/ 부분 가림) 을이해해 , RAG 검색에 바로 쓰는 정규화된 음식명을 안정적으로 뽑는것이 핵심이라 VLM 을 사용
> 

### 탐지는 박스를 잘 찾는 문제고 푸디의 기능은 대표 음식 정규화 → 영양소 검색을 하나의 모델로 빠르게 도입하는 것이 목적

- 한국인의 다빈도 섭치 외식메뉴 400종이 학습된 Yolo v3 모델이 존재

[AI-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=74)

- 하지만 학습 데이터의 클래스 범위 밖에는 영양성분, 음식 이름 추론이 상대적으로 약하다.
- 그에 비해 VLM 은 이미지 + 텍스트 기반 지식이기 때문에 클래스 외에도 일반화가 상대적으로 쉽고, 
LoRA 학습을 통해 도메인 확정을 빠르게 할 수 있다.
→ ***운영 중 지속 개선 가능한 구조가 가능***

---

# Base Model의 문제점

### 1. LLM 모델 기반으로 단순히 **음식명만을 추론하기 어렵다**

![image.png](./images/image.png)

→ 음식 명 추론 프롬프트

- 해당 프롬프트를 사용하더라도 한국어로 답변하지 않고 문자가 깨지거나 외국어가 섞어 나오는 문제점이 존재

![image.png](./images/image%201.png)

![image.png](./images/image%202.png)

***150종 테스트 결과 정확도 4% 를 보여주고 있다***

### 2. 한국 음식에 대해서는 낮은 정확도 (학습 데이터 분포 문제)

- 떡볶이 → 파스타
- 계란말이 → 오믈렛

![image.png](./images/image%203.png)

![image.png](./images/image%204.png)

- Qwen 계열 VLM 은 글로벌 이미지, 텍스트 데이터 비중이 높다
- 한국음식 같은 경우 외형이 유사한 메뉴가 많고 (ex. 국, 찌개, 볶음),
명칭이 조리법, 재료 지역에 따라 세분화 되어간다.

***한국 음식데이터에 대한 데이터 분포를 확장하는 것이 필요***

---

# 학습 전략

### 1. LoRA (Low-Rank Adaption) 및 QLoRA 기반 Fine-Tuning

- 전체 모델을 학습하는 대신, LoRA 어댑터만 학습하여 메모리 사용량 절감
- 전체 모델 대비 학습 가능한 파라미터를 2%미만으로 축소 하면서도, 타겟(한국 음식)에 대한 높은 성능을 유지
- Double Quantization : 양자화 상수 자체에서도 양자화하여 추가 메모리 절감 유도

| **파라미터** | **기본값** | 설명 |
| --- | --- | --- |
| `lora_r` | `16` | LoRA Rank |
| `lora_alpha` | `32` | LoRA Scaling Factor |
| `lora_dropout` | `0.05` | LoRA Dropout Rate |
| `lora_target_modules` | 7개 모듈 | `q_proj`,  `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, 
`down_proj` |

### 2. Image Token Explosion 방지

- Vision-Language 모델 특성상 이미지가 너무 많은 토큰으로 변환되어 메모리가 폭발하는 문제 존재
1. 이미지 리사이징 : 224 * 224 로 리사이즈 작업
2. 이미지 + 텍스트 합산 1536 (Max Length 제한)
3. Tail-Preserving Truncation : 길이 초과 시 앞부분을 잘라내고 뒷부분을 보존
(정답 토큰은 뒤에 있기 때문에 정답손실을 최소화 하도록 유도)

Loss Masking 을 이용하여 프롬프트는 음수 가중치(-100)을 부여하여 학습을 진행시키지 않고,
 정답만 학습하여 Pytorch 가 정답만 집중하도록 유도

![image.png](./images/image%205.png)

### 3. Chat Template 기반 학습 데이터 구성

- Qwen 모델의 ‘apply_chat_temolate’ 메서드 활용하여 모델의 대화 형식에 맞게 데이터 구성
- User : 이미지 + 프롬프트, Assistant : 정답(음식명)

![image.png](./images/image%206.png)

- 학습 프롬프트

![image.png](./images/image%207.png)

### 4. 답변(정답) Label 전처리

- 학습 csv 에서 음식명이 모델의 출력할 형식과 일치하도록 전처리
→ 학습 데이터와 추론 결과의 일관성을 보장
- 공백, 특수문자 제거, 최대 토큰 제한

![image.png](./images/image%208.png)

### 5. 단계 학습

- 메모리 제약 조건을 해결하기 위해 전체 데이터셋을 나누어 점진적 학습 진행
- 초기, 중기, 후기 로 나누어 학습률을 다르게 하여 모델이 수렴하도록 유도

1. **초기 (Learning Rate : 8e-5)
- 사전 학습된 Qwen 모델을 한국 음식 도메인에 빠르게 적응시키기 위한 목적
- Loss를 빠르게 떨어뜨리기 위함**

1. **중기 (Learning Rate : 5e-5)**
- 이미 학습된 지식을 유지하면서 정교한 패턴 학습
- 초기에 비해 급격한 가중치 변화 방지 및 과적합 없이 성능 향상 목적

2. **후기 (Learning Rate : 2e-5)
- 최적점 근처에서 세밀하게 조정 (Fine-Grained)
- 새로운 데이터 추가 시에도 기존 지식 보존**

### 6. Data (Ai-Hub K-Food 공개 데이터 셋)

[AI-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=79)

클래스 : 150

총 학슴 샘플 수 : 약 4,000 개

학습 환경에 따른 메모리 폭발 문제를 방지하기 한 클래스당 평균 20개 균형 분포 형식으로 버전을 나누어 점진적 학습 진행

### 기타

- Gradient Accumulation : 8 steps
- Gradient Ceckpointing : 중간 활성화 값을 메모리에 제거하고 역전파 시 재계산
- adamw_8big : Optimizer 상태를 8bit 로 저장하여 메모리 축소
- FP16 Mixed Precisiton

---

성과

### 기존 모델 대비 약 65% 향상 ( 4 % → 69%)

- 한식 모델 클래스 150개 기반 테스트 진행

### 테스트 결과

![image.png](./images/image%209.png)

- 정답 개선

![image.png](./images/image%2010.png)

- 오답 값이지만, 외국어 혼용, 글씨가 깨지는 문제 해결

### 실제 서비스 결과

![image.png](./images/image%2011.png)

![image.png](./images/image%2012.png)
