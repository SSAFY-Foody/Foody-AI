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