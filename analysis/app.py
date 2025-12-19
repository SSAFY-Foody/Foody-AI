import os
import json
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

import requests
import pymysql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


DIABETES_DB_DIR = "./chroma_diabetes_guideline"
DIABETES_COLLECTION_NAME = "diabetes_guideline"
DIABETES_COLLECTION = None  # startup 때 로드
DIABETES_RETRIEVER = None  # startup 때 로드


load_dotenv()  # .env 파일 내용 읽어오기

GMS_KEY = os.getenv("GMS_KEY")
GEMINI_URL = os.getenv("AI_URL")

# DB 설정
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


# character 테이블에서 푸디 캐릭터 목록 가져오기
def load_characters_from_db() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, name, ai_learning_comment FROM characters WHERE id >=3  ORDER BY id ASC"
            )
            rows = cursor.fetchall()
            return rows
    finally:
        conn.close()


# Pydantic 모델 정의 (Spring AiReportRequest/AiReportResponse)

class StdInfo(BaseModel):
    stdWeight: float
    stdKcal: float
    stdCarb: float
    stdProtein: float
    stdFat: float
    stdSugar: float
    stdNatrium: float


class FoodInfo(BaseModel):
    name: str
    eatenWeight: float
    eatenKcal: float
    eatenCarb: float
    eatenProtein: float
    eatenFat: float
    eatenSugar: float
    eatenNatrium: float


class MealInfo(BaseModel):
    mealType: str  # BREAKFAST, LUNCH, DINNER, SNACK
    totalKcal: float
    totalCarb: float
    totalProtein: float
    totalFat: float
    totalSugar: float
    totalNatrium: float
    foods: List[FoodInfo]


class AiReportRequestModel(BaseModel):
    # 유저 정보
    stdInfo: StdInfo
    userActivityLevelDesc: str
    userIsDiaBetes: bool

    # 하루치 종합 정보
    dayTotalKcal: float
    dayTotalCarb: float
    dayTotalProtein: float
    dayTotalFat: float
    dayTotalSugar: float
    dayTotalNatrium: float

    # 끼니 정보
    meals: List[MealInfo]


class AiReportResponse(BaseModel):
    score: Optional[float]  # null 허용
    comment: str
    characterId: int


# 공통 프롬프트 빌더 (비당뇨용)

def build_prompt(ai_request: AiReportRequestModel,
                 characters: List[Dict[str, Any]]) -> str:
    # 요청 JSON 그대로도 모델에 넘겨줌
    request_text = json.dumps(
        ai_request.model_dump(),
        ensure_ascii=False,
        indent=2
    )

    # 캐릭터 목록을 텍스트로 정리
    characters_text_lines = []
    for ch in characters:
        desc = ch.get("ai_learning_comment", "")
        characters_text_lines.append(
            f"- id: {ch['id']}\n"
            f"  name: {ch['name']}\n"
            f"  description: {desc}"
        )
    characters_block = "\n\n".join(characters_text_lines)

    prompt = f"""
너는 식단 관리 서비스 '푸디(Foody)'의 캐릭터 추천 AI야.

너에게는 다음 정보가 주어진다:
- [1] 오늘자 AI 분석 요청 전체 JSON (AiReportRequest)
- [2] DB에서 가져온 푸디 캐릭터 목록 (id, name, ai_learning_comment)

너의 임무:
1) 오늘 하루의 섭취 성향을 가장 잘 표현하는 캐릭터 1명을 선택한다.
2) 오늘 식단에 대한 평가 점수(score)를 0~100 사이 실수로 준다.
3) 한국어로 맞춤 추천 멘트(comment)를 작성한다.
   - 오늘 섭취 성향 요약
   - 좋았던 점 / 아쉬운 점
   - 내일부터 실천할 수 있는 개선 팁 2~3가지

최종 출력은 반드시 아래 JSON 형식 ONLY로 출력해야 한다:

{{
  "characterId": <정수>,
  "score": <실수 또는 정수>,
  "comment": "<한국어 멘트>"
}}

그 외 어떤 문장도 출력하지 마라.
특히 설명, 해설, 마크다운, 자연어 문장은 금지다.
오직 위 JSON 한 덩어리만 출력해라.

------------------------------------------------------------
[1] 오늘자 AI 분석 요청 정보 (AiReportRequest JSON)

아래 JSON은 Spring 서버에서 하루치 식단을 정리해 보낸 데이터이다.

{request_text}

설명:
- stdInfo: 이 사용자의 표준 체중/칼로리/탄단지/당/나트륨 (1일 기준 권장량)
  - stdKcal, stdCarb, stdProtein, stdFat, stdSugar, stdNatrium
- userActivityLevelDesc: 사용자의 활동량 설명 (예: '가벼운 활동', '매우 활동적' 등)
- dayTotalKcal, dayTotalCarb, ...: 오늘 하루 실제 섭취 총합
- meals: 끼니 단위 정보
  - 각 끼니별 총합 (totalKcal 등)
  - foods: 해당 끼니에서 먹은 개별 음식들
    - eatenWeight, eatenKcal, eatenCarb, eatenProtein, eatenFat, eatenSugar, eatenNatrium

즉, 너는
  "오늘 하루 실제 섭취량 vs stdInfo에 있는 권장 섭취량"
을 비교해서 오늘의 식습관을 해석해야 한다.

------------------------------------------------------------
[2] 1일 기준 Foody 캐릭터 설명 (DB에서 불러온 단기판)

아래는 데이터베이스에서 불러온 푸디 캐릭터 목록이다.
각 캐릭터는 다음 정보를 가진다:

- id: 캐릭터를 나타내는 정수형 ID
- name: 캐릭터 이름
- description: 이 캐릭터가 나타내는 "오늘 하루" 식습관/영양 상태에 대한 상세 설명
  (DB의 ai_learning_comment 필드 내용)

캐릭터 목록:

{characters_block}

description에는 예를 들어
- "오늘 칼로리 낮고 단백질을 잘 챙긴 하루"
- "오늘 나트륨이 과다한 하루"
- "오늘 단 음식/간식 위주로 먹은 하루"
- "오늘 영양 균형이 잘 맞는 하루"
와 같은 식으로 **1일 기준** 해석이 적혀 있다.

너는 이 description을 그대로 참고해서,
오늘자 섭취 데이터와 가장 잘 어울리는 캐릭터 1명을 선택해라.

------------------------------------------------------------
[3] 캐릭터 선택 규칙 (1일 기준 단기 버전)

- 오늘 하루 기록만 보고 즉시 판단한다.
- 가장 두드러진 문제/특징(칼로리 과/저체, 단백질 부족, 당류 과다, 나트륨 과다, 간식 과다 등)을 우선 반영한다.
- 캐릭터 description에 이미 어떤 상황에서 쓰이는 캐릭터인지 상세히 적혀 있으므로,
  각 캐릭터의 설명과 오늘자 수치를 비교해서 가장 잘 맞는 캐릭터 1명을 고른다.
- 2번 새싹푸디는 초기값 이므로 선정하면 안 된다.
- 여러 캐릭터가 겹치는 경우, 우선순위 예시는 다음과 같다
  (실제 캐릭터 이름/설명에 맞추어 유사하게 적용해라):

  1) 짜구리(나트륨 과다), 달다구리(당류 과다), 주전부엉(간식 위주), 왕마니(과식)
  2) 슬리만더(저칼로리 + 고단백), 요마니(전반적으로 매우 적은 섭취)
  3) 탄단지오(영양 균형이 좋은 하루)
  4) 잠마니는 '아침 결식' 등 별도 조건이 있을 때만 선택

------------------------------------------------------------
[4] 점수(score) 산정 기준 (반드시 아래 수식 그대로 계산)

점수는 하루 전체 섭취량을 하루 권장량과 비교하여 계산한다.

1) 영양소별 비율 R 계산 (하루 전체 기준)
- R = (하루 총 섭취량) / (하루 권장량)

예시:
- R_칼로리 = dayTotalKcal / stdKcal
- R_탄수화물 = dayTotalCarb / stdCarb
- R_단백질 = dayTotalProtein / stdProtein
- R_지방 = dayTotalFat / stdFat
- R_당류 = dayTotalSugar / stdSugar

2) 영양소별 점수 함수 (나트륨은 점수에서 제외!)
- 점수(R): (칼로리/탄수/단백질/지방에 적용)

점수(R) =
  100                         if 0.9 ≤ R ≤ 1.1
  90 + (R - 0.8) × 100         if 0.8 ≤ R < 0.9
  90 + (1.2 - R) × 100         if 1.1 < R ≤ 1.2
  80 + (R - 0.7) × 100         if 0.7 ≤ R < 0.8
  80 + (1.3 - R) × 100         if 1.2 < R ≤ 1.3
  70 + (R - 0.6) × 100         if 0.6 ≤ R < 0.7
  70 + (1.4 - R) × 100         if 1.3 < R ≤ 1.4
  max(0, 70 - (R - 1.4) × 100) if R > 1.4
  max(0, 70 - (0.6 - R) × 100) if R < 0.6

- 점수_당류(R): (당류는 "낮을수록 높은 점수" 규칙 적용)

점수_당류(R) =
  100                          if R ≤ 1.0
  90 + (1.2 - R) × 50          if 1.0 < R ≤ 1.2
  80 + (1.4 - R) × 50          if 1.2 < R ≤ 1.4
  max(0, 80 - (R - 1.4) × 100) if R > 1.4

3) 영양소별 가중치 (나트륨 제외)
- 칼로리 0.30
- 탄수화물 0.20
- 단백질 0.25
- 지방 0.15
- 당류 0.10
- 나트륨은 점수 계산에서 제외한다. (감점/가중치 0)

4) 하루 최종 점수(score) 계산
- score = (kcalScore×0.30) + (carbScore×0.20) + (proteinScore×0.25)
        + (fatScore×0.15) + (sugarScore×0.10)
- 각 영양소 점수는 0~100으로 clamp 한다.
- 소수점은 1자리 반올림 (예: 83.4)

5) 퍼센티지 표시에 대한 내부 계산(멘트에 참고 가능)
- 퍼센티지 = (하루 총 섭취량 / 하루 권장량) × 100
- 이 퍼센티지는 멘트에서 "권장량 대비 몇 %" 같은 표현을 만들 때 참고해도 된다.

※ 주의:
- 나트륨은 점수에서 제외하지만, 멘트/캐릭터 선택에는 참고할 수 있다.
- score는 반드시 위 계산으로 산출한 값이어야 한다(감으로 주지 마라).
- 하루 전체 섭취량(dayTotalXXX)을 사용하여 계산하며, 끼니별 평균을 내지 않는다.

[5] 출력 형식

다시 강조한다. 출력은 반드시 아래 JSON 형식 ONLY:

{{
  "characterId": <정수>,
  "score": <실수 또는 정수>,
  "comment": "<한국어 멘트>"
}}

그 외 어떤 텍스트, 설명, 주석, 마크다운, 백틱, 자연어 문장도 출력하지 마라.
"""
    return prompt.strip()


# 당뇨용: 당뇨병 진료지침 RAG 컨텍스트 빌더

def build_diabetes_context(ai_request: AiReportRequestModel) -> str:
    """
    당뇨병 환자의 하루 섭취 요약으로 Chroma에 질의해서
    지침 관련 문단 몇 개를 가져온다.
    """
    global DIABETES_COLLECTION
    if DIABETES_COLLECTION is None:
        return ""

    query_text = (
        f"당뇨병 환자의 하루 식단 요약. "
        f"총 칼로리 {ai_request.dayTotalKcal} kcal, "
        f"탄수화물 {ai_request.dayTotalCarb} g, "
        f"단백질 {ai_request.dayTotalProtein} g, "
        f"지방 {ai_request.dayTotalFat} g, "
        f"당류 {ai_request.dayTotalSugar} g, "
        f"나트륨 {ai_request.dayTotalNatrium} mg. "
        "당뇨병 환자의 식사요법, 혈당 관리, 탄수화물 조절, 나트륨 제한에 대한 진료지침."
    )

    try:
        # 랭쳋인 방식 : Document 형태로 반환됨
        docs = DIABETES_RETRIEVER.get_relevant_documents(query_text)
        if not docs:
            return ""
        # 각 document의 page_content만 꺼내서 하나의 문맥 블록으로 합치는 작업
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        print(f"[WARN] 당뇨 지침 Chroma 조회 실패: {e}")
        return ""


def build_prompt_diabetes(ai_request: AiReportRequestModel,
                          characters: List[Dict[str, Any]],
                          diabetes_context: str) -> str:
    """
    당뇨 환자 전용 프롬프트.
    - 기본 캐릭터/점수 로직은 그대로
    - 추가로 RAG에서 가져온 '당뇨병 진료지침 요약'을 참고하게 함
    """
    request_text = json.dumps(
        ai_request.model_dump(),
        ensure_ascii=False,
        indent=2
    )

    characters_text_lines = []
    for ch in characters:
        desc = ch.get("ai_learning_comment", "")
        characters_text_lines.append(
            f"- id: {ch['id']}\n"
            f"  name: {ch['name']}\n"
            f"  description: {desc}"
        )
    characters_block = "\n\n".join(characters_text_lines)

    diabetes_block = diabetes_context if diabetes_context else "※ 검색된 진료지침 문단이 충분하지 않습니다. 일반적인 당뇨병 식사요법 원칙을 바탕으로 답변하세요."

    prompt = f"""
너는 식단 관리 서비스 '푸디(Foody)'의 캐릭터 추천 AI이자,
당뇨병 환자 식사요법에 익숙한 전문가야.

지금 사용자는 **당뇨병을 가지고 있다(userIsDiaBetes = true)**.
당뇨병 환자의 혈당 관리, 저혈당/고혈당 위험, 체중 관리, 합병증 예방 관점에서
조금 더 엄격하게 식단을 평가해야 한다.

너에게는 다음 정보가 주어진다:
- [1] 오늘자 AI 분석 요청 전체 JSON (AiReportRequest)
- [2] DB에서 가져온 푸디 캐릭터 목록 (id, name, ai_learning_comment)
- [3] 당뇨병 진료지침에서 검색한 관련 문단 요약 (RAG 결과)

너의 임무:
1) 오늘 하루의 섭취 성향을 가장 잘 표현하는 캐릭터 1명을 선택한다.
2) 오늘 식단에 대한 평가 점수(score)를 0~100 사이 실수로 준다.
   - 당뇨병 환자라는 점을 고려하여, 고당분/고GI/고탄수화물/고나트륨 섭취에 대해
     일반 사용자보다 더 큰 감점을 줄 수 있다.
3) 한국어로 맞춤 추천 멘트(comment)를 작성한다.
   - 오늘 섭취 성향 요약
   - 혈당/당뇨 관리 관점에서 좋았던 점 / 아쉬운 점
   - 내일부터 실천할 수 있는 개선 팁 2~3가지
4) 약물 용량 조절이나 인슐린 조절과 같은 구체적인 의료 행위는 절대로 지시하지 말고,
   필요한 경우 "주치의와 상의해야 한다"는 표현으로만 안내한다.

최종 출력은 반드시 아래 JSON 형식 ONLY로 출력해야 한다:

{{
  "characterId": <정수>,
  "score": <실수 또는 정수>,
  "comment": "<한국어 멘트>"
}}

그 외 어떤 문장도 출력하지 마라.
특히 설명, 해설, 마크다운, 자연어 문장은 금지다.
오직 위 JSON 한 덩어리만 출력해라.

------------------------------------------------------------
[1] 오늘자 AI 분석 요청 정보 (AiReportRequest JSON)

아래 JSON은 Spring 서버에서 하루치 식단을 정리해 보낸 데이터이다.

{request_text}

(설명은 공통 버전과 동일하다.)

------------------------------------------------------------
[2] 1일 기준 Foody 캐릭터 설명 (DB에서 불러온 단기판)

캐릭터 목록:

{characters_block}

캐릭터 선택 규칙은 비당뇨 사용자와 동일하나,
- 당류/탄수화물/나트륨 과다일 때는
  그에 해당하는 캐릭터(예: 짜구리, 달다구리, 주전부엉, 왕마니 등)를
  조금 더 적극적으로 선택해라.

------------------------------------------------------------
[3] 당뇨병 진료지침 RAG 결과

아래 내용은 당뇨병 진료지침/전문 문헌에서 검색한 관련 문단이다.
이 내용을 근거로 혈당 관리, 식사요법, 탄수화물/당류/나트륨 조절에 대해
사용자에게 전문성을 느낄 수 있는 조언을 해라.

{diabetes_block}

이 진료지침과 모순되는 조언을 하지 말고,
가능하면 진료지침의 요지를 사용자가 이해하기 쉬운 말로 풀어 써라.
"""
    return prompt.strip()


# Gemini 호출

def call_gemini(prompt: str) -> Dict[str, Any]:
    if not GMS_KEY:
        raise RuntimeError("GMS_KEY 환경 변수가 설정되지 않았습니다.")

    params = {"key": GMS_KEY}
    headers = {"Content-Type": "application/json"}

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(
        GEMINI_URL,
        params=params,
        headers=headers,
        data=json.dumps(body, ensure_ascii=False),
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API 호출 실패: {response.status_code} {response.text}"
        )

    data = response.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"예상하지 못한 Gemini 응답 형식: {data}")

    # 디버깅용: 원본 출력
    print("=== RAW GEMINI OUTPUT ===")
    print(text)
    print("=== END RAW GEMINI OUTPUT ===")

    cleaned = text.strip()

    # ```json ... ``` 형태로 오는 경우 처리
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    # JSON 시작/끝 위치 찾기
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"JSON 본문을 찾지 못했습니다. cleaned={cleaned!r}")

    json_str = cleaned[start:end + 1]

    print("=== PARSED JSON STRING ===")
    print(json_str)
    print("=== END PARSED JSON STRING ===")

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패: {e}; json_str={json_str!r}")

    return result


# FastAPI

app = FastAPI(title="Foody Character Recommender API")

CHARACTERS_CACHE: List[Dict[str, Any]] = []


@app.on_event("startup")
def on_startup():
    """
    서버 시작 시 캐릭터 목록 + 당뇨 RAG 컬렉션 로딩
    """
    global CHARACTERS_CACHE, DIABETES_COLLECTION, DIABETES_RETRIEVER

    # 캐릭터 캐시
    try:
        CHARACTERS_CACHE = load_characters_from_db()
        if not CHARACTERS_CACHE:
            print("[WARN] characters 테이블에서 가져온 데이터가 없습니다.")
        else:
            print(f"[INFO] {len(CHARACTERS_CACHE)}개의 캐릭터 정보를 로드했습니다.")
    except Exception as e:
        print(f"[ERROR] characters 로딩 실패: {e}")
        CHARACTERS_CACHE = []

    # 당뇨 지침 Chroma (LangChain)
    try:
        # 인덱싱할 때 썼던 임베딩 모델 이름이랑 맞춰야 함!
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectordb = Chroma(
            persist_directory=DIABETES_DB_DIR,
            collection_name=DIABETES_COLLECTION_NAME,
            embedding_function=embeddings,
        )

        DIABETES_RETRIEVER = vectordb.as_retriever(search_kwargs={"k": 4})
        print("[INFO] Diabetes guideline retriever 로드 완료.")
    except Exception as e:
        print(f"[WARN] Diabetes guideline Chroma 로드 실패: {e}")
        DIABETES_RETRIEVER = None



# Spring AiReportService.analyzeMeal() 이 호출할 엔드포인트
@app.post("/api/analysis/report", response_model=AiReportResponse)
def analyze_meal(ai_request: AiReportRequestModel):
    # 요청 바디에 들어온 실제 값 확인
    print(f"[INFO] userIsDiaBetes = {ai_request.userIsDiaBetes}")

    if not CHARACTERS_CACHE:
        try:
            characters = load_characters_from_db()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"characters 정보를 불러오지 못했습니다: {e}",
            )
    else:
        characters = CHARACTERS_CACHE
        print("[INFO] 캐릭터 정보를 캐시에서 사용합니다.")

    # 당뇨 여부에 따라 프롬프트 분기
    if ai_request.userIsDiaBetes:
        print("[INFO] 당뇨병 환자용 AI 분석 요청입니다. (RAG 사용)")
        diabetes_context = build_diabetes_context(ai_request)
        prompt = build_prompt_diabetes(ai_request, characters, diabetes_context)
    else:
        print("[INFO] 일반 사용자용 AI 분석 요청입니다. (기존 프롬프트)")
        prompt = build_prompt(ai_request, characters)

    # Gemini 호출
    try:
        gemini_result = call_gemini(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 키 이름 통일 (characterId / character_id 둘 다 허용)
    character_id = gemini_result.get("characterId", gemini_result.get("character_id"))
    score = gemini_result.get("score")
    comment = gemini_result.get("comment")

    if character_id is None or comment is None:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini 응답에 필요한 필드가 없습니다: {gemini_result}",
        )

    try:
        character_id = int(character_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=500,
            detail=f"characterId가 정수가 아닙니다: {character_id!r}",
        )

    # score는 없으면 None 허용, 숫자로 캐스팅 시도
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None

    # FastAPI → Spring JSON
    return AiReportResponse(
        characterId=character_id,
        score=score,
        comment=str(comment),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=True)
