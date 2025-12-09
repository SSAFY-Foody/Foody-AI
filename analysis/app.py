import os
import json
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

import requests
import pymysql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 환경 변수 / 설정

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
                "SELECT id, name, ai_learning_comment FROM characters"
            )
            rows = cursor.fetchall()
            return rows
    finally:
        conn.close()


# Pydantic 모델 정의 (AiReportRequest 대응)

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

    # 하루치 종합 정보
    dayTotalKcal: float
    dayTotalCarb: float
    dayTotalProtein: float
    dayTotalFat: float
    dayTotalSugar: float
    dayTotalNatrium: float

    # 끼니 정보
    meals: List[MealInfo]


# 프롬프트 빌더

def build_prompt(ai_request: AiReportRequestModel,
                 characters: List[Dict[str, Any]]) -> str:
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

    - 여러 캐릭터가 겹치는 경우, 우선순위 예시는 다음과 같다
      (실제 캐릭터 이름/설명에 맞추어 유사하게 적용해라):

      1) 짜구리(나트륨 과다), 달다구리(당류 과다), 주전부엉(간식 위주), 왕마니(과식)
      2) 슬리만더(저칼로리 + 고단백), 요마니(전반적으로 매우 적은 섭취)
      3) 탄단지오(영양 균형이 좋은 하루)
      4) 새싹 푸디(데이터 부족 / 특징 불명확)
      5) 잠마니는 '아침 결식' 등 별도 조건이 있을 때만 선택

    ------------------------------------------------------------
    [4] 점수(score)와 멘트(comment) 작성 기준

    - score:
      - 0 ~ 100 사이 값
      - 100에 가까울수록 "오늘 식단이 표준/균형에 잘 맞음"
      - 50 미만이면 개선이 많이 필요한 상태
    - comment:
      - 전부 한국어로 작성
      - 분량: 3~6문장 정도
      - 포함 내용:
        1) 오늘 식단 상태 요약 (예: "오늘은 나트륨과 당류 섭취가 다소 높았어요.")
        2) 잘한 점 (있다면 간단히 칭찬)
        3) 구체적인 개선 팁 2~3가지
           (예: "라면 국물은 절반만 먹어보기", "간식 대신 견과류나 과일 선택" 등)

    ------------------------------------------------------------
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


# =========================
# Gemini 호출
# =========================

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


# FastAPI 애플리케이션

app = FastAPI(title="Foody Character Recommender API")

CHARACTERS_CACHE: List[Dict[str, Any]] = []


@app.on_event("startup")
def on_startup():
    """
    서버 시작 시 캐릭터 목록 캐싱
    """
    global CHARACTERS_CACHE
    try:
        CHARACTERS_CACHE = load_characters_from_db()
        if not CHARACTERS_CACHE:
            print("[WARN] characters 테이블에서 가져온 데이터가 없습니다.")
        else:
            print(f"[INFO] {len(CHARACTERS_CACHE)}개의 캐릭터 정보를 로드했습니다.")
    except Exception as e:
        print(f"[ERROR] characters 로딩 실패: {e}")
        CHARACTERS_CACHE = []


# Spring AiReportService.analyzeMeal() 이 호출할 엔드포인트
@app.post("/api/analysis/report")
def analyze_meal(ai_request: AiReportRequestModel):
    # 캐릭터 목록 확보 (캐시 우선)
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

    # 프롬프트 생성
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

    # spring 에 맞춰 변환
    return {
        "characterId": character_id,
        "score": score,
        "comment": str(comment),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=True)
