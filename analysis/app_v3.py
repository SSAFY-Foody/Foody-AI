import os
import json
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from datetime import datetime

import requests
import pymysql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain 최신 버전 호환 import
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)

# =========================
# Config
# =========================
DIABETES_DB_DIR = "./chroma_diabetes_guideline"
DIABETES_COLLECTION_NAME = "diabetes_guideline"

DIABETES_RETRIEVER = None  # startup 때 로드

load_dotenv()  # .env 파일 로드

GMS_KEY = os.getenv("GMS_KEY")
GEMINI_URL = os.getenv("AI_URL")

# DB 설정
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# 로그 디렉토리 생성
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 캐릭터 캐시
CHARACTERS_CACHE: List[Dict[str, Any]] = []

# Runnable Chain 전역
RAG_CHAIN = None


# =========================
# Utils: Logging
# =========================
def save_log_to_json(log_data: Dict[str, Any]) -> str:
    """
    로그 데이터를 JSON 파일로 저장합니다.
    파일명: logs/log_YYYYMMDD_HHMMSS_microseconds.json
    Returns: 저장된 파일 경로
    """
    timestamp = datetime.now()
    filename = timestamp.strftime("log_%Y%m%d_%H%M%S_%f.json")
    filepath = os.path.join(LOG_DIR, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"[LOG] 로그 저장 완료: {filepath}")
        return filepath
    except Exception as e:
        print(f"[ERROR] 로그 저장 실패: {e}")
        return ""


# =========================
# DB
# =========================
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


def load_characters_from_db() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, name, ai_learning_comment FROM characters WHERE id >=3 ORDER BY id ASC"
            )
            return cursor.fetchall()
    finally:
        conn.close()


def _load_characters_cached() -> List[Dict[str, Any]]:
    global CHARACTERS_CACHE
    if CHARACTERS_CACHE:
        return CHARACTERS_CACHE
    # 캐시 비었으면 DB 조회 (fallback)
    return load_characters_from_db()


# =========================
# Pydantic Models
# =========================
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


# =========================
# Prompt Builders (기존 로직 유지)
# =========================
def build_prompt(ai_request: AiReportRequestModel, characters: List[Dict[str, Any]]) -> str:
    request_text = json.dumps(ai_request.model_dump(), ensure_ascii=False, indent=2)

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

캐릭터 목록:

{characters_block}

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
  4) 잠마니는 '아침 결식' 즉 아침에 먹은 값이 없을 경우 산정해준다. 아침에 먹은 값이 있으면 잠마니를 선택하지 않는다.

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

[절대 규칙 - 반드시 지켜라]
1) 결식(끼니를 걸렀음)이 1회라도 있으면, 최종 점수는 60점을 절대 초과할 수 없다. (CAP=60)
2) 결식이 2회 이상이면, 최종 점수는 40점을 절대 초과할 수 없다. (CAP=40)
3) 점수는 아래 항목 점수의 합(가감점)으로만 계산한다. 직감/칭찬/추측으로 점수를 올리지 마라.
4) 정보가 부족하면 보수적으로(낮게) 평가한다. (모르면 0점 처리 또는 최소 점수)
5) 출력은 JSON만. 설명 문장 금지.

[입력]
- meals: 아침/점심/저녁/간식 등의 섭취 목록

[평가 방법: base 50점에서 시작]
A. 결식 페널티 (가장 우선)
- 결식 1회: -25
- 결식 2회: -40
- 결식 3회: -60
(결식 횟수는 meals 중 비어있는 끼니 또는 isWaited=true이면 최소 1회로 판단)

B. 균형 점수(0~30)
- 탄/단/지 중 단백질이 포함된 끼니 수: 끼니당 +5 (최대 15)
- 채소/과일/식이섬유 추정 가능: 끼니당 +5 (최대 10)
- 과도한 당류/가공식품/음료만 섭취: 끼니당 -5 (최대 -10)

C. 과식/과다열량 패널티(0~-20)
- 고칼로리/튀김/패스트푸드로 추정되는 끼니: -7 (최대 -14)
- 야식/늦은 시간 폭식으로 추정: -6

D. 데이터 신뢰도(0~-10)
- 음식명이 불명확/추정치가 많음: -5
- 끼니 정보가 빈약함: -5



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


def build_prompt_diabetes(
    ai_request: AiReportRequestModel,
    characters: List[Dict[str, Any]],
    diabetes_context: str,
) -> str:
    request_text = json.dumps(ai_request.model_dump(), ensure_ascii=False, indent=2)

    characters_text_lines = []
    for ch in characters:
        desc = ch.get("ai_learning_comment", "")
        characters_text_lines.append(
            f"- id: {ch['id']}\n"
            f"  name: {ch['name']}\n"
            f"  description: {desc}"
        )
    characters_block = "\n\n".join(characters_text_lines)

    diabetes_block = (
        diabetes_context
        if diabetes_context
        else "※ 검색된 진료지침 문단이 충분하지 않습니다. 일반적인 당뇨병 식사요법 원칙을 바탕으로 답변하세요."
    )

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
3) 한국어로 맞춤 추천 멘트(comment)를 작성한다.
4) 약물 용량 조절/인슐린 조절 등 구체적인 의료 행위는 지시하지 말고,
   필요하면 "주치의와 상의" 수준으로만 안내한다.

최종 출력은 반드시 아래 JSON 형식 ONLY로 출력해야 한다:

{{
  "characterId": <정수>,
  "score": <실수 또는 정수>,
  "comment": "<한국어 멘트>"
}}

그 외 어떤 문장도 출력하지 마라.

------------------------------------------------------------
[1] 오늘자 AI 분석 요청 정보 (AiReportRequest JSON)

{request_text}

------------------------------------------------------------
[2] 캐릭터 목록

{characters_block}

------------------------------------------------------------
[3] 당뇨병 진료지침 RAG 결과

{diabetes_block}

[4] 캐릭터 선택 규칙 (1일 기준 단기 버전)

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
  4) 잠마니는 '아침 결식' 즉 아침에 먹은 값이 없을 경우 산정해준다. 아침에 먹은 값이 있으면 잠마니를 선택하지 않는다.

------------------------------------------------------------
[5] 점수(score) 산정 기준 (반드시 아래 수식 그대로 계산)

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

[절대 규칙 - 반드시 지켜라]
1) 결식(끼니를 걸렀음)이 1회라도 있으면, 최종 점수는 60점을 절대 초과할 수 없다. (CAP=60)
2) 결식이 2회 이상이면, 최종 점수는 40점을 절대 초과할 수 없다. (CAP=40)
3) 점수는 아래 항목 점수의 합(가감점)으로만 계산한다. 직감/칭찬/추측으로 점수를 올리지 마라.
4) 정보가 부족하면 보수적으로(낮게) 평가한다. (모르면 0점 처리 또는 최소 점수)
5) 출력은 JSON만. 설명 문장 금지.

[입력]
- meals: 아침/점심/저녁/간식 등의 섭취 목록

[평가 방법: base 50점에서 시작]
A. 결식 페널티 (가장 우선)
- 결식 1회: -25
- 결식 2회: -40
- 결식 3회: -60
(결식 횟수는 meals 중 비어있는 끼니 또는 isWaited=true이면 최소 1회로 판단)

B. 균형 점수(0~30)
- 탄/단/지 중 단백질이 포함된 끼니 수: 끼니당 +5 (최대 15)
- 채소/과일/식이섬유 추정 가능: 끼니당 +5 (최대 10)
- 과도한 당류/가공식품/음료만 섭취: 끼니당 -5 (최대 -10)

C. 과식/과다열량 패널티(0~-20)
- 고칼로리/튀김/패스트푸드로 추정되는 끼니: -7 (최대 -14)
- 야식/늦은 시간 폭식으로 추정: -6

D. 데이터 신뢰도(0~-10)
- 음식명이 불명확/추정치가 많음: -5
- 끼니 정보가 빈약함: -5



5) 퍼센티지 표시에 대한 내부 계산(멘트에 참고 가능)
- 퍼센티지 = (하루 총 섭취량 / 하루 권장량) × 100
- 이 퍼센티지는 멘트에서 "권장량 대비 몇 %" 같은 표현을 만들 때 참고해도 된다.

※ 주의:
- 나트륨은 점수에서 제외하지만, 멘트/캐릭터 선택에는 참고할 수 있다.
- score는 반드시 위 계산으로 산출한 값이어야 한다(감으로 주지 마라).
- 하루 전체 섭취량(dayTotalXXX)을 사용하여 계산하며, 끼니별 평균을 내지 않는다.

[6] 출력 형식

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
# Gemini Call
# =========================
def call_gemini(prompt: str) -> Tuple[Dict[str, Any], str]:
    """
    Gemini API를 호출하고 결과를 반환합니다.
    Returns:
        (파싱된 JSON dict, 원본 응답 텍스트)
    """
    if not GMS_KEY:
        raise RuntimeError("GMS_KEY 환경 변수가 설정되지 않았습니다.")
    if not GEMINI_URL:
        raise RuntimeError("AI_URL(GEMINI_URL) 환경 변수가 설정되지 않았습니다.")

    params = {"key": GMS_KEY}
    headers = {"Content-Type": "application/json"}
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(
        GEMINI_URL,
        params=params,
        headers=headers,
        data=json.dumps(body, ensure_ascii=False),
        timeout=90,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Gemini API 호출 실패: {response.status_code} {response.text}")

    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"예상하지 못한 Gemini 응답 형식: {data}")

    raw_text = text
    cleaned = text.strip()

    # ```json ... ``` 형태 처리
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"JSON 본문을 찾지 못했습니다. cleaned={cleaned!r}")

    json_str = cleaned[start : end + 1]

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패: {e}; json_str={json_str!r}")

    return result, raw_text


# =========================
# RAG + Runnable Chain (완전 체인화)
# =========================
def _make_rag_query(ai_request: AiReportRequestModel) -> str:
    return (
        f"당뇨병 환자의 하루 식단 요약. "
        f"총 칼로리 {ai_request.dayTotalKcal} kcal, "
        f"탄수화물 {ai_request.dayTotalCarb} g, "
        f"단백질 {ai_request.dayTotalProtein} g, "
        f"지방 {ai_request.dayTotalFat} g, "
        f"당류 {ai_request.dayTotalSugar} g, "
        f"나트륨 {ai_request.dayTotalNatrium} mg. "
        "당뇨병 환자의 식사요법, 혈당 관리, 탄수화물 조절, 나트륨 제한에 대한 진료지침."
    )


def _retrieve_diabetes_docs(inputs: Dict[str, Any]) -> List[Any]:
    global DIABETES_RETRIEVER
    if DIABETES_RETRIEVER is None:
        return []

    ai_request: AiReportRequestModel = inputs["ai_request"]
    query_text = _make_rag_query(ai_request)
    docs = DIABETES_RETRIEVER.invoke(query_text)  # k=4
    return docs or []


def _docs_to_context(docs: List[Any]) -> str:
    if not docs:
        return ""
    parts = []
    for d in docs:
        content = getattr(d, "page_content", None)
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _prompt_general(inputs: Dict[str, Any]) -> Dict[str, Any]:
    ai_request: AiReportRequestModel = inputs["ai_request"]
    characters: List[Dict[str, Any]] = inputs["characters"]
    prompt = build_prompt(ai_request, characters)
    return {**inputs, "prompt": prompt}


def _prompt_diabetes(inputs: Dict[str, Any]) -> Dict[str, Any]:
    ai_request: AiReportRequestModel = inputs["ai_request"]
    characters: List[Dict[str, Any]] = inputs["characters"]
    ctx: str = inputs.get("diabetes_context", "")
    prompt = build_prompt_diabetes(ai_request, characters, ctx)
    return {**inputs, "prompt": prompt}


def _gemini_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
    prompt: str = inputs["prompt"]
    gemini_result, raw = call_gemini(prompt)
    return {**inputs, "gemini_result": gemini_result, "gemini_raw": raw}


def _validate_to_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
    gemini_result: Dict[str, Any] = inputs.get("gemini_result", {})

    character_id = gemini_result.get("characterId", gemini_result.get("character_id"))
    score = gemini_result.get("score")
    comment = gemini_result.get("comment")

    if character_id is None or comment is None:
        raise HTTPException(status_code=500, detail=f"Gemini 응답에 필요한 필드가 없습니다: {gemini_result}")

    try:
        character_id = int(character_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=500, detail=f"characterId가 정수가 아닙니다: {character_id!r}")

    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None

    response = AiReportResponse(
        characterId=character_id,
        score=score,
        comment=str(comment),
    )

    return {**inputs, "final_response": response}


def build_runnable_chain():
    """
    최종 체인:
      입력 {"ai_request": AiReportRequestModel}
        -> characters 주입
        -> (당뇨면) retrieval + context
        -> prompt 생성(당뇨/일반 분기)
        -> gemini 호출
        -> 검증/응답 생성
      반환: dict (debug 포함) {"final_response": AiReportResponse, ...}
    """
    base = RunnablePassthrough()

    # characters 주입
    with_characters = base | RunnableLambda(
        lambda inputs: {**inputs, "characters": _load_characters_cached()}
    )

    # diabetes branch: docs + context
    diabetes_enrich = (
        with_characters
        | RunnableLambda(lambda inputs: {**inputs, "diabetes_docs": _retrieve_diabetes_docs(inputs)})
        | RunnableLambda(lambda inputs: {**inputs, "diabetes_context": _docs_to_context(inputs.get("diabetes_docs", []))})
    )

    # general branch: empty context
    general_enrich = with_characters | RunnableLambda(lambda inputs: {**inputs, "diabetes_docs": [], "diabetes_context": ""})

    # 분기: userIsDiaBetes
    enrich_branch = RunnableBranch(
        (lambda inputs: bool(inputs["ai_request"].userIsDiaBetes), diabetes_enrich),
        general_enrich,
    )

    # prompt 분기
    prompt_branch = RunnableBranch(
        (lambda inputs: bool(inputs["ai_request"].userIsDiaBetes), RunnableLambda(_prompt_diabetes)),
        RunnableLambda(_prompt_general),
    )

    chain = (
        enrich_branch
        | prompt_branch
        | RunnableLambda(_gemini_step)
        | RunnableLambda(_validate_to_response)
    )

    return chain


# =========================
# FastAPI App
# =========================
app = FastAPI(title="Foody Character Recommender API")


@app.on_event("startup")
def on_startup():
    """
    서버 시작 시 캐릭터 목록 + 당뇨 RAG 컬렉션 + Runnable Chain 로딩
    """
    global CHARACTERS_CACHE, DIABETES_RETRIEVER, RAG_CHAIN

    # 1) 캐릭터 캐시
    try:
        CHARACTERS_CACHE = load_characters_from_db()
        if not CHARACTERS_CACHE:
            print("[WARN] characters 테이블에서 가져온 데이터가 없습니다.")
        else:
            print(f"[INFO] {len(CHARACTERS_CACHE)}개의 캐릭터 정보를 로드했습니다.")
    except Exception as e:
        print(f"[ERROR] characters 로딩 실패: {e}")
        CHARACTERS_CACHE = []

    # 2) 당뇨 지침 Chroma + retriever
    try:
        print("\n" + "=" * 60)
        print("[STARTUP] 당뇨병 진료지침 RAG 초기화 시작")
        print("=" * 60)
        print(f"[STARTUP] ChromaDB 경로: {DIABETES_DB_DIR}")
        print(f"[STARTUP] Collection 이름: {DIABETES_COLLECTION_NAME}")

        if not os.path.exists(DIABETES_DB_DIR):
            print(f"[STARTUP ERROR] ChromaDB 디렉토리가 존재하지 않습니다: {DIABETES_DB_DIR}")
            DIABETES_RETRIEVER = None
        else:
            print("[STARTUP] 임베딩 모델 로딩 중...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("[STARTUP] 임베딩 모델 로딩 완료")

            print("[STARTUP] ChromaDB 연결 중...")
            vectordb = Chroma(
                persist_directory=DIABETES_DB_DIR,
                collection_name=DIABETES_COLLECTION_NAME,
                embedding_function=embeddings,
            )
            print("[STARTUP] ChromaDB 연결 완료")

            print("[STARTUP] Retriever 생성 중 (k=4)...")
            DIABETES_RETRIEVER = vectordb.as_retriever(search_kwargs={"k": 4})

            # 테스트 쿼리
            print("[STARTUP] 테스트 쿼리 실행 중...")
            test_docs = DIABETES_RETRIEVER.invoke("당뇨병 식사요법")
            print(f"[STARTUP] 테스트 쿼리 결과: {len(test_docs)}개 문서 검색됨")

            print("[STARTUP] ✅ Diabetes guideline retriever 로드 완료.")
            print("=" * 60 + "\n")

    except Exception as e:
        print(f"[STARTUP ERROR] ❌ Diabetes guideline Chroma 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        DIABETES_RETRIEVER = None

    # 3) Runnable Chain 빌드
    try:
        RAG_CHAIN = build_runnable_chain()
        print("[STARTUP] ✅ Runnable RAG_CHAIN 생성 완료")
    except Exception as e:
        print(f"[STARTUP ERROR] ❌ Runnable Chain 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        RAG_CHAIN = None


@app.post("/api/analysis/report", response_model=AiReportResponse)
def analyze_meal(ai_request: AiReportRequestModel):
    """
    Spring AiReportService.analyzeMeal() 이 호출할 엔드포인트
    - Runnable 체인 1번 호출로 전체 처리
    - 로그: request / diabetes_context / prompt / gemini raw / parsed / final_response 저장
    """
    log_data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "request": ai_request.model_dump(),
        "is_diabetes": ai_request.userIsDiaBetes,
        "diabetes_context": None,
        "final_prompt": None,
        "gemini_raw_response": None,
        "gemini_parsed_response": None,
        "final_response": None,
        "error": None,
    }

    print(f"[INFO] userIsDiaBetes = {ai_request.userIsDiaBetes}")

    try:
        if RAG_CHAIN is None:
            raise HTTPException(status_code=500, detail="RAG_CHAIN이 초기화되지 않았습니다.")

        # 체인 호출
        result_bundle: Dict[str, Any] = RAG_CHAIN.invoke({"ai_request": ai_request})

        # 디버그/로그 수집
        log_data["diabetes_context"] = result_bundle.get("diabetes_context")
        log_data["final_prompt"] = result_bundle.get("prompt")
        log_data["gemini_raw_response"] = result_bundle.get("gemini_raw")
        log_data["gemini_parsed_response"] = result_bundle.get("gemini_result")

        final_response: AiReportResponse = result_bundle.get("final_response")
        if final_response is None:
            raise HTTPException(status_code=500, detail="체인 결과에 final_response가 없습니다.")

        log_data["final_response"] = final_response.model_dump()
        save_log_to_json(log_data)
        return final_response

    except HTTPException as he:
        log_data["error"] = f"HTTPException: {he.detail}"
        save_log_to_json(log_data)
        raise

    except Exception as e:
        log_data["error"] = f"예상치 못한 에러: {str(e)}"
        save_log_to_json(log_data)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=True)
