# app.py
import os
import json
import re
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
import requests
import pymysql

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================
# ENV / CONSTANTS
# =========================
DIABETES_DB_DIR = "./chroma_diabetes_guideline"
DIABETES_COLLECTION_NAME = "diabetes_guideline"

load_dotenv()

GMS_KEY = os.getenv("GMS_KEY")
GEMINI_URL = os.getenv("AI_URL")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


# =========================
# GLOBALS (중요: 전역으로 잡아야 startup에서 세팅한 걸 함수들이 사용함)
# =========================
CHARACTERS_CACHE: List[Dict[str, Any]] = []
DIABETES_RETRIEVER = None  # ✅ 전역 retriever


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
    """
    ⚠️ 규칙상 id=2(새싹푸디)는 '선정하면 안됨'이지만
    DB에서 프롬프트에 포함시키는 건 가능(선정 금지 규칙이 프롬프트에 있음)
    다만 네가 원래 SQL에서 id>=3로 제한하고 싶다면 WHERE id >= 3 유지하면 됨.

    여기서는 "프롬프트에 보이는 값"과 "실제 데이터" 불일치 디버깅을 위해
    id>=2로 가져오고, 프롬프트 규칙으로 선정 금지시키는 형태로 맞춤.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, name, ai_learning_comment "
                "FROM characters "
                "WHERE id >= 3 "
                "ORDER BY id ASC"
            )
            rows = cursor.fetchall()
            print("[DEBUG] loaded character ids:", [r["id"] for r in rows][:50])
            return rows
    finally:
        conn.close()


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
    stdInfo: StdInfo
    userActivityLevelDesc: str
    userIsDiaBetes: bool

    dayTotalKcal: float
    dayTotalCarb: float
    dayTotalProtein: float
    dayTotalFat: float
    dayTotalSugar: float
    dayTotalNatrium: float

    meals: List[MealInfo]


class AiReportResponse(BaseModel):
    score: Optional[float]
    comment: str
    characterId: int


# =========================
# Prompt Builders
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

------------------------------------------------------------
[2] 1일 기준 Foody 캐릭터 설명 (DB에서 불러온 단기판)

캐릭터 목록:

{characters_block}

------------------------------------------------------------
[3] 캐릭터 선택 규칙 (1일 기준 단기 버전)

- 오늘 하루 기록만 보고 즉시 판단한다.
- 가장 두드러진 문제/특징(칼로리 과/저체, 단백질 부족, 당류 과다, 나트륨 과다, 간식 과다 등)을 우선 반영한다.
- 캐릭터 description을 참고해서 오늘자 수치와 가장 잘 맞는 캐릭터 1명을 고른다.
- 2번 새싹푸디는 초기값이므로 절대 선택하면 안 된다.
- 여러 캐릭터가 겹치는 경우 우선순위 예시는 다음과 같다:

  1) 짜구리(나트륨 과다), 달다구리(당류 과다), 주전부엉(간식 위주), 왕마니(과식)
  2) 슬리만더(저칼로리 + 고단백), 요마니(전반적으로 매우 적은 섭취)
  3) 탄단지오(영양 균형이 좋은 하루)
  4) 잠마니는 '아침 결식' 등 별도 조건이 있을 때만 선택

------------------------------------------------------------
[4] 점수(score) 산정 기준 (반드시 아래 수식 그대로 계산)

(너가 작성한 규칙/수식이 매우 길어서 그대로 유지하려면 여기에 원문을 통째로 붙여도 되고,
 지금처럼 이미 프롬프트에 포함하고 있었다면 그대로 복붙해도 됨.
 단, 토큰이 길어져 응답 불안정하면 이 구간을 "핵심 규칙 요약"으로 압축하는 게 더 안정적임.)

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


def build_diabetes_context(ai_request: AiReportRequestModel) -> str:
    """
    ✅ 전역 DIABETES_RETRIEVER 사용
    ✅ Document 리스트를 page_content로 join
    """
    global DIABETES_RETRIEVER
    if DIABETES_RETRIEVER is None:
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
        docs = DIABETES_RETRIEVER.get_relevant_documents(query_text)
        if not docs:
            return ""
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"[WARN] 당뇨 지침 Chroma 조회 실패: {e}")
        return ""


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

너에게는 다음 정보가 주어진다:
- [1] 오늘자 AI 분석 요청 전체 JSON (AiReportRequest)
- [2] DB에서 가져온 푸디 캐릭터 목록 (id, name, ai_learning_comment)
- [3] 당뇨병 진료지침에서 검색한 관련 문단 요약 (RAG 결과)

너의 임무:
1) 오늘 하루의 섭취 성향을 가장 잘 표현하는 캐릭터 1명을 선택한다.
2) 오늘 식단에 대한 평가 점수(score)를 0~100 사이 실수로 준다.
3) 한국어로 맞춤 추천 멘트(comment)를 작성한다.
4) 약물/인슐린 용량 조절 같은 의료 행위는 지시하지 말고, 필요 시 "주치의와 상의"로 안내한다.

최종 출력은 반드시 아래 JSON 형식 ONLY로 출력해야 한다:

{{
  "characterId": <정수>,
  "score": <실수 또는 정수>,
  "comment": "<한국어 멘트>"
}}

그 외 어떤 문장도 출력하지 마라. 오직 JSON만.

------------------------------------------------------------
[1] 오늘자 AI 분석 요청 정보 (AiReportRequest JSON)

{request_text}

------------------------------------------------------------
[2] 캐릭터 목록

{characters_block}

------------------------------------------------------------
[3] 당뇨병 진료지침 RAG 결과

{diabetes_block}

------------------------------------------------------------
[4] 출력 형식

{{
  "characterId": <정수>,
  "score": <실수 또는 정수>,
  "comment": "<한국어 멘트>"
}}
"""
    return prompt.strip()


# =========================
# Gemini Call
# =========================
def call_gemini(prompt: str) -> Dict[str, Any]:
    if not GMS_KEY:
        raise RuntimeError("GMS_KEY 환경 변수가 설정되지 않았습니다.")
    if not GEMINI_URL:
        raise RuntimeError("AI_URL(GEMINI_URL) 환경 변수가 설정되지 않았습니다.")

    params = {"key": GMS_KEY}
    headers = {"Content-Type": "application/json"}

    body = {"contents": [{"parts": [{"text": prompt}]}]}

    # ✅ 디버깅 로그: 프롬프트가 실제로 얼마나 들어갔는지 확인
    print("[DEBUG] prompt length:", len(prompt))
    print("[DEBUG] prompt head(300):\n", prompt[:300])
    print("[DEBUG] prompt tail(300):\n", prompt[-300:])

    response = requests.post(
        GEMINI_URL,
        params=params,
        headers=headers,
        json=body,  # ✅ data= 대신 json=
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Gemini API 호출 실패: {response.status_code} {response.text}")

    data = response.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"예상하지 못한 Gemini 응답 형식: {data}")

    print("=== RAW GEMINI OUTPUT ===")
    print(text)
    print("=== END RAW GEMINI OUTPUT ===")

    cleaned = text.strip()

    # ✅ fence 제거를 더 안전하게
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"JSON 본문을 찾지 못했습니다. cleaned={cleaned!r}")

    json_str = cleaned[start : end + 1]

    print("=== PARSED JSON STRING ===")
    print(json_str)
    print("=== END PARSED JSON STRING ===")

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패: {e}; json_str={json_str!r}")

    return result


# =========================
# FastAPI App
# =========================
app = FastAPI(title="Foody Character Recommender API")


@app.on_event("startup")
def on_startup():
    """
    ✅ 서버 시작 시 캐릭터 캐시 + 당뇨 RAG retriever 전역 세팅
    """
    global CHARACTERS_CACHE, DIABETES_RETRIEVER

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
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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


@app.post("/api/analysis/report", response_model=AiReportResponse)
def analyze_meal(ai_request: AiReportRequestModel):
    print(f"[INFO] userIsDiaBetes = {ai_request.userIsDiaBetes}")

    # 캐릭터 로딩 fallback
    if not CHARACTERS_CACHE:
        try:
            characters = load_characters_from_db()
            print("[INFO] 캐릭터 정보를 DB에서 즉시 로드합니다.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"characters 정보를 불러오지 못했습니다: {e}")
    else:
        characters = CHARACTERS_CACHE
        print("[INFO] 캐릭터 정보를 캐시에서 사용합니다.")

    # 프롬프트 분기
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

    # 키 이름 통일
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

    return AiReportResponse(characterId=character_id, score=score, comment=str(comment))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=True)
