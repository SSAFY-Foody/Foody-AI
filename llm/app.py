import os
import json
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from urllib.parse import urlparse

import requests
import pymysql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# GMS 설정
load_dotenv()  # .env 파일 내용 읽어오기

GMS_KEY = os.getenv("GMS_KEY")
GEMINI_URL = (
    "https://gms.ssafy.io/gmsapi/"
    "generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-pro:generateContent"
)


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


# report모델 정의

class Report(BaseModel):
    id: Optional[int] = None
    user_id: str

    score: Optional[float] = None
    comment: Optional[str] = None

    total_kcal: Optional[float] = None
    total_carb_g: Optional[float] = None
    total_protein_g: Optional[float] = None
    total_fat_g: Optional[float] = None
    total_sugar_g: Optional[float] = None
    total_natrium_g: Optional[float] = None

    is_waited: Optional[int] = None

    user_age: Optional[int] = None
    user_height: Optional[float] = None
    user_weight: Optional[float] = None
    user_gender: Optional[str] = None
    user_activity_level: Optional[int] = None

    user_std_weight: Optional[float] = None
    user_std_kcal: Optional[float] = None
    user_std_carb_g: Optional[float] = None
    user_std_protein_g: Optional[float] = None
    user_std_fat_g: Optional[float] = None
    user_std_sugar_g: Optional[float] = None
    user_std_natrium_g: Optional[float] = None



# 프롬프트 빌더

def build_prompt(report: Report, characters: List[Dict[str, Any]]) -> str:
    report_text = json.dumps(report.model_dump()
                             , ensure_ascii=False,
                                indent=2)

    # 캐릭터 목록을 텍스트로 정리
    # 각 캐릭터는 id, name, description(ai_learning_comment)를 가진다고 설명
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
    - [1] 오늘자 reports JSON
    - [2] DB에서 가져온 캐릭터 목록(id, name, ai_learning_comment)
    를 기반으로, 너가 준 1일 기준 규칙/출력 형식을 포함한 프롬프트 생성
    너는 식단 관리 서비스 '푸디(Foody)'의 캐릭터 추천 AI야.

    너의 임무는 당일 1일치 레포트(reports) 데이터를 기반으로,
    푸디 캐릭터들 중에서 오늘의 식습관을 가장 잘 나타내는 캐릭터 1명을 선택하고,
    한국어로 맞춤 추천 멘트(comment)를 제공하는 것이다.

    아래 "사용자 리포트 정보"와 "푸디 캐릭터 설명"을 참고해,
    오늘 하루의 섭취 성향을 가장 잘 표현하는 캐릭터 하나를 고르고,
    추천 이유 및 개선 방향을 comment에 작성해라.

    최종 출력은 반드시 아래 JSON 형식 ONLY:

        {{
        "character_id": <정수>,
        "comment": "<한국어 멘트>"
        }}

    그 외 문장은 출력하지 마라.

    [1] 오늘자 사용자 리포트 정보 (1일 기준)

    아래 JSON은 reports 테이블에서 가져온, 오늘 하루 기준 1일치 사용자 레포트다.

    {report_text}

    설명:
    - total_kcal: 오늘 하루 총 섭취 칼로리
    - total_carb_g / total_protein_g / total_fat_g: 당일 탄단지 섭취량
    - total_sugar_g: 당류 섭취량
    - total_natrium_g: 나트륨 섭취량
    - is_waited: 오늘 AI/트레이너 코멘트 대기 여부 (캐릭터 선택에는 사용하지 않는다)
    - user_age, user_gender, user_height, user_weight, user_activity_level 등: 사용자 기초정보
    - user_std_kcal, user_std_carb_g 등: 사용자 인적사항 기반 권장 영양소(1일 기준)

    즉, **오늘 하루의 섭취량 vs 권장 섭취량**만 비교해서 판단해야 한다.
    누적 패턴, 여러 날 평균, 장기 습관은 절대 고려하지 않는다.

    ------------------------------------------------------------
    [2] 1일 기준 Foody 캐릭터 설명 (DB에서 불러온 단기판)

    아래는 데이터베이스에서 불러온 푸디 캐릭터 목록이다.
    각 캐릭터는 다음 정보를 가진다:

    - id: 캐릭터를 나타내는 정수형 ID
    - name: 캐릭터 이름
    - description: 이 캐릭터가 나타내는 오늘 하루 식습관/영양 상태에 대한 상세 설명(ai_learning_comment 필드)

    캐릭터 목록:

    {characters_block}

    이 description에는 각각의 캐릭터가 어떤 하루를 의미하는지,
    예를 들어 "오늘 칼로리 낮고 단백질을 잘 챙긴 하루", "오늘 나트륨이 과다한 하루",
    "오늘 단 음식/간식 위주로 먹은 하루", "오늘 영양 균형이 잘 맞는 하루" 등과 같이,
    **1일 기준으로 해석해야 하는 캐릭터 설명이 이미 포함되어 있다.**

    너는 위 description 텍스트를 그대로 참고해서,
    오늘자 리포트와 가장 잘 어울리는 캐릭터 1명을 고르면 된다.

    ------------------------------------------------------------
    [3] 캐릭터 선택 규칙 (단기 1일 버전)

    - 오늘 하루 기록만 보고 즉시 판단한다.
    - 가장 두드러진 항목(칼로리 과/저체, 당류 과다, 나트륨 과다 등)을 우선 반영한다.
    - 캐릭터 description에 이미 어떤 상황에 쓰이는 캐릭터인지 상세히 적혀 있으므로,
    각 캐릭터의 설명과 오늘자 리포트의 수치를 비교해서 가장 잘 맞는 캐릭터를 선택해라.

    - 만약 2개 이상 매칭될 경우 우선순위는 다음과 같다 (이름은 description에 적힌 의미와 연결된다):

    1) 짜구리(나트륨 과다), 달다구리(당류 과다), 주전부엉(간식 위주), 왕마니(과식)
    2) 슬리만더(저칼로리 + 고단백), 요마니(전반적으로 매우 적은 섭취)
    3) 탄단지오(영양 균형이 좋은 하루)
    4) 새싹 푸디(데이터 부족 / 특징 불명확)
    5) 잠마니는 '아침 결식' 등 별도 조건이 있을 때만 선택

    - 출력에는 character_id 하나만 정수로 반환한다.
    - comment는 한국어로 오늘 식단에 맞춘 설명과 개선 팁 2~3가지를 작성해라.
    - 예: "오늘은 나트륨이 많았어요, 라면/찌개 국물은 조금 남겨 보는 건 어떨까요?" 같은 톤.

    ------------------------------------------------------------
    [4] 출력 형식

    반드시 아래 형태로만 출력해야 함:

    {{
    "character_id": <정수>,
    "comment": "<한국어 멘트>"
    }}

그 외 어떤 문장도 출력하지 마라.
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
                    {
                        "text": prompt
                    }
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

    # 디버깅용: 원본 출력 한번 찍어보기
    print("=== RAW GEMINI OUTPUT ===")
    print(text)
    print("=== END RAW GEMINI OUTPUT ===")

    # 1. 앞뒤 공백 제거
    cleaned = text.strip()

    #백틱 형태로 감싸져 있을경우 제거
    if cleaned.startswith("```"):
        # 양쪽 ``` 제거
        cleaned = cleaned.strip("`").strip()
        # 맨 앞이 json 이나 JSON 으로 중복될 수 있으니 이름 떼기
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    # JSON 시작/끝 위치 찾기 (첫 { ~ 마지막 })
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"JSON 본문을 찾지 못했습니다. cleaned={cleaned!r}")

    json_str = cleaned[start:end + 1]

    # 디버깅용: 실제 파싱할 문자열
    print("=== PARSED JSON STRING ===")
    print(json_str)
    print("=== END PARSED JSON STRING ===")

    # 4. 최종 파싱
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패: {e}; json_str={json_str!r}")

    return result


# =========================
# FastAPI 애플리케이션
# =========================

app = FastAPI(title="Foody Character Recommender API")

CHARACTERS_CACHE: List[Dict[str, Any]] = []


@app.on_event("startup")
def on_startup():
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


@app.post("/report")
def recommend_character(report: Report):
    
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
        print("[INFO] 캐릭터 정보를 캐시에서 사용합니다.", characters)

    prompt = build_prompt(report, characters)

    try:
        gemini_result = call_gemini(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "character_id" not in gemini_result or "comment" not in gemini_result:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini 응답에 필요한 필드가 없습니다: {gemini_result}",
        )

    return {
        "character_id": int(gemini_result["character_id"]),
        "comment": str(gemini_result["comment"]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=True)