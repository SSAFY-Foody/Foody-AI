import chromadb
import json

DB_DIR = "./chroma_diabetes_guideline"

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("diabetes_guideline")

data = collection.get()

# JSON 구조로 변환
json_output = {
    "count": len(data["ids"]),
    "items": [
        {
            "id": data["ids"][i],
            "document": data["documents"][i],
            "metadata": data["metadatas"][i] if data.get("metadatas") else None
        }
        for i in range(len(data["ids"]))
    ]
}

# JSON 예쁘게 출력
print(json.dumps(json_output, indent=2, ensure_ascii=False))

with open("chroma_dump.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2, ensure_ascii=False)

print("JSON 파일 저장 완료: chroma_dump.json")

