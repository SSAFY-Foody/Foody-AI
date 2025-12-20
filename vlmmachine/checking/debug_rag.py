"""
ChromaDB RAG 검색 테스트 스크립트
사용법: python debug_rag.py "음식명"
"""
import sys
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = os.getenv("FOODY_CHROMA_DIR", "./chroma_foods")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=os.getenv("FOODY_EMBED_MODEL", "jhgan/ko-sroberta-multitask")
)

food_collection = chroma_client.get_or_create_collection(
    name="food_nutrition",
    embedding_function=ko_embedding,
)

def test_rag_search(food_name: str, top_k: int = 5):
    print(f"Testing RAG search for: '{food_name}'")
    print(f"Collection size: {food_collection.count()}")
    print("=" * 80)
    
    result = food_collection.query(query_texts=[food_name], n_results=top_k)
    
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    documents = result.get("documents", [[]])[0]
    
    if not metadatas:
        print("No results found!")
        return
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"\nResult #{i+1}:")
        print(f"  Document: {doc}")
        print(f"  Distance: {dist:.4f}")
        print(f"  Nutrition:")
        print(f"    - kcal: {meta.get('kcal')}")
        print(f"    - carb: {meta.get('carb')}g")
        print(f"    - protein: {meta.get('protein')}g")
        print(f"    - fat: {meta.get('fat')}g")
        print(f"    - sugar: {meta.get('sugar')}g")
        print(f"    - natrium: {meta.get('natrium')}mg")
    
    # 평균 계산 (현재 방식)
    n = len(metadatas)
    avg_kcal = sum(m.get("kcal", 0.0) for m in metadatas) / n
    avg_carb = sum(m.get("carb", 0.0) for m in metadatas) / n
    avg_protein = sum(m.get("protein", 0.0) for m in metadatas) / n
    avg_fat = sum(m.get("fat", 0.0) for m in metadatas) / n
    
    print(f"\n{'='*80}")
    print(f"Current method (Top-{n} Average):")
    print(f"  kcal: {avg_kcal:.2f}, carb: {avg_carb:.2f}g, protein: {avg_protein:.2f}g, fat: {avg_fat:.2f}g")
    
    print(f"\nProposed method (Top-1 Only):")
    print(f"  kcal: {metadatas[0].get('kcal')}, carb: {metadatas[0].get('carb')}g, "
          f"protein: {metadatas[0].get('protein')}g, fat: {metadatas[0].get('fat')}g")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 기본 테스트 케이스
        test_cases = ["계란말이", "오믈렛", "김밥", "밥", "샐러드"]
        for food in test_cases:
            test_rag_search(food, top_k=3)
            print("\n" + "=" * 80 + "\n")
    else:
        food_name = sys.argv[1]
        test_rag_search(food_name, top_k=5)
