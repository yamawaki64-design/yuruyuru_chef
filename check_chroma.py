# check_chroma.py
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

col = client.get_collection(name="ingredients", embedding_function=embed_fn)

# 「ねぎ」「ご飯」「キムチ」「豆腐」のドキュメントを直接取得
results = col.get(
    where={"食材名": {"$in": ["ねぎ", "ご飯", "キムチ", "豆腐"]}},
    include=["documents", "metadatas"]
)

for i, doc in enumerate(results["documents"]):
    name = results["metadatas"][i]["食材名"]
    print(f"\n【{name}】")
    print(f"  登録テキスト: {doc}")

# check_chroma.py の末尾に追加
# 「ねぎ」で検索したとき何が返ってくるか距離付きで確認
print("\n\n=== 「ねぎ」で検索した結果 ===")
results2 = col.query(query_texts=["ねぎ"], n_results=5)
for i, meta in enumerate(results2["metadatas"][0]):
    print(f"  {meta['食材名']} 距離:{round(results2['distances'][0][i], 4)}")

print("\n=== 「ご飯」で検索した結果 ===")
results3 = col.query(query_texts=["ご飯"], n_results=5)
for i, meta in enumerate(results3["metadatas"][0]):
    print(f"  {meta['食材名']} 距離:{round(results3['distances'][0][i], 4)}")