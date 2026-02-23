"""
test_search.py
ChromaDBのベクトル検索動作確認スクリプト。
食材を入力して料理が返ってくるか確認する。
【改善版】食材を1つずつ個別検索する方式に変更
"""

import json
import re
import chromadb
from chromadb.utils import embedding_functions

# ────────────────────────────
# 設定
# ────────────────────────────
CHROMA_DIR = "./chroma_db"
RECIPE_COLLECTION = "recipes"
INGREDIENT_COLLECTION = "ingredients"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def get_collections():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    recipe_col = client.get_collection(
        name=RECIPE_COLLECTION, embedding_function=embed_fn
    )
    ingredient_col = client.get_collection(
        name=INGREDIENT_COLLECTION, embedding_function=embed_fn
    )
    return recipe_col, ingredient_col


def split_input(user_input: str) -> list:
    """
    入力テキストを食材単語に分割する。
    本番はGroqが担当するが、テスト用にシンプルな分割を実装。
    「と」「や」「に」「も」「、」「，」「スペース」で分割する。
    """
    tokens = re.split(r'[とやにも、，\s　]+', user_input)
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 1]
    return tokens


def search_one_ingredient(ingredient_col, word: str) -> dict | None:
    """
    1単語で食材を1件だけ検索して返す。
    距離が0.35より遠い場合はノイズとして除外。
    """
    results = ingredient_col.query(query_texts=[word], n_results=1)
    if not results["metadatas"][0]:
        return None
    meta = results["metadatas"][0][0]
    distance = results["distances"][0][0]
    if distance > 0.35:
        return None
    return {
        "食材名": meta["食材名"],
        "カテゴリ": json.loads(meta["カテゴリ"]),
        "生食可": meta["生食可"] == "True",
        "距離": round(distance, 4),
        "入力単語": word,
    }


def search_recipes_by_categories(recipe_col, categories: list, n=5) -> list:
    """食材カテゴリのリストで料理をベクトル検索する"""
    query = "、".join(categories) + "を使った料理"
    results = recipe_col.query(query_texts=[query], n_results=n)
    hits = []
    for i, meta in enumerate(results["metadatas"][0]):
        recipe_categories = json.loads(meta["使える食材カテゴリ"])
        match_count = len(set(categories) & set(recipe_categories))
        hits.append({
            "name": meta["name"],
            "ジャンル": meta["ジャンル"],
            "必要調理法": meta["必要調理法"],
            "加熱": meta["加熱"] == "True",
            "本物の食材": json.loads(meta["本物の食材"]),
            "使える食材カテゴリ": recipe_categories,
            "加工手順": json.loads(meta["加工手順"]),
            "説明文": meta["説明文"],
            "一致カテゴリ数": match_count,
            "距離": round(results["distances"][0][i], 4),
        })
    # カテゴリ一致数が多い順、距離が近い順でソート
    hits.sort(key=lambda x: (-x["一致カテゴリ数"], x["距離"]))
    return hits


def run_test(user_input: str):
    print(f"\n{'='*50}")
    print(f"入力：「{user_input}」")
    print("=" * 50)

    recipe_col, ingredient_col = get_collections()

    # ① 入力を単語に分割
    words = split_input(user_input)
    print(f"\n【分割結果】{words}")

    # ② 食材を1つずつ検索
    print("\n【食材検索結果】")
    found_ingredients = []
    for word in words:
        hit = search_one_ingredient(ingredient_col, word)
        if hit:
            raw = "生食可" if hit["生食可"] else "加熱必要"
            print(f"  「{hit['入力単語']}」→ {hit['食材名']} "
                  f"({'/'.join(hit['カテゴリ'])}) [{raw}] 距離:{hit['距離']}")
            found_ingredients.append(hit)
        else:
            print(f"  「{word}」→ 該当なし（距離が遠すぎるため除外）")

    # ③ カテゴリを集める（重複なし）
    found_categories = []
    for ing in found_ingredients:
        for cat in ing["カテゴリ"]:
            if cat not in found_categories:
                found_categories.append(cat)

    print(f"\n【検出されたカテゴリ】{found_categories}")

    if not found_categories:
        print("⚠️ カテゴリが検出されませんでした。救済ルートに入るケースです。")
        return

    # ④ 料理を検索
    print("\n【料理検索結果（カテゴリ一致数→距離でソート済み）】")
    recipe_hits = search_recipes_by_categories(recipe_col, found_categories, n=5)
    for hit in recipe_hits:
        heat = "加熱あり" if hit["加熱"] else "加熱なし"
        print(f"  {hit['name']} ({hit['ジャンル']}) [{heat}] "
              f"カテゴリ一致:{hit['一致カテゴリ数']} 距離:{hit['距離']}")
        print(f"    本物の食材：{hit['本物の食材']}")
        print(f"    加工手順：{hit['加工手順']}")


# ────────────────────────────
# テストケース
# ────────────────────────────
if __name__ == "__main__":
    print("ゆるゆるコックさん ベクトル検索テスト【改善版】")

    # テスト1：定番食材（複数カテゴリ）
    run_test("卵とご飯とねぎ")

    # テスト2：代替品が出てきそうな食材
    run_test("魚肉ソーセージとピーマン")

    # テスト3：万能メニューが出てきそうな食材
    run_test("キムチと豆腐")

    # テスト4：雑な入力
    run_test("冷蔵庫に肉と野菜がある")

    # テスト5：救済ルートになりそうな入力
    run_test("砂糖と塩")

    print("\n\n✅ テスト完了だぞい！")
