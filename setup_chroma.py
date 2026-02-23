"""
setup_chroma.py
料理DB・食材DBをChromaDBに登録するスクリプト。
初回セットアップ時に一度だけ実行する。
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

# ────────────────────────────
# 設定
# ────────────────────────────
CHROMA_DIR = "./chroma_db"          # ChromaDBの保存先
RECIPE_JSON = "./data/recipe_db.json"
INGREDIENT_JSON = "./data/ingredient_db.json"

RECIPE_COLLECTION = "recipes"
INGREDIENT_COLLECTION = "ingredients"

# 埋め込みモデル（sentence-transformers の日本語対応モデル）
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_recipe_document(recipe: dict) -> str:
    """料理DBの1件をベクトル検索用のテキストに変換する"""
    ingredients = "、".join(recipe["本物の食材"])
    categories = "、".join(recipe["使える食材カテゴリ"])
    steps = "、".join(recipe.get("加工手順", []))
    return (
        f"{recipe['name']}。"
        f"ジャンル：{recipe['ジャンル']}。"
        f"食材：{ingredients}。"
        f"使える食材カテゴリ：{categories}。"
        f"調理法：{recipe['必要調理法']}。"
        f"手順：{steps}。"
        f"{recipe['説明文']}"
    )


def build_ingredient_document(ingredient: dict) -> str:
    """食材DBの1件をベクトル検索用のテキストに変換する"""
    categories = "、".join(ingredient["カテゴリ"])
    raw = "生食可" if ingredient["生食可"] else "加熱必要"
    name = ingredient["食材名"]

    # 食材名を冒頭に3回繰り返す（表記ゆれ対策）
    # 例：「ねぎ ねぎ ねぎ。ネギ。葱。」のようにすることで
    # 「ねぎ」「ネギ」「葱」「ネギ」どの入力でも近くなる
    name_emphasis = f"{name} {name} {name}。"

    return (
        f"{name_emphasis}"
        f"カテゴリ：{categories}。"
        f"{raw}。"
        f"{ingredient['説明']}"
    )


def register_recipes(collection, recipes: list):
    """料理DBをChromaDBに登録する"""
    documents = []
    metadatas = []
    ids = []

    for i, recipe in enumerate(recipes):
        doc = build_recipe_document(recipe)
        meta = {
            "name": recipe["name"],
            "ジャンル": recipe["ジャンル"],
            "加熱": str(recipe["加熱"]),  # ChromaDBはboolを受け付けないのでstr変換
            "必要調理法": recipe["必要調理法"],
            "本物の食材": json.dumps(recipe["本物の食材"], ensure_ascii=False),
            "使える食材カテゴリ": json.dumps(recipe["使える食材カテゴリ"], ensure_ascii=False),
            "加工手順": json.dumps(recipe.get("加工手順", []), ensure_ascii=False),
            "説明文": recipe["説明文"],
        }
        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"recipe_{i:03d}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"  料理DB：{len(recipes)}件を登録しました")


def register_ingredients(collection, ingredients: list):
    """食材DBをChromaDBに登録する"""
    documents = []
    metadatas = []
    ids = []

    for i, ingredient in enumerate(ingredients):
        doc = build_ingredient_document(ingredient)
        meta = {
            "食材名": ingredient["食材名"],
            "カテゴリ": json.dumps(ingredient["カテゴリ"], ensure_ascii=False),
            "生食可": str(ingredient["生食可"]),  # ChromaDBはboolを受け付けないのでstr変換
            "説明": ingredient["説明"],
        }
        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"ingredient_{i:03d}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"  食材DB：{len(ingredients)}件を登録しました")


def main():
    print("=" * 40)
    print("ゆるゆるコックさん ChromaDB セットアップ")
    print("=" * 40)

    # JSONを読み込む
    print("\n[1] JSONファイルを読み込み中...")
    recipes = load_json(RECIPE_JSON)
    ingredients = load_json(INGREDIENT_JSON)
    print(f"  料理DB：{len(recipes)}件")
    print(f"  食材DB：{len(ingredients)}件")

    # ChromaDBクライアントを作成
    print("\n[2] ChromaDBを初期化中...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 埋め込み関数を設定
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # 既存のコレクションを削除（再実行時にクリーンにする）
    for col_name in [RECIPE_COLLECTION, INGREDIENT_COLLECTION]:
        try:
            client.delete_collection(col_name)
            print(f"  既存の「{col_name}」コレクションを削除しました")
        except Exception:
            pass  # 初回は存在しないので無視

    # コレクションを作成
    recipe_col = client.create_collection(
        name=RECIPE_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    ingredient_col = client.create_collection(
        name=INGREDIENT_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # データを登録
    print("\n[3] データを登録中...")
    print("  ※初回は埋め込みモデルのダウンロードがあるので少し時間がかかります")
    register_recipes(recipe_col, recipes)
    register_ingredients(ingredient_col, ingredients)

    # 登録件数を確認
    print("\n[4] 登録件数を確認中...")
    print(f"  recipes コレクション：{recipe_col.count()}件")
    print(f"  ingredients コレクション：{ingredient_col.count()}件")

    print("\n✅ セットアップ完了だぞい！")
    print(f"   ChromaDBの保存先：{os.path.abspath(CHROMA_DIR)}")


if __name__ == "__main__":
    main()
