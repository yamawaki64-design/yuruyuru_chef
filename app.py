"""
app.py
ゆるゆるコックさん - Groq版（全4画面）
ステップ4：Groq追加・3種類のセリフをGroqで生成
"""

import json
import random
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from groq import Groq

# ────────────────────────────
# ページ設定
# ────────────────────────────
st.set_page_config(
    page_title="ゆるゆるコックさん",
    page_icon="🍳",
    layout="centered",
)

# ────────────────────────────
# 定数
# ────────────────────────────
CHROMA_DIR = "./chroma_db"
RECIPE_COLLECTION = "recipes"
INGREDIENT_COLLECTION = "ingredients"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# 一致率による前置き
MATCH_PREFIXES = {
    90: "完璧に",
    70: "かなりいい感じに",
    50: "まあまあ",
    30: "かなり無理くりだけど",
    0:  "ほぼ無理やりだけど",
}

# ジャンル別調味料ヒント
SEASONING_HINTS = {
    "和食": "醤油・みりん・砂糖・だしの素があると和食っぽくなるぞい。でも実はめんつゆだけでもなんとかなるぞい",
    "洋食": "塩・こしょう・バターがあると洋食っぽくなるぞい。ケチャップやマヨネーズも強い味方になってくれるぞい",
    "中華": "醤油・ごま油・オイスターソースがあると中華っぽくなるぞい。でも鶏がらスープの素とチューブのニンニクの合わせ技も捨てがたいぞい",
    "エスニック": "ナンプラーかごま油があるとエスニックっぽくなるぞい。なかったら醤油とチューブのニンニクで代用するといいぞい",
}

# ジャンル別食べ方ヒント
EATING_HINTS = {
    "和食": "ご飯と一緒に食べるとおいしいぞい。汁物があるとさらにいいぞい",
    "洋食": "パンと一緒でもご飯と一緒でもおいしいぞい",
    "中華": "白いご飯と一緒に食べると最高だぞい",
    "エスニック": "ご飯と一緒でも麺類と一緒でもいけるぞい",
}

# 買い物アドバイス（救済版）
SHOPPING_ADVICE = [
    "ふりかけとかお漬物とか卵買っとくと、ご飯がおいしく食べれるぞい",
    "卵と豆腐があればだいたいなんとかなるぞい。買っておくといいぞい",
    "缶詰（ツナとかサバとか）を棚に常備しておくと便利だぞい",
    "冷凍うどんとか冷凍チャーハンがあると、何もないときに助かるぞい",
    "納豆はご飯さえあればそれだけで立派な食事になるぞい",
    "インスタントのスープや味噌汁やわかめスープがあると、お湯だけで1品増やせるぞい",
    "調味料に迷ったら、塩だけ醬油だけでもなんとかなるぞい",
]


# ────────────────────────────
# Groqクライアント
# ────────────────────────────
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# ────────────────────────────
# Groqセリフ生成（① 食材解析）
# ────────────────────────────
def groq_normalize_ingredients(user_input: str) -> tuple[list[str], str]:
    """
    ユーザーの入力テキストをGroqで解析し、正規化された食材リストとセリフを返す。
    戻り値: (正規化食材リスト, セリフ文字列)
    失敗時: ([], "") を返す
    """
    try:
        client = get_groq_client()
        prompt = f"""あなたは食材を正規化する専門家です。
ユーザーが入力した食材テキストを解析して、以下のJSON形式で返してください。

入力テキスト：「{user_input}」

ルール：
- 表記ゆれを正規化する（例：たまご→卵、冷ごはん→ご飯、ネギ→ねぎ）
- 修飾語を除去して食材名だけにする（例：残り物のハム→ハム）
- 日本語の一般的な食材名に統一する
- 食材ではないもの（調理法・量・状態など）は除外する

返すJSONの形式（他のテキストは一切含めないこと）：
{{
  "ingredients": ["食材1", "食材2", "食材3"],
  "message": "○○と△△と□□があるんだぞい！ちょっと考えてみるぞい…"
}}

messageは「ゆるゆるコックさん」というキャラクターのセリフで、語尾は「〜ぞい」「〜だぞい」を使い、食材名を入れて元気よく書いてください。
食材が1つだけのときは「〇〇があるんだぞい！」のように単体で話し、「と」で繋げないでください。"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        # JSONを取り出す
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return [], ""
        data = json.loads(raw[start:end])
        ingredients = data.get("ingredients", [])
        message = data.get("message", "")
        return ingredients, message
    except Exception:
        return [], ""


# ────────────────────────────
# Groqセリフ生成（② 調理手順）
# ────────────────────────────
def groq_cooking_steps(recipe: dict, user_input_words: list) -> str:
    """
    調理手順セリフをGroqで生成する（代替食材名で話す）。
    user_input_words: Groqが正規化したユーザーの入力食材リスト（ChromaDB検索結果ではない）
    戻り値: セリフ文字列（失敗時は空文字列）
    """
    try:
        client = get_groq_client()

        # 食材マッピングを作る（本物の食材 → ユーザーが持っている食材）
        # カテゴリが一致する食材を優先して代替に割り当てる
        real_ingredients_list = recipe["本物の食材"]  # 順序を保持するためlistで扱う
        user_names = user_input_words  # Groq正規化リストを使う
        ingredient_map = get_ingredient_map()  # 食材名→カテゴリの辞書

        # ユーザー食材のカテゴリを取得
        user_categories = {name: ingredient_map.get(name, []) for name in user_names}

        # 代替候補（本物にない食材）
        substitutes = [n for n in user_names if n not in real_ingredients_list]

        mapping = {}
        used_substitutes = set()

        for real in real_ingredients_list:
            if real in user_names:
                mapping[real] = real  # 完全一致
            else:
                # 本物食材のカテゴリを取得
                real_cats = set(ingredient_map.get(real, []))
                # カテゴリが一致する代替食材を優先して探す
                best = None
                for sub in substitutes:
                    if sub in used_substitutes:
                        continue
                    sub_cats = set(user_categories.get(sub, []))
                    if real_cats & sub_cats:  # カテゴリが1つでも一致
                        best = sub
                        break
                if best is None:
                    # カテゴリ一致なし → 未使用の代替食材を順番に割り当て
                    for sub in substitutes:
                        if sub not in used_substitutes:
                            best = sub
                            break
                if best:
                    mapping[real] = f"{best}（代替）"
                    used_substitutes.add(best)
                else:
                    mapping[real] = real  # 代替なし→そのまま

        steps = recipe["加工手順"]
        cooking_method = recipe["必要調理法"]
        genre = recipe["ジャンル"]

        prompt = f"""あなたは「ゆるゆるコックさん」というキャラクターです。
語尾は「〜ぞい」「〜だぞい」「〜するぞい」を使い、全力肯定でやさしく話します。

以下の料理の作り方を、食材マッピングに基づいてユーザーの持っている食材名で話してください。

料理名：{recipe['name']}
ジャンル：{genre}
加工手順：{json.dumps(steps, ensure_ascii=False)}
調理法：{cooking_method}
食材マッピング（本物→ユーザーの食材）：{json.dumps(mapping, ensure_ascii=False)}

ルール：
- 代替食材は「（代替）」を取り除いて自然に話す
- 手順は2〜4文でざっくりまとめる
- 「これはおいしくなるぞい！」など応援の言葉を最後に入れる
- 200文字以内で簡潔に"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


# ────────────────────────────
# Groqセリフ生成（③ お見送り）
# ────────────────────────────
def groq_farewell(recipe: dict) -> str:
    """
    お見送りセリフをGroqで生成する（本物の食材で話す）。
    戻り値: セリフ文字列（失敗時は空文字列）
    """
    try:
        client = get_groq_client()

        real_ingredients = recipe["本物の食材"]
        description = recipe["説明文"]

        prompt = f"""あなたは「ゆるゆるコックさん」というキャラクターです。
語尾は「〜ぞい」「〜だぞい」「〜するぞい」を使い、全力肯定でやさしくお見送りします。

料理名：{recipe['name']}
本物の食材：{json.dumps(real_ingredients, ensure_ascii=False)}
説明文：{description}

上記を参考に、料理の魅力を伝えながら「またいつでも来てほしいぞい」という気持ちのお見送りセリフを100文字以内で書いてください。
注意：これはまだ「作り方を提案した段階」です。「おいしかった」「食べた」などの過去形は使わず、「きっとおいしいぞい」「得意料理になるぞい」「また来てほしいぞい」のような未来・期待のニュアンスにしてください。
セリフだけを返してください。"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


# ────────────────────────────
# ingredient_db読み込み（カテゴリ検索用）
# ────────────────────────────
@st.cache_resource
def get_ingredient_map() -> dict:
    """食材名→カテゴリの辞書を返す（Groq正規化リストのカテゴリ引き用）"""
    with open("./data/ingredient_db.json", encoding="utf-8") as f:
        db = json.load(f)
    return {item["食材名"]: item["カテゴリ"] for item in db}


def get_categories_from_words(words: list, ingredient_map: dict) -> list:
    """Groq正規化リストの食材名からカテゴリを取得する"""
    categories = []
    for word in words:
        cats = ingredient_map.get(word, [])
        for cat in cats:
            if cat not in categories:
                categories.append(cat)
    return categories


# ────────────────────────────
# ChromaDB登録用ドキュメント生成（setup_chroma.pyと同一ロジック）
# ────────────────────────────
def _build_recipe_document(recipe: dict) -> str:
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


def _build_ingredient_document(ingredient: dict) -> str:
    """食材DBの1件をベクトル検索用のテキストに変換する（食材名3回で表記ゆれ対策）"""
    categories = "、".join(ingredient["カテゴリ"])
    raw = "生食可" if ingredient["生食可"] else "加熱必要"
    name = ingredient["食材名"]
    name_emphasis = f"{name} {name} {name}。"
    return (
        f"{name_emphasis}"
        f"カテゴリ：{categories}。"
        f"{raw}。"
        f"{ingredient['説明']}"
    )


# ────────────────────────────
# ChromaDB接続（なければJSONから自動構築）
# ────────────────────────────
@st.cache_resource
def get_collections():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    existing = [c.name for c in client.list_collections()]

    # ── レシピコレクション ──
    if RECIPE_COLLECTION not in existing:
        with st.spinner("レシピDBを準備中だぞい…（初回だけ少し時間がかかるぞい）"):
            recipe_col = client.create_collection(
                name=RECIPE_COLLECTION,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            with open("./data/recipe_db.json", encoding="utf-8") as f:
                recipes = json.load(f)
            ids, docs, metas = [], [], []
            for i, r in enumerate(recipes):
                ids.append(f"recipe_{i:03d}")
                docs.append(_build_recipe_document(r))
                metas.append({
                    "name": r["name"],
                    "ジャンル": r["ジャンル"],
                    "必要調理法": r["必要調理法"],
                    "加熱": str(r["加熱"]),
                    "本物の食材": json.dumps(r["本物の食材"], ensure_ascii=False),
                    "使える食材カテゴリ": json.dumps(r["使える食材カテゴリ"], ensure_ascii=False),
                    "加工手順": json.dumps(r.get("加工手順", []), ensure_ascii=False),
                    "説明文": r["説明文"],
                })
            recipe_col.add(ids=ids, documents=docs, metadatas=metas)
    else:
        recipe_col = client.get_collection(
            name=RECIPE_COLLECTION, embedding_function=embed_fn
        )

    # ── 食材コレクション ──
    if INGREDIENT_COLLECTION not in existing:
        with st.spinner("食材DBを準備中だぞい…"):
            ingredient_col = client.create_collection(
                name=INGREDIENT_COLLECTION,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            with open("./data/ingredient_db.json", encoding="utf-8") as f:
                ingredients = json.load(f)
            ids, docs, metas = [], [], []
            for i, item in enumerate(ingredients):
                ids.append(f"ingredient_{i:03d}")
                docs.append(_build_ingredient_document(item))
                metas.append({
                    "食材名": item["食材名"],
                    "カテゴリ": json.dumps(item["カテゴリ"], ensure_ascii=False),
                    "生食可": str(item["生食可"]),
                    "説明": item["説明"],
                })
            ingredient_col.add(ids=ids, documents=docs, metadatas=metas)
    else:
        ingredient_col = client.get_collection(
            name=INGREDIENT_COLLECTION, embedding_function=embed_fn
        )

    return recipe_col, ingredient_col


# ────────────────────────────
# 検索ロジック
# ────────────────────────────
def search_one_ingredient(ingredient_col, word: str) -> dict | None:
    """1単語で食材を1件検索する"""
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


def search_recipes(recipe_col, categories: list, tools: list,
                   temperature: str, exclude_names: list, n=5) -> list:
    """カテゴリ・道具・温度で料理を検索する"""
    query = "、".join(categories) + "を使った料理"
    results = recipe_col.query(query_texts=[query], n_results=20)

    hits = []
    for i, meta in enumerate(results["metadatas"][0]):
        name = meta["name"]

        if name in exclude_names:
            continue

        is_heated = meta["加熱"] == "True"
        if temperature == "あったかいのがいい" and not is_heated:
            continue

        cooking_method = meta["必要調理法"]
        needs_stove = cooking_method in ["炒め", "炒め煮", "煮る", "煮込み", "焼き", "茹でる", "炊く"]
        needs_microwave = False

        has_stove = "コンロ" in tools
        has_microwave = "電子レンジ" in tools
        no_heat_needed = cooking_method == "なし"

        # ゆるゆるコックさん：道具がなくても除外しない（誰かの力を借りればOK）
        # 加熱不要な料理はいつでもOK。加熱必要な料理も道具の有無に関係なく提案する。

        recipe_categories = json.loads(meta["使える食材カテゴリ"])
        match_count = len(set(categories) & set(recipe_categories))

        # 道具なし = コンロもレンジもない かつ 加熱が必要な料理
        no_tools = not has_stove and not has_microwave
        needs_heat = needs_stove and not no_heat_needed
        uses_microwave_instead = needs_stove and not has_stove and has_microwave

        hits.append({
            "name": name,
            "ジャンル": meta["ジャンル"],
            "必要調理法": cooking_method,
            "加熱": is_heated,
            "本物の食材": json.loads(meta["本物の食材"]),
            "使える食材カテゴリ": recipe_categories,
            "加工手順": json.loads(meta["加工手順"]),
            "説明文": meta["説明文"],
            "一致カテゴリ数": match_count,
            "道具なし": no_tools and needs_heat,          # コンロもレンジもない＋加熱必要
            "レンジ代用": uses_microwave_instead,          # レンジでコンロを代用
            "距離": round(results["distances"][0][i], 4),
        })

    # カテゴリ一致が1件もない料理は除外（ベクトル類似度だけで引っかかるのを防ぐ）
    hits = [h for h in hits if h["一致カテゴリ数"] > 0]
    hits.sort(key=lambda x: (-x["一致カテゴリ数"], x["距離"]))
    return hits[:n]


def calc_match_rate(recipe: dict, found_ingredients: list,
                    user_input_words: list = None) -> int:
    """一致率を計算する（食材80点＋調理法20点）
    user_input_words: Groq正規化リスト。あればこちらを優先して一致率計算に使う。
    """
    real_ingredients = set(recipe["本物の食材"])
    # Groq正規化リストがあればそちらを使う（ChromaDB混入を防ぐ）
    if user_input_words:
        found_names = set(user_input_words)
    else:
        found_names = set(ing["食材名"] for ing in found_ingredients)

    if real_ingredients:
        matched = sum(1 for ri in real_ingredients if ri in found_names)
        ingredient_score = int((matched / len(real_ingredients)) * 80)
    else:
        ingredient_score = 0

    # ゆるゆるコックさん：道具なしでも気持ちを応援するので調理点は常に20点
    cooking_score = 20

    return min(ingredient_score + cooking_score, 100)


def get_match_prefix(rate: int) -> str:
    """一致率に応じた前置きを返す"""
    for threshold in sorted(MATCH_PREFIXES.keys(), reverse=True):
        if rate >= threshold:
            return MATCH_PREFIXES[threshold]
    return MATCH_PREFIXES[0]


def build_recipe_name(recipe: dict, found_ingredients: list,
                      user_input_words: list = None) -> str:
    """命名を生成する（前置き＋料理名ぽいのん＋代替食材）
    user_input_words: Groqが正規化したユーザーの入力食材リスト（代替判定に使う）
    """
    rate = calc_match_rate(recipe, found_ingredients, user_input_words=user_input_words)
    prefix = get_match_prefix(rate)

    real_ingredients = set(recipe["本物の食材"])

    # Groq正規化リストがあればそちらを優先（ChromaDB検索結果より正確）
    if user_input_words:
        user_names = user_input_words
    else:
        user_names = [ing["食材名"] for ing in found_ingredients]

    substitutes = [name for name in user_names if name not in real_ingredients]

    if substitutes:
        if len(substitutes) == 1:
            suffix = f"（{substitutes[0]}入り）"
        else:
            suffix = f"（{'と'.join(substitutes[:2])}入り）"
    else:
        suffix = ""

    return f"{prefix}{recipe['name']}ぽいのん{suffix}", rate


# ────────────────────────────
# セッション初期化
# ────────────────────────────
def init_session():
    defaults = {
        "screen": "top",
        "user_input": "",
        "temperature": "どっちでもいい",
        "tools": [],
        "found_ingredients": [],
        "found_categories": [],
        "groq_normalized_words": [],   # Groqが正規化した食材名リスト（命名・詳細で使う）
        "selected_recipe": None,
        "recipe_name": "",
        "match_rate": 0,
        "last_recipes": [],
        "groq_analyze_message": "",    # ① 食材解析セリフ（Groq）
        "groq_cooking_message": "",    # ② 調理手順セリフ（Groq）
        "groq_farewell_message": "",   # ③ お見送りセリフ（Groq）
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ────────────────────────────
# 画面①：トップ
# ────────────────────────────
def show_top():
    st.title("ゆるゆるコックさん 🍳")
    st.write("手元の食材を教えてくれたら、何か作れるか考えるぞい！")
    st.divider()

    user_input = st.text_area(
        "今ある食材を教えてほしいぞい",
        placeholder="例：卵、ご飯、ねぎ、残り物のハム",
        height=120,
    )

    st.write("**温度はどうするぞい？**")
    temperature = st.radio(
        label="温度",
        options=["あったかいのがいい", "どっちでもいい"],
        index=1,
        label_visibility="collapsed",
        horizontal=True,
    )

    st.write("**使える道具はあるかぞい？**")
    col1, col2 = st.columns(2)
    with col1:
        has_stove = st.checkbox("コンロ")
    with col2:
        has_microwave = st.checkbox("電子レンジ")

    tools = []
    if has_stove:
        tools.append("コンロ")
    if has_microwave:
        tools.append("電子レンジ")

    st.divider()

    button_disabled = not user_input.strip()
    if st.button(
        "コックさんに相談するぞい 🍳",
        use_container_width=True,
        type="primary",
        disabled=button_disabled,
    ):
        recipe_col, ingredient_col = get_collections()

        # ─── Groqで食材を正規化 ───
        with st.spinner("食材を解析中だぞい…"):
            normalized_words, analyze_message = groq_normalize_ingredients(user_input)

        # Groq成功 → 正規化リストを使う / 失敗 → 従来方式にフォールバック
        if normalized_words:
            words_for_search = normalized_words
        else:
            words_for_search = [
                w.strip()
                for w in user_input.replace("、", " ").replace(",", " ").split()
                if w.strip()
            ]

        # ─── ChromaDBで食材検索 ───
        found_ingredients = []
        for word in words_for_search:
            hit = search_one_ingredient(ingredient_col, word)
            if hit and hit["食材名"] not in [f["食材名"] for f in found_ingredients]:
                found_ingredients.append(hit)

        # ─── カテゴリ取得（Groq正規化リスト優先・失敗時はChromaDB結果で代替）───
        if normalized_words:
            # Groq正規化リストからingredient_dbを直接引いてカテゴリを取得
            # → ChromaDBのベクトル検索による誤カテゴリ混入を防ぐ
            ingredient_map = get_ingredient_map()
            found_categories = get_categories_from_words(normalized_words, ingredient_map)
            if not found_categories:
                # ingredient_dbにない食材ばかりの場合はChromaDB結果にフォールバック
                found_categories = []
                for ing in found_ingredients:
                    for cat in ing["カテゴリ"]:
                        if cat not in found_categories:
                            found_categories.append(cat)
        else:
            # Groq失敗時はChromaDB検索結果からカテゴリを取得
            found_categories = []
            for ing in found_ingredients:
                for cat in ing["カテゴリ"]:
                    if cat not in found_categories:
                        found_categories.append(cat)

        # ─── 料理検索 ───
        if found_categories:
            recipes = search_recipes(
                recipe_col, found_categories, tools, temperature,
                exclude_names=st.session_state.last_recipes
            )
        else:
            recipes = []

        # ─── セッションに保存 ───
        st.session_state.user_input = user_input
        st.session_state.temperature = temperature
        st.session_state.tools = tools
        st.session_state.found_ingredients = found_ingredients
        st.session_state.found_categories = found_categories
        st.session_state.groq_normalized_words = normalized_words  # Groq正規化リストを保存
        st.session_state.groq_analyze_message = analyze_message

        if recipes:
            top_recipes = recipes[:3]
            selected = random.choice(top_recipes)
            recipe_name, match_rate = build_recipe_name(selected, found_ingredients, user_input_words=normalized_words)
            st.session_state.selected_recipe = selected
            st.session_state.recipe_name = recipe_name
            st.session_state.match_rate = match_rate
            st.session_state.last_recipes = [selected["name"]]
            st.session_state.screen = "analyze"
        else:
            st.session_state.selected_recipe = None
            st.session_state.screen = "analyze_rescue"

        st.rerun()


# ────────────────────────────
# 画面②-a：解析＋命名（成功版）
# ────────────────────────────
def show_analyze():
    recipe = st.session_state.selected_recipe
    recipe_name = st.session_state.recipe_name
    match_rate = st.session_state.match_rate
    found_ingredients = st.session_state.found_ingredients
    analyze_message = st.session_state.groq_analyze_message

    st.title("ゆるゆるコックさん 🍳")

    # ─── 食材解析セリフ（Groqあり→Groq / なし→テンプレ）───
    if analyze_message:
        st.write(analyze_message)
    else:
        found_names = "と".join([ing["食材名"] for ing in found_ingredients]) if found_ingredients else "いろいろ"
        st.write(f"「{found_names}」があるんだぞい。ちょっと考えてみるぞい…")

    st.divider()

    # ─── 命名 ───
    st.subheader(f"{recipe_name}が作れそうだぞい！")
    # 道具の状況に応じておせっかいな一言を追加
    if recipe.get("道具なし"):
        st.caption("（加熱器具がないぞい。誰かにレンチンとかさせてもらうんだぞい。生はダメだぞい！）")
    elif recipe.get("レンジ代用"):
        st.caption("（レンジでなんとかするぞい！）")

    # ─── 一致率メーター ───
    st.write("**一致率**")
    st.progress(match_rate / 100)
    st.caption(f"{match_rate}% ー 調理しようとした気持ちも込みだぞい！")

    st.divider()

    if st.button("詳しく教えてほしいぞい →", use_container_width=True, type="primary"):
        # 詳細画面に進む前に調理手順セリフをGroqで生成
        with st.spinner("作り方を考え中だぞい…"):
            cooking_message = groq_cooking_steps(recipe, st.session_state.get("groq_normalized_words", []))
        st.session_state.groq_cooking_message = cooking_message
        st.session_state.screen = "detail"
        st.rerun()


# ────────────────────────────
# 画面②-a：解析＋命名（救済版）
# ────────────────────────────
def show_analyze_rescue():
    st.title("ゆるゆるコックさん 🍳")

    st.write("うーん、いいのが思い浮かばなかったぞい。ごめんなさい。")

    st.divider()

    advice = random.choice(SHOPPING_ADVICE)
    st.info(f"💡 {advice}")

    st.divider()

    if st.button("次へ →", use_container_width=True):
        st.session_state.screen = "farewell_rescue"
        st.rerun()


# ────────────────────────────
# 画面②-b：詳細説明
# ────────────────────────────
def show_detail():
    recipe = st.session_state.selected_recipe
    found_ingredients = st.session_state.found_ingredients
    recipe_name = st.session_state.recipe_name
    cooking_message = st.session_state.groq_cooking_message
    # Groq正規化リストがあればそちらを使う（ChromaDB検索混入防止）
    groq_words = st.session_state.get("groq_normalized_words", [])

    st.title("ゆるゆるコックさん 🍳")

    # ─── 食材の仕分けセリフ ───
    real_ingredients = set(recipe["本物の食材"])

    # ユーザーが実際に入力した食材：Groq正規化リストを優先
    if groq_words:
        user_names = set(groq_words)
    else:
        user_names = set(ing["食材名"] for ing in found_ingredients)

    missing = real_ingredients - user_names
    substitutes = user_names - real_ingredients
    unused_candidates = user_names - real_ingredients - substitutes  # 通常は空

    if missing:
        missing_str = "と".join(missing)
        st.write(f"本物は{missing_str}が入るらしいけど、これもきっとおいしいぞい！")

    if substitutes:
        sub_str = "と".join(substitutes)
        st.write(f"{sub_str}は今回の{recipe['name']}には入らないやつだけど、いい仕事してくれるぞい！")

    st.divider()

    # ─── 調理手順セリフ（Groqあり→Groq / なし→テンプレ）───
    st.write("**作り方（ざっくり）**")
    if cooking_message:
        st.write(cooking_message)
    else:
        if recipe["加工手順"]:
            steps_str = "、".join(recipe["加工手順"])
            cooking = recipe["必要調理法"]
            st.write(f"{steps_str}して、{cooking}したらできるぞい！")

    st.divider()

    # ─── 調味料・食べ方ヒント ───
    genre = recipe["ジャンル"]
    st.write("**調味料はこんな感じだぞい**")
    st.write(SEASONING_HINTS.get(genre, "手元にあるやつ入れたらいいぞい"))

    st.write("**食べ方のヒントだぞい**")
    # 主食系食材（ご飯・麺・パンなど）が既に入ってる料理は「ご飯と一緒に」を言わない
    found_categories = st.session_state.get("found_categories", [])
    recipe_categories = recipe.get("使える食材カテゴリ", [])
    has_staple = "主食系" in recipe_categories  # レシピ自体に主食系が含まれるか
    default_eating_hint = EATING_HINTS.get(genre, "好きなように食べるといいぞい")
    if has_staple:
        # 主食系が入ってる料理はそのまま食べるのを推奨
        eating_hint = "これだけで立派な一食になるぞい！お好みで汁物を添えるといいぞい"
    else:
        eating_hint = default_eating_hint
    st.write(eating_hint)

    st.divider()

    st.write("よかったよかったぞい 🎉")

    if st.button("次へ →", use_container_width=True, type="primary"):
        # お見送り画面に進む前にセリフをGroqで生成
        with st.spinner("お見送りの言葉を考え中だぞい…"):
            farewell_message = groq_farewell(recipe)
        st.session_state.groq_farewell_message = farewell_message
        st.session_state.screen = "farewell"
        st.rerun()


# ────────────────────────────
# 画面③：お見送り（成功版）
# ────────────────────────────
def show_farewell():
    recipe_name = st.session_state.recipe_name
    farewell_message = st.session_state.groq_farewell_message
    cooking_message = st.session_state.groq_cooking_message

    st.title("ゆるゆるコックさん 🍳")

    # ─── お見送りセリフ（Groqあり→Groq / なし→テンプレ）───
    if farewell_message:
        st.write(farewell_message)
    else:
        st.write("また、何か作りたくなったら来るといいぞい 🍳")

    st.divider()

    # ─── シェア用テキスト（調理手順も含める）───
    if cooking_message:
        share_text = f"ゆるゆるコックさんに「{recipe_name}」の作り方を教えてもらったぞい\n\n【作り方】\n{cooking_message}"
    else:
        share_text = f"ゆるゆるコックさんに「{recipe_name}」の作り方を教えてもらったぞい"

    st.write("**作った料理をシェアするぞい📋**")
    st.code(share_text, language=None)

    # JavaScriptに渡すためにエスケープ処理
    share_text_js = share_text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    copy_js = f"""
        <button onclick="navigator.clipboard.writeText('{share_text_js}').then(() => {{
            this.textContent = 'コピーできたぞい ✅';
            setTimeout(() => this.textContent = 'コピーするぞい 📋', 2000);
        }})"
        style="
            background-color: #ff6b6b;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        ">コピーするぞい 📋</button>
    """
    st.components.v1.html(copy_js, height=60)

    st.divider()

    if st.button("トップに戻るぞい", use_container_width=True):
        for key in ["screen", "user_input", "temperature", "tools",
                    "found_ingredients", "found_categories",
                    "selected_recipe", "recipe_name", "match_rate",
                    "groq_analyze_message", "groq_cooking_message", "groq_farewell_message"]:
            del st.session_state[key]
        st.rerun()


# ────────────────────────────
# 画面③：お見送り（救済版）
# ────────────────────────────
def show_farewell_rescue():
    st.title("ゆるゆるコックさん 🍳")

    st.write("また、何か作りたくなったら来るといいぞい 🍳")
    st.write("次は何かおいしいもの見つかるといいぞい！")

    st.divider()

    if st.button("トップに戻るぞい", use_container_width=True):
        for key in ["screen", "user_input", "temperature", "tools",
                    "found_ingredients", "found_categories",
                    "selected_recipe", "recipe_name", "match_rate",
                    "groq_analyze_message", "groq_cooking_message", "groq_farewell_message"]:
            del st.session_state[key]
        st.rerun()


# ────────────────────────────
# 画面ルーティング
# ────────────────────────────
init_session()

screen = st.session_state.screen

if screen == "top":
    show_top()
elif screen == "analyze":
    show_analyze()
elif screen == "analyze_rescue":
    show_analyze_rescue()
elif screen == "detail":
    show_detail()
elif screen == "farewell":
    show_farewell()
elif screen == "farewell_rescue":
    show_farewell_rescue()
