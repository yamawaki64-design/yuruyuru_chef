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
# CSS・背景・UI共通関数
# ────────────────────────────
import base64

def _get_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def apply_styles():
    """背景画像・全体CSS・タイトルバーCSSを適用する"""
    try:
        img_b64 = _get_base64_image("./assets/kawaii_kokkusan_background_napkin_1600x900.jpg")
        bg_css = f"url('data:image/jpeg;base64,{img_b64}')"
    except Exception:
        bg_css = "none"

    st.markdown(f"""
    <style>
    /* ── ライトモード強制（ダークモード無効化） ── */
    :root {{
        color-scheme: light !important;
    }}
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        color-scheme: light !important;
    }}

    /* ── 背景 ── */
    .stApp {{
        background-image: {bg_css};
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-color: #fdf6e3 !important;
    }}

    /* ── Streamlit組み込みヘッダー（ハンバーガーメニューバー）を隠す ── */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* ── メインコンテンツをタイトルバー分下げる ── */
    .main .block-container {{
        padding-top: 3.8rem !important;
        max-width: 680px;
    }}

    /* ── 固定タイトルバー ── */
    .yuru-titlebar {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 9999;
        background: rgba(255, 248, 225, 0.95);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-bottom: 2px solid #e8c97a;
        padding: 0.45rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 2px 8px rgba(180,140,40,0.13);
    }}
    .yuru-titlebar-icon {{
        font-size: 1.2rem;
        line-height: 1;
    }}
    .yuru-titlebar-text {{
        font-size: 1rem;
        font-weight: bold;
        color: #7a4f10 !important;
        letter-spacing: 0.04em;
    }}

    /* ── ふきだし（コックさんセリフ） ── */
    .yuru-bubble {{
        background: #fff8e1 !important;
        border: 2px solid #e8c97a;
        border-radius: 16px 16px 16px 4px;
        padding: 0.9rem 1.1rem 0.9rem 1.3rem;
        margin-bottom: 1rem;
        position: relative;
        color: #5c3d0e !important;
        font-size: 1rem;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(180,140,40,0.10);
    }}
    .yuru-bubble::before {{
        content: "🍳";
        position: absolute;
        top: -1.1rem;
        left: 0.6rem;
        font-size: 1.4rem;
    }}

    /* ── セクション見出し ── */
    .yuru-section-label {{
        font-size: 0.78rem;
        font-weight: bold;
        color: #a0700a !important;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
        padding-left: 0.1rem;
        display: block;
    }}

    /* ── 料理名ビッグテキスト ── */
    .yuru-recipe-name {{
        font-size: 1.35rem;
        font-weight: bold;
        color: #5c3d0e !important;
        line-height: 1.5;
        margin: 0.3rem 0 0.2rem 0;
    }}

    /* ── 道具注記 ── */
    .yuru-tool-note {{
        font-size: 0.85rem;
        color: #c0732a !important;
        margin-top: 0.2rem;
    }}

    /* ── スピナー（ローディング）枠を透明に ── */
    [data-testid="stSpinner"] > div,
    [data-testid="stSpinnerContainer"],
    div[class*="stSpinner"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    /* スピナーのテキスト色 */
    [data-testid="stSpinner"] p,
    [data-testid="stSpinner"] span {{
        color: #7a4f10 !important;
    }}

    /* ── ローディング全体の白背景を消す ── */
    .stStatusWidget, [data-testid="stStatusWidget"] {{
        background: transparent !important;
    }}

    /* ── st.container(border=True) のパネルスタイル上書き ── */
    /* stLayoutWrapperがborderコンテナの実体 */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, 0.84) !important;
        border: 1px solid rgba(232,201,122,0.7) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(160,120,30,0.10) !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] > div,
    [data-testid="stVerticalBlockBorderWrapper"] > div > div {{
        background: transparent !important;
    }}
    /* stLayoutWrapper内の直接の子borderも対象 */
    [data-testid="stLayoutWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, 0.84) !important;
    }}

    /* ── コードブロック（シェアテキスト）を明るく ── */
    .stCode, .stCode > div, [data-testid="stCode"],
    [data-testid="stCode"] > div,
    pre, pre > code {{
        background: rgba(255, 248, 225, 0.95) !important;
        color: #3d2600 !important;
        border: 1px solid #e8c97a !important;
        border-radius: 8px !important;
    }}
    /* コードブロック内のコピーボタン */ 
    [data-testid="stCode"] button {{
        color: #7a4f10 !important;
    }}

    /* ── 全テキスト要素の文字色（ダークモード上書き） ── */
    .stApp p, .stApp span, .stApp div,
    .stMarkdown, .stMarkdown p,
    [data-testid="stText"],
    [data-testid="stMarkdownContainer"] p {{
        color: #3d2600 !important;
    }}

    /* ── ラジオ・チェックボックス ── */
    .stRadio label, .stRadio span,
    .stCheckbox label, .stCheckbox span {{
        color: #5c3d0e !important;
    }}
    .stRadio [data-testid="stWidgetLabel"] p,
    .stCheckbox [data-testid="stWidgetLabel"] p {{
        color: #5c3d0e !important;
    }}

    /* ── テキストエリア ── */
    .stTextArea textarea {{
        background: rgba(255,255,255,0.90) !important;
        border: 1.5px solid #d4a84b !important;
        border-radius: 8px !important;
        color: #3d2600 !important;
    }}
    .stTextArea textarea::placeholder {{
        color: #b08040 !important;
        opacity: 1 !important;
    }}

    /* ── ボタン系 ── */
    .stButton > button[kind="primary"] {{
        background-color: #e8a020 !important;
        border-color: #e8a020 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #cf8c18 !important;
        border-color: #cf8c18 !important;
    }}
    .stButton > button:not([kind="primary"]) {{
        background-color: rgba(255,255,255,0.80) !important;
        border: 1.5px solid #d4a84b !important;
        color: #7a4f10 !important;
        border-radius: 10px !important;
    }}

    /* ── Streamlit標準h1を非表示（タイトルバーで代替） ── */
    h1 {{ display: none !important; }}

    /* ── プログレスバー色 ── */
    .stProgress > div > div > div > div {{
        background-color: #e8a020 !important;
    }}

    /* ── caption ── */
    .stCaptionContainer p, [data-testid="stCaptionContainer"] p {{
        color: #8a6020 !important;
    }}

    /* ── divider ── */
    hr {{
        border-color: rgba(232,201,122,0.4) !important;
    }}

    /* ── モバイル（iPhone SE対応） ── */
    @media (max-width: 420px) {{
        .main .block-container {{
            padding-top: 3.4rem !important;
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
        }}
        .yuru-titlebar {{
            padding: 0.35rem 0.8rem;
        }}
        .yuru-titlebar-text {{
            font-size: 0.9rem;
        }}
        .yuru-bubble {{
            font-size: 0.93rem;
            padding: 0.75rem 0.9rem;
        }}
        .yuru-recipe-name {{
            font-size: 1.15rem;
        }}
        .stButton > button {{
            font-size: 0.92rem !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


def show_titlebar(title: str):
    """固定タイトルバーを表示する"""
    st.markdown(f"""
    <div class="yuru-titlebar">
        <span class="yuru-titlebar-icon">🍳</span>
        <span class="yuru-titlebar-text">{title}</span>
    </div>
    """, unsafe_allow_html=True)
    # iframe内のスクロールをトップに戻す（Streamlit Cloud対応）
    st.components.v1.html("""
    <script>
        // 自分自身（iframe内）をスクロール
        window.scrollTo({top: 0, behavior: 'instant'});
        // Streamlitのメインコンテナを探してスクロール
        try {
            const main = window.parent.document.querySelector('[data-testid="stAppViewBlockContainer"]');
            if (main) main.scrollTop = 0;
            const appView = window.parent.document.querySelector('.main');
            if (appView) appView.scrollTop = 0;
            // ページ全体
            window.parent.document.documentElement.scrollTop = 0;
            window.parent.document.body.scrollTop = 0;
        } catch(e) {}
    </script>
    """, height=0)


def bubble(text: str):
    """コックさんのふきだしセリフを表示する"""
    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    st.markdown(f'<div class="yuru-bubble">{safe_text}</div>', unsafe_allow_html=True)


def section_label(text: str):
    """パネル内のセクション見出し（st.container内で使う）"""
    st.markdown(f'<span class="yuru-section-label">{text}</span>', unsafe_allow_html=True)


# panel_open / panel_close は廃止。各画面で with st.container(border=True): を使う。


def typing_animation(text: str, speed_ms: int = 30):
    """タイピングアニメーションでテキストを表示する（JavaScript）"""
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    uid = f"typing_{random.randint(10000, 99999)}"
    st.components.v1.html(f"""
    <div id="{uid}" style="
        font-size: 1rem;
        color: #5c3d0e;
        line-height: 1.7;
        white-space: pre-wrap;
        word-break: break-all;
        min-height: 1.5em;
        font-family: inherit;
    "></div>
    <script>
    (function() {{
        const el = document.getElementById('{uid}');
        const text = `{safe_text}`;
        let i = 0;
        function type() {{
            if (i < text.length) {{
                el.textContent += text[i++];
                setTimeout(type, {speed_ms});
            }}
        }}
        type();
    }})();
    </script>
    """, height=80)


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
- 料理名・メニュー名は食材に分解する（例：牛丼→牛肉・玉ねぎ・ご飯、から揚げ弁当→鶏肉・ご飯、ビッグマック→牛肉・パン・チーズ・野菜）
- コンビニ弁当・ファストフード・外食メニューなども同様に含まれる食材に分解する
- パン類（食パン・トースト・ロールパン・バゲットなど）は「パン」に統一する
- ご飯・冷ご飯・白米・米などは「ご飯」に統一する
- うどん・そば・ラーメン・パスタなど麺類は「〇〇」とそのまま正規化するが、総称で入力された場合は「麺」にする

返すJSONの形式（他のテキストは一切含めないこと）：
{{
  "ingredients": ["食材1", "食材2", "食材3"],
  "message": "○○と△△と□□があるんだぞい！ちょっと考えてみるぞい…"
}}

messageは「ゆるゆるコックさん」というキャラクターのセリフで、語尾は「〜ぞい」「〜だぞい」を使い、食材名を入れて元気よく書いてください。
食材が1つだけのときは「〇〇があるんだぞい！」のように単体で話し、「と」で繋げないでください。
必ず日本語のみで出力してください。"""

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
        return [], "groq_error"  # エラー時はフラグとして"groq_error"を返す


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
        # 主食系（ご飯・パン・麺類など）は他カテゴリの代替にはならないので除外する
        # カテゴリ未登録（ingredient_dbにない）食材も除外する
        substitutes = [
            n for n in user_names
            if n not in real_ingredients_list
            and "主食系" not in ingredient_map.get(n, [])
            and len(ingredient_map.get(n, [])) > 0
        ]

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

        # 加工手順の文字列をPython側で事前に置換する（Groqに任せると揺れるため）
        # 長い食材名から先に置換して部分一致の誤爆を防ぐ
        replaced_steps = list(steps)
        sorted_mapping = sorted(mapping.items(), key=lambda x: -len(x[0]))
        for i, step in enumerate(replaced_steps):
            for real, user_name in sorted_mapping:
                display_name = user_name.replace("（代替）", "")
                if display_name != real:
                    replaced_steps[i] = replaced_steps[i].replace(real, display_name)

        prompt = f"""あなたは「ゆるゆるコックさん」というキャラクターです。
語尾は「〜ぞい」「〜だぞい」「〜するぞい」を使い、全力肯定でやさしく話します。
必ず日本語のみで出力してください。他の言語（英語・韓国語・中国語など）を混ぜてはいけません。

以下の料理の作り方を、すでに食材名を置き換えた加工手順をベースにして話してください。

料理名：{recipe['name']}
ジャンル：{genre}
加工手順（置換済み）：{json.dumps(replaced_steps, ensure_ascii=False)}
調理法：{cooking_method}

ルール：
- 加工手順の食材名はそのまま使う（勝手に別の食材名に変えない）
- 手順は2〜4文でざっくりまとめる
- 「これはおいしくなるぞい！」など応援の言葉を最後に入れる
- 200文字以内で簡潔に
- 日本語のみ使用すること"""

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
セリフだけを返してください。必ず日本語のみで出力してください。"""

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
# スタイル適用（全画面共通・最初に1回）
# ────────────────────────────
apply_styles()

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
        "groq_error": False,           # Groqエラーフラグ
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ────────────────────────────
# 画面①：トップ
# ────────────────────────────
def show_top():
    show_titlebar("ゆるゆるコックさん")

    bubble("持ってるもの教えてくれたら、何が作れそうか考えるぞい！")

    with st.container(border=True):
        section_label("今ある食べ物")
        user_input = st.text_area(
            "使える食べ物を教えてほしいぞい",
            value=st.session_state.get("user_input", ""),
            placeholder="例：卵、冷ご飯とネギ、コンビニのから揚げ弁当 など",
            height=110,
            label_visibility="collapsed",
        )

        st.write("")
        section_label("あったかいのだけから探すか選べるぞい")
        temperature = st.radio(
            label="温度",
            options=["あったかいのがいい", "どっちでもいい"],
            index=1,
            label_visibility="collapsed",
            horizontal=True,
        )

        st.write("")
        section_label("使える道具も知りたいぞい")
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

    # ─── DB準備（初回のみspinnerを表示、2回目以降は即返る）───
    _db_ready = "db_initialized" in st.session_state
    if not _db_ready:
        with st.spinner("レシピDBを準備中だぞい…（初回だけ少し時間がかかるぞい）"):
            recipe_col, ingredient_col = get_collections()
        st.session_state.db_initialized = True
    else:
        recipe_col, ingredient_col = get_collections()

    button_disabled = not user_input.strip()
    if st.button(
        "コックさんに相談するぞい 🍳",
        use_container_width=True,
        type="primary",
        disabled=button_disabled,
    ):
        # recipe_col, ingredient_col はボタン上で取得済み（@cache_resourceで使い回し）

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
        st.session_state.groq_normalized_words = normalized_words
        st.session_state.groq_analyze_message = analyze_message
        st.session_state.groq_error = (analyze_message == "groq_error")  # エラーフラグ

        if recipes:
            top_recipes = recipes[:3]
            selected = random.choice(top_recipes)
            recipe_name, match_rate = build_recipe_name(selected, found_ingredients, user_input_words=normalized_words)
            st.session_state.selected_recipe = selected
            st.session_state.recipe_name = recipe_name
            st.session_state.match_rate = match_rate
            st.session_state.last_recipes = st.session_state.get("last_recipes", []) + [selected["name"]]
            st.session_state.screen = "analyze"
        else:
            st.session_state.selected_recipe = None
            st.session_state.screen = "analyze_rescue"

        st.rerun()


# ────────────────────────────
# 画面②-a：解析＋命名（成功版）
# ────────────────────────────
def show_analyze():
    show_titlebar("メニューを決めるぞい")

    recipe = st.session_state.selected_recipe
    recipe_name = st.session_state.recipe_name
    match_rate = st.session_state.match_rate
    found_ingredients = st.session_state.found_ingredients
    analyze_message = st.session_state.groq_analyze_message

    # ─── 食材解析セリフ（ふきだし）───
    if analyze_message:
        bubble(analyze_message)
    else:
        found_names = "と".join([ing["食材名"] for ing in found_ingredients]) if found_ingredients else "いろいろ"
        bubble(f"「{found_names}」があるんだぞい。ちょっと考えてみるぞい…")

    # ─── 命名＋一致率パネル ───
    with st.container(border=True):
        section_label("おすすめメニュー")
        st.markdown(f'<div class="yuru-recipe-name">✨ {recipe_name}が作れそうだぞい！</div>', unsafe_allow_html=True)
        if recipe.get("道具なし"):
            st.markdown('<div class="yuru-tool-note">⚠️ 加熱器具がないぞい。誰かにレンチンとかさせてもらうんだぞい。生はダメだぞい！</div>', unsafe_allow_html=True)
        elif recipe.get("レンジ代用"):
            st.markdown('<div class="yuru-tool-note">💡 レンジでなんとかするぞい！</div>', unsafe_allow_html=True)

        st.write("")
        section_label("食材一致率")

        # 一致率に応じてキャラクター表情・コメントを変える
        if match_rate >= 90:
            face = "🤩"
            face_comment = "完璧だぞい！！"
            bar_color = "#4caf50"
        elif match_rate >= 70:
            face = "😄"
            face_comment = "かなりいい感じだぞい！"
            bar_color = "#8bc34a"
        elif match_rate >= 50:
            face = "🙂"
            face_comment = "やりくり上手だぞい！"
            bar_color = "#e8a020"
        elif match_rate >= 30:
            face = "😅"
            face_comment = "言い切れば大丈夫だぞい！"
            bar_color = "#ff9800"
        else:
            face = "😬"
            face_comment = "オリジナルを生み出したぞい！"
            bar_color = "#f44336"

        bar_width = max(match_rate, 4)  # 0%でも少し見える
        st.markdown(f"""
        <div style="margin: 0.3rem 0 0.6rem 0;">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
                <span style="font-size:2rem; line-height:1;">{face}</span>
                <div>
                    <span style="font-size:1.5rem; font-weight:bold; color:{bar_color};">{match_rate}%</span>
                    <span style="font-size:0.85rem; color:#8a6020; margin-left:0.4rem;">{face_comment}</span>
                </div>
            </div>
            <div style="
                background: rgba(200,180,130,0.2);
                border-radius: 999px;
                height: 10px;
                overflow: hidden;
            ">
                <div style="
                    width: {bar_width}%;
                    height: 100%;
                    background: linear-gradient(90deg, {bar_color}cc, {bar_color});
                    border-radius: 999px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <div style="font-size:0.78rem; color:#a08040; margin-top:0.3rem;">
                調理しようとした気持ちも込みだぞい！
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("作り方を説明するぞい →", use_container_width=True, type="primary"):
        with st.spinner("作り方を考え中だぞい…"):
            cooking_message = groq_cooking_steps(recipe, st.session_state.get("groq_normalized_words", []))
        st.session_state.groq_cooking_message = cooking_message
        st.session_state.screen = "detail"
        st.rerun()


# ────────────────────────────
# 画面②-a：解析＋命名（救済版）
# ────────────────────────────
def show_analyze_rescue():
    groq_error = st.session_state.get("groq_error", False)

    if groq_error:
        show_titlebar("ちょっと待ってほしいぞい")
        bubble("うーん、ちょっと頭が混乱してるぞい。\nすこし待ってから、もう一回試してほしいぞい！🙏")
    else:
        show_titlebar("降参だぞい")
        bubble("うーん、その食材からはいいのが思い浮かばなかったぞい。\n買い物のアドバイスもするぞい！")

    with st.container(border=True):
        section_label("買い物アドバイスだぞい 💡")
        advice = random.choice(SHOPPING_ADVICE)
        st.write(advice)

    if st.button("次へ →", use_container_width=True):
        st.session_state.screen = "farewell_rescue"
        st.rerun()


# ────────────────────────────
# 画面②-b：詳細説明
# ────────────────────────────
def show_detail():
    show_titlebar("作り方を教えるぞい")

    recipe = st.session_state.selected_recipe
    found_ingredients = st.session_state.found_ingredients
    recipe_name = st.session_state.recipe_name
    cooking_message = st.session_state.groq_cooking_message
    groq_words = st.session_state.get("groq_normalized_words", [])

    # ─── 食材の仕分け ───
    real_ingredients = set(recipe["本物の食材"])
    if groq_words:
        user_names = set(groq_words)
    else:
        user_names = set(ing["食材名"] for ing in found_ingredients)

    missing = real_ingredients - user_names
    substitutes = user_names - real_ingredients

    # ふきだし：食材の仕分けセリフ
    if missing and substitutes:
        missing_str = "と".join(missing)
        sub_str = "と".join(substitutes)
        bubble(f"本物は{missing_str}が入るらしいけど、{sub_str}がいい仕事してくれるぞい！")
    elif missing:
        missing_str = "と".join(missing)
        bubble(f"本物は{missing_str}が入るらしいけど、これもきっとおいしいぞい！")
    elif substitutes:
        sub_str = "と".join(substitutes)
        bubble(f"{sub_str}は{recipe['name']}でも、いい味だしてくれるはずだぞい！")
    else:
        bubble(f"ばっちりな食材が揃ってるぞい！最高だぞい！")

    # ─── 作り方パネル ───
    with st.container(border=True):
        section_label("作り方（ざっくり）")
        if cooking_message:
            st.write(cooking_message)
        else:
            if recipe["加工手順"]:
                steps_str = "、".join(recipe["加工手順"])
                cooking = recipe["必要調理法"]
                st.write(f"{steps_str}して、{cooking}したらできるぞい！")

        st.divider()

        genre = recipe["ジャンル"]
        section_label("調味料のヒント")
        st.write(SEASONING_HINTS.get(genre, "手元にあるやつ入れたらいいぞい"))

        st.divider()

        section_label("食べ方のヒント")
        found_categories = st.session_state.get("found_categories", [])
        recipe_categories = recipe.get("使える食材カテゴリ", [])
        has_staple = "主食系" in recipe_categories
        if has_staple:
            eating_hint = "これだけで立派な一食になるぞい！お好みで汁物を添えるといいぞい"
        else:
            eating_hint = EATING_HINTS.get(genre, "好きなように食べるといいぞい")
        st.write(eating_hint)

    bubble("よかったよかった。これでおなかいっぱいになるぞい 🎉")

    if st.button("次へ →", use_container_width=True, type="primary"):
        with st.spinner("お見送りの言葉を考え中だぞい…"):
            farewell_message = groq_farewell(recipe)
        st.session_state.groq_farewell_message = farewell_message
        st.session_state.screen = "farewell"
        st.rerun()


# ────────────────────────────
# 画面③：お見送り（成功版）
# ────────────────────────────
def show_farewell():
    show_titlebar("お見送り")

    recipe_name = st.session_state.recipe_name
    farewell_message = st.session_state.groq_farewell_message
    cooking_message = st.session_state.groq_cooking_message

    # ─── お見送りセリフ（ふきだし）───
    if farewell_message:
        bubble(farewell_message)
    else:
        bubble("また、何か作りたくなったら来るといいぞい 🍳")

    # st.info("💡 「トップに戻るぞい」で同じ食材のまま別のメニューを相談できるぞい！")

    # ─── シェアパネル ───
    APP_URL = "https://yuruyuruchef.streamlit.app/"
    with st.container(border=True):
        section_label("作った料理をシェアするぞい 📋")
        if cooking_message:
            share_text = f"ゆるゆるコックさんに「{recipe_name}」の作り方を教えてもらったぞい\n\n【作り方】\n{cooking_message}\n\n🍳 {APP_URL}"
        else:
            share_text = f"ゆるゆるコックさんに「{recipe_name}」の作り方を教えてもらったぞい\n\n🍳 {APP_URL}"

        # st.codeの代わりにHTMLで折り返し対応のプレビューボックスを表示
        safe_preview = share_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        st.markdown(f"""
        <div style="
            background: rgba(255,248,225,0.95);
            border: 1px solid #e8c97a;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 0.88rem;
            color: #3d2600;
            line-height: 1.7;
            word-break: break-word;
            overflow-wrap: break-word;
            white-space: pre-wrap;
            margin-bottom: 0.5rem;
        ">{safe_preview}</div>
        """, unsafe_allow_html=True)

        share_text_js = share_text.replace("\\", "\\\\").replace("'", "\\'").replace("`", "\\`").replace("\n", "\\n")
        copy_js = f"""
            <button onclick="navigator.clipboard.writeText('{share_text_js}').then(() => {{
                this.textContent = 'コピーできたぞい ✅';
                setTimeout(() => this.textContent = 'コピーするぞい 📋', 2000);
            }})"
            style="
                background-color: #e8a020;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
                cursor: pointer;
                width: 100%;
            ">コピーするぞい 📋</button>
        """
        st.components.v1.html(copy_js, height=60)

    st.markdown("""
    <div style="
        background: rgba(255,248,225,0.85);
        border: 1px solid #e8c97a;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #7a5c20;
        line-height: 1.7;
        margin: 0.8rem 0;
    ">
        💡 同じ食材のままもう一度「相談する」で、別のメニューも考えられるぞい！<br>
        お好みのメニューが出てこない時は「きっちりコックさん」たち（
        <a href="https://cookpad.com" target="_blank" style="color:#c07020; font-weight:bold;">クックパッド</a>や
        <a href="https://delishkitchen.tv" target="_blank" style="color:#c07020; font-weight:bold;">デリッシュキッチン</a>）
        に相談するといいぞい！
    </div>
    """, unsafe_allow_html=True)

    if st.button("トップに戻るぞい", use_container_width=True):
        for key in ["screen", "temperature", "tools",
                    "found_ingredients", "found_categories",
                    "selected_recipe", "recipe_name", "match_rate",
                    "groq_analyze_message", "groq_cooking_message", "groq_farewell_message",
                    "groq_error"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ────────────────────────────
# 画面③：お見送り（救済版）
# ────────────────────────────
def show_farewell_rescue():
    show_titlebar("お見送り")

    bubble("また、何か作りたくなったら来るといいぞい 🍳\n次は何かおいしいもの見つかるといいぞい！")

    st.markdown("""
    <div style="
        background: rgba(255,248,225,0.85);
        border: 1px solid #e8c97a;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #7a5c20;
        line-height: 1.7;
        margin: 0.8rem 0;
    ">
        お好みのメニューが出てこない時は「きっちりコックさん」たち（
        <a href="https://cookpad.com" target="_blank" style="color:#c07020; font-weight:bold;">クックパッド</a>や
        <a href="https://delishkitchen.tv" target="_blank" style="color:#c07020; font-weight:bold;">デリッシュキッチン</a>）
        に相談するといいぞい！
    </div>
    """, unsafe_allow_html=True)

    if st.button("トップに戻るぞい", use_container_width=True):
        for key in ["screen", "temperature", "tools",
                    "found_ingredients", "found_categories",
                    "selected_recipe", "recipe_name", "match_rate",
                    "groq_analyze_message", "groq_cooking_message", "groq_farewell_message",
                    "groq_error"]:
            if key in st.session_state:
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
