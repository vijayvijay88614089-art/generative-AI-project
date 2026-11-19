from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EXAMPLES = [
    ("What is the average revenue?", "number", "df['revenue'].mean()"),
    ("What is total sales?", "number", "df['sales'].sum()"),
    ("How many rows are in the table?", "number", "len(df)"),
    ("Which product has the highest revenue?", "category", "df.groupby('product')['revenue'].sum().idxmax()"),
    ("Show the top five products by sales", "list[category]", "df.groupby('product')['sales'].sum().nlargest(5).index.tolist()"),
    ("Is revenue greater than 100000?", "boolean", "bool(df['revenue'].sum() > 100000)"),
    ("How many values are missing in revenue?", "number", "int(df['revenue'].isna().sum())"),
]


def retrieve(question: str, answer_type: str, k: int = 2) -> list[dict]:
    candidates = [x for x in EXAMPLES if x[1] == answer_type]
    if not candidates:
        return []
    texts = [x[0] for x in candidates] + [question]
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(texts)
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    return [{"question": candidates[i][0], "type": candidates[i][1], "expression": candidates[i][2], "score": float(scores[i])}
            for i in scores.argsort()[::-1][:k] if scores[i] > 0]
