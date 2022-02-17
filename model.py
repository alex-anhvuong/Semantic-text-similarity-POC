from flask import current_app
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

from IPython.display import display


def run_model():
    modelPath = "./all-MiniLM-L6-v2"
    model = SentenceTransformer(modelPath)

    SENTENCE_TO_BE_COMPARED = current_app.config["TEXT"]

    # column_names = ["id", "title", "content", "title&content"]
    test_df = pd.read_csv(
        current_app.config["FILE_PATH"],
        index_col=False,
    )

    # Bert model
    test_df["title&content"] = test_df['title'] + '. ' + test_df['content']
    corpus = test_df["title&content"].values

    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    sim_scores = util.pytorch_cos_sim(
        model.encode(SENTENCE_TO_BE_COMPARED, convert_to_tensor=True), corpus_embeddings
    )[0].sort(descending=True)
    sim_scores_thresh = sim_scores[0][sim_scores[0] > 0.5]

    bert_df = test_df.iloc[sim_scores[1][: list(sim_scores_thresh.size())[0]].tolist()]
    bert_df["score"] = sim_scores_thresh

    # Tf-idf model

    vectorizer = TfidfVectorizer()
    tfidf_embedding = vectorizer.fit_transform(
        np.insert(corpus, 0, SENTENCE_TO_BE_COMPARED, axis=0)
    )
    (tfidf_embedding)

    top_5_scores = -np.sort(-cosine_similarity(tfidf_embedding[0:1], tfidf_embedding),)[
        :, :6
    ][0]
    top_5_scores = np.delete(top_5_scores, 0)
    top_5_id = np.argsort(-cosine_similarity(tfidf_embedding[0:1], tfidf_embedding),)[
        :, :6
    ][0]
    top_5_id = np.delete(top_5_id, 0)

    tfidf_df = test_df.iloc[top_5_id.tolist()]
    tfidf_df["score"] = top_5_scores

    bert_html = [
        bert_df.style.set_table_attributes(current_app.config["TABLE_STYLE"]).to_html(
            classes="data"
        )
    ]
    tfidf_html = [
        tfidf_df.style.set_table_attributes(current_app.config["TABLE_STYLE"]).to_html(
            classes="data"
        )
    ]

    return (bert_html, tfidf_html)
