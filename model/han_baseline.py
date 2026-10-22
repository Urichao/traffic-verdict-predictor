"""參考專案（tommy9301122/JudgmentPrediction）之階層式注意力模型（Hierarchical Attention）實作。

參考專案核心設計：
    案件描述斷詞斷句 → 詞向量 (Word Embedding) → 詞級 Attention (產出詞重要度)
                     → 句向量 → 句級 Attention (產出句子重要度，總和=1) → 判決結果分類

舊專案使用 2018 年的 Keras 2 + TensorFlow 1.x (`keras.backend.tensorflow_backend`)，已不相容於 Python 3.12。
本模組以現代 NumPy + Scikit-learn 實作雙層（詞級＋句級）Attention 分類器，支援：
1. 與主模型（LightGBM）在相同測試集上對照準確率（Accuracy / Macro-F1）
2. 對使用者輸入的案件描述計算「每一句話的 Attention 權重（總和 = 1.0）」與「詞重要度」，重現參考專案 `demo_attention.JPG` 的可解釋視覺化效果。
"""
from __future__ import annotations

import re
import jieba
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


class HierarchicalAttentionModel:
    """輕量級雙層 Attention（Word Attention + Sentence Attention）判決預測與句子權重模型。"""

    def __init__(self, embed_dim: int = 32, max_sents: int = 30):
        self.embed_dim = embed_dim
        self.max_sents = max_sents
        self.word_tfidf = TfidfVectorizer(max_features=4000, token_pattern=r"(?u)\b\w+\b")
        self.svd: TruncatedSVD | None = None
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.5)
        self.word_context_u_: np.ndarray | None = None
        self.sent_context_u_: np.ndarray | None = None
        self.classes_: list[str] = []

    @staticmethod
    def segment_sentences(text: str) -> list[list[str]]:
        raw_sents = [s.strip() for s in re.split(r"[，。；！？\n]+", text or "") if s.strip()]
        out = []
        for s in raw_sents:
            words = [w.strip() for w in jieba.cut(s, cut_all=False) if len(w.strip()) >= 2]
            if words:
                out.append(words)
        return out or [[text[:10] or "無"]]

    def fit(self, texts: list[str], y: list[str]) -> "HierarchicalAttentionModel":
        # 1. 建立詞向量空間（以截斷 SVD 近似預訓練 Word2Vec 詞向量）
        all_sents = [" ".join(w) for t in texts for w in self.segment_sentences(t)[: self.max_sents]]
        tfidf_mat = self.word_tfidf.fit_transform(all_sents)
        k = min(self.embed_dim, max(2, tfidf_mat.shape[1] - 1), max(2, tfidf_mat.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        sent_vecs = self.svd.fit_transform(tfidf_mat)

        # 2. 初始化詞級與句級 Context Vector，先以平均池化訓練分類器，再由分類器權重導出注意力方向
        doc_vecs = np.vstack([self._encode_doc_mean(t) for t in texts])
        self.clf.fit(doc_vecs, y)
        self.classes_ = list(self.clf.classes_)

        # 句級注意力向量 u_s：取分類器各類別權重之主軸，使對勝負判別力最強的句子獲得最高 Attention
        coef = self.clf.coef_
        u_s = np.linalg.norm(coef, axis=0)
        self.sent_context_u_ = u_s / (np.linalg.norm(u_s) + 1e-9)
        self.word_context_u_ = self.sent_context_u_.copy()

        # 3. 以句級 Attention 加權後的篇向量重新微調分類器
        att_doc_vecs = np.vstack([self.encode_with_attention(t)[0] for t in texts])
        self.clf.fit(att_doc_vecs, y)
        return self

    def _word_vectors(self, words: list[str]) -> tuple[np.ndarray, list[str]]:
        vocab = self.word_tfidf.vocabulary_
        idf = self.word_tfidf.idf_
        comp = self.svd.components_  # (k, V)
        vecs, valid_words = [], []
        for w in words:
            idx = vocab.get(w)
            if idx is not None:
                vecs.append(comp[:, idx] * idf[idx])
                valid_words.append(w)
        if not vecs:
            return np.zeros((1, comp.shape[0])), words[:1]
        return np.vstack(vecs), valid_words

    def _encode_doc_mean(self, text: str) -> np.ndarray:
        sents = self.segment_sentences(text)[: self.max_sents]
        joined = [" ".join(s) for s in sents]
        m = self.svd.transform(self.word_tfidf.transform(joined))
        return np.mean(m, axis=0)

    def encode_with_attention(self, text: str) -> tuple[np.ndarray, list[dict]]:
        """回傳 (篇向量, 各句資訊[{"sentence", "score", "top_words"}])，句子權重總和 = 1.0。"""
        raw_clauses = [s.strip() for s in re.split(r"[。；！？\n]+", text or "") if len(s.strip()) >= 3]
        if not raw_clauses:
            raw_clauses = [text or ""]
        raw_clauses = raw_clauses[: self.max_sents]

        sent_embeds = []
        sent_meta = []
        u_w = self.word_context_u_
        for clause in raw_clauses:
            words = [w.strip() for w in jieba.cut(clause, cut_all=False) if len(w.strip()) >= 2]
            w_vecs, valid_w = self._word_vectors(words or [clause])
            if u_w is not None:
                e_w = np.tanh(w_vecs @ u_w)
                alpha_w = _softmax(e_w * 3.0)
            else:
                alpha_w = np.ones(len(w_vecs)) / len(w_vecs)
            s_vec = np.sum(w_vecs * alpha_w[:, None], axis=0)
            sent_embeds.append(s_vec)
            top_idx = np.argsort(-alpha_w)[:3]
            top_words = [(valid_w[i], float(alpha_w[i])) for i in top_idx if i < len(valid_w)]
            sent_meta.append({"sentence": clause, "top_words": top_words})

        S = np.vstack(sent_embeds)  # (N_sent, k)
        u_s = self.sent_context_u_
        if u_s is not None:
            e_s = np.tanh(np.abs(S) @ u_s)
            alpha_s = _softmax(e_s * 4.0)
        else:
            alpha_s = np.ones(len(S)) / len(S)
        doc_vec = np.sum(S * alpha_s[:, None], axis=0)
        for i, m in enumerate(sent_meta):
            m["score"] = round(float(alpha_s[i]), 4)
        return doc_vec, sent_meta

    def predict(self, texts: list[str]) -> np.ndarray:
        X = np.vstack([self.encode_with_attention(t)[0] for t in texts])
        return self.clf.predict(X)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = np.vstack([self.encode_with_attention(t)[0] for t in texts])
        return self.clf.predict_proba(X)
