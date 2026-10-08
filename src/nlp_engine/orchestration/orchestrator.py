from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

from nlp_engine.preprocessing.text_preprocessor import extract_keywords


class NLPOrchestrator:
    """Simple orchestrator for the initial NLP vertical slice."""

    def __init__(self) -> None:
        self.model = self._build_demo_model()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

    def _build_demo_model(self) -> Pipeline:
        training_data = {
            "positive": [
                "I love this product and it works very well",
                "This is amazing and extremely useful",
                "Fantastic experience and great quality",
                "I am very happy with the performance",
                "The result is excellent and satisfying",
            ],
            "negative": [
                "This is terrible and disappointing",
                "I hate this product and it fails badly",
                "The experience was awful and frustrating",
                "Very poor quality and not useful",
                "It was bad and I regret buying it",
            ],
            "neutral": [
                "The service is available and the report is ready",
                "I reviewed the document and noted the details",
                "The meeting was scheduled for this afternoon",
                "The account status is unchanged at this time",
                "I received the files and will review them",
            ],
        }
        texts: list[str] = []
        labels: list[str] = []
        for label, values in training_data.items():
            texts.extend(values)
            labels.extend([label] * len(values))

        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipeline.fit(texts, labels)
        return pipeline

    def predict(self, text: str, task: str = "classification") -> dict[str, Any]:
        label = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        confidence = float(max(probabilities))
        return {
            "task": task,
            "label": label,
            "confidence": confidence,
            "model": "tfidf-logistic-regression",
        }

    def analyze(self, text: str, task: str = "classification") -> dict[str, Any]:
        if task == "sentiment":
            return self.predict(text, task="sentiment")
        if task == "keywords":
            return {"task": "keywords", "keywords": extract_keywords(text, top_n=5)}
        if task == "ner":
            return {"task": "ner", "entities": []}
        if task == "summary":
            return {"task": "summary", "summary": text[:220].strip()}
        if task == "similarity":
            return {"task": "similarity", "score": 0.0, "label": "unknown"}
        return self.predict(text, task=task)

    def similarity(self, text: str, candidates: list[str]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        all_texts = [text, *candidates]
        vectors = self.vectorizer.fit_transform(all_texts)
        similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:])
        results = []
        for index, score in enumerate(similarity_matrix[0]):
            results.append({
                "document": candidates[index],
                "score": float(score),
            })
        return sorted(results, key=lambda item: item["score"], reverse=True)
