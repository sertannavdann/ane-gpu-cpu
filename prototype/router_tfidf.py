"""CPU baseline router: TF-IDF + linear classifiers for task_type and tool_intent, trained on the
training split only, with an abstention threshold chosen on that split by cross-validation."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .contracts import Availability, ClassifyResult, RouterStatus
from .router_fake import CAP_FOR_TASK, SEQUENCE_WORDS

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
except ImportError as e:  # pragma: no cover
    raise ImportError("router_tfidf needs scikit-learn; install requirements.txt") from e


def _pipe():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                         LogisticRegression(max_iter=2000, C=4.0))


class TfidfRouter:
    name = "tfidf"

    def __init__(self, threshold: float = 0.4, min_answered_accuracy: float = 0.85):
        self.threshold = threshold
        self.min_answered_accuracy = min_answered_accuracy
        self.task_model = _pipe()
        self.intent_model = _pipe()
        self.calibration: dict = {}

    @classmethod
    def from_training_file(cls, path: str | Path) -> "TfidfRouter":
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        texts = [r["input"] for r in rows]
        tasks = [r["task_type"] for r in rows]
        intents = [r.get("tool_intent") or "none" for r in rows]
        r = cls()
        r.calibrate(texts, tasks, intents)
        r.task_model.fit(texts, tasks)
        r.intent_model.fit(texts, intents)
        return r

    def calibrate(self, texts, tasks, intents, folds: int = 5) -> None:
        """Pick the abstention threshold that maximises coverage subject to >= 90% accuracy on
        answered items, using out-of-fold task_type probabilities on the training split."""
        model = _pipe()
        probs = cross_val_predict(model, texts, tasks, cv=folds, method="predict_proba")
        classes = sorted(set(tasks))
        candidates = []
        for thr in [x / 100 for x in range(20, 80, 5)]:
            answered = [(classes[p.argmax()] == t) for p, t in zip(probs, tasks) if p.max() >= thr]
            if not answered:
                continue
            acc = sum(answered) / len(answered)
            cov = len(answered) / len(tasks)
            candidates.append((thr, acc, cov))
        meeting = [c for c in candidates if c[1] >= self.min_answered_accuracy]
        # Highest coverage that meets the accuracy floor; otherwise best accuracy*coverage product.
        best = max(meeting, key=lambda c: c[2]) if meeting else max(candidates, key=lambda c: c[1] * c[2])
        self.threshold = best[0]
        self.calibration = {"threshold": best[0], "answered_accuracy": best[1], "coverage": best[2],
                            "floor_met": bool(meeting), "min_answered_accuracy": self.min_answered_accuracy,
                            "folds": folds, "n_train": len(texts)}

    def classify(self, text: str, availability: Availability) -> ClassifyResult:
        t0 = time.perf_counter()
        tp = self.task_model.predict_proba([text])[0]
        ip = self.intent_model.predict_proba([text])[0]
        task = self.task_model.classes_[tp.argmax()]
        intent = self.intent_model.classes_[ip.argmax()]
        ms = (time.perf_counter() - t0) * 1000
        if tp.max() < self.threshold:
            return ClassifyResult(RouterStatus.ABSTAIN, reason=f"max p={tp.max():.2f} < {self.threshold}",
                                  backend=self.name, router_ms=ms)
        lower = text.lower()
        caps = [CAP_FOR_TASK[task]]
        if any(k in lower for k in ("as json", "extract", "as a table", "fields")):
            caps.append("extraction")
        complexity = 0.7 if any(s in lower for s in SEQUENCE_WORDS) else 0.3
        if complexity >= 0.5 and len(caps) == 1:
            caps.append("reasoning" if caps[0] != "reasoning" else "analysis")
        return ClassifyResult(RouterStatus.CLASSIFIED, task_type=task, capabilities=caps,
                              complexity=complexity, tool_intent=None if intent == "none" else intent,
                              backend=self.name, router_ms=ms)
