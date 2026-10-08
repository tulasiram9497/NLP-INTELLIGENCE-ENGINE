from nlp_engine.orchestration.orchestrator import NLPOrchestrator


def test_orchestrator_predicts_sentiment():
    orchestrator = NLPOrchestrator()
    result = orchestrator.analyze("I absolutely love this product and it is amazing.", task="sentiment")

    assert result["task"] == "sentiment"
    assert result["label"] in {"positive", "negative", "neutral"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model"]
