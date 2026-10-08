from nlp_engine.preprocessing.text_preprocessor import normalize_text, tokenize, extract_keywords


def test_normalize_text_lowercases_and_strips_punctuation():
    text = "Hello, WORLD!!! This is great."
    normalized = normalize_text(text)

    assert "hello" in normalized
    assert "world" in normalized
    assert "!!!" not in normalized


def test_tokenize_returns_clean_tokens():
    text = "Hello, WORLD!!! This is great."
    tokens = tokenize(text)

    assert "hello" in tokens
    assert "world" in tokens
    assert "great" in tokens
    assert all(token.isalpha() for token in tokens)


def test_extract_keywords_returns_top_terms():
    text = "Machine learning helps build a better model for text classification and sentiment analysis."
    keywords = extract_keywords(text, top_n=5)

    assert len(keywords) <= 5
    assert all(isinstance(item, str) for item in keywords)
