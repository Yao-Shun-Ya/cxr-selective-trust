from src.medvlm_trust.inference.schema import PRIMARY_LABELS, parse_vlm_output


def test_parse_valid_json():
    raw = '''
    {
      "findings": {
        "atelectasis": "positive",
        "cardiomegaly": "negative",
        "consolidation": "uncertain",
        "edema": "negative",
        "pleural_effusion": "positive",
        "pneumonia": "negative",
        "pneumothorax": "negative",
        "no_finding": "negative"
      },
      "probability_positive": {
        "atelectasis": 0.9,
        "cardiomegaly": 0.1,
        "consolidation": 0.5,
        "edema": 0.1,
        "pleural_effusion": 0.8,
        "pneumonia": 0.1,
        "pneumothorax": 0.05,
        "no_finding": 0.1
      },
      "confidence": {
        "atelectasis": 0.85,
        "cardiomegaly": 0.9,
        "consolidation": 0.4,
        "edema": 0.9,
        "pleural_effusion": 0.8,
        "pneumonia": 0.9,
        "pneumothorax": 0.95,
        "no_finding": 0.9
      },
      "rationale": "test",
      "image_reliance_claim": "visual evidence used"
    }
    '''
    parsed = parse_vlm_output(raw)
    assert parsed.parse_success
    assert parsed.findings["atelectasis"].status == "positive"
    assert parsed.findings["cardiomegaly"].probability_positive == 0.1
    assert parsed.findings["pleural_effusion"].status == "positive"
    assert parsed.image_reliance_claim == "visual evidence used"


def test_parse_markdown_fence():
    raw = '```json\n{"findings": {"atelectasis": "positive"}}\n```'
    parsed = parse_vlm_output(raw)
    assert parsed.parse_success
    assert parsed.findings["atelectasis"].status == "positive"
    assert parsed.findings["cardiomegaly"].status == "not_mentioned"


def test_parse_invalid_json():
    parsed = parse_vlm_output("I cannot answer this question.")
    assert not parsed.parse_success
    assert parsed.parse_error == "no_json_object_found"


def test_missing_findings_object():
    parsed = parse_vlm_output('{"confidence": {"atelectasis": 0.9}}')
    assert not parsed.parse_success
    assert parsed.parse_error == "findings_missing_or_not_object"


def test_invalid_probability_is_null():
    raw = '{"findings": {"atelectasis": "positive"}, "probability_positive": {"atelectasis": 1.5}}'
    parsed = parse_vlm_output(raw)
    assert parsed.parse_success
    assert parsed.findings["atelectasis"].probability_positive is None


def test_all_labels_present():
    parsed = parse_vlm_output('{"findings": {}}')
    assert parsed.parse_success
    assert set(parsed.findings.keys()) == set(PRIMARY_LABELS)
