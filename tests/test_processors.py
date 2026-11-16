from genai_suite.processors import extract_entities, infer_priority, redact_pii, summarize


def test_redacts_email_phone_pan_and_valid_card():
    text = "Mail ravi@acme.in, call 9876543210, PAN ABCDE1234F, card 4111 1111 1111 1111"
    redacted, found = redact_pii(text)
    assert "ravi@acme.in" not in redacted and "9876543210" not in redacted
    assert found == {"EMAIL": 1, "PHONE": 1, "PAN": 1, "CARD": 1}


def test_card_requires_luhn():
    _, found = redact_pii("order 1234 5678 9012 3456")
    assert "CARD" not in found


def test_aadhaar_detected():
    redacted, found = redact_pii("Aadhaar 2345 6789 0123")
    assert found.get("AADHAAR") == 1 and "[AADHAAR]" in redacted


def test_entities_and_priority():
    ents = extract_entities("Invoice INV-2231 for INR 45,000 due in 30 days")
    assert ents["ids"] == ["INV-2231"] and ents["amounts"] and ents["durations"] == ["30 days"]
    assert infer_priority("URGENT: account locked") == "high"
    assert infer_priority("quick question about leave") == "normal"


def test_summarize_keeps_requested_length():
    text = "Leave is 18 days. Sick leave is 12 days. Parental leave varies. Remote work is two days."
    assert len(summarize(text, 2).split(". ")) <= 2
