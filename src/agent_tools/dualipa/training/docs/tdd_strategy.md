# DuaLipa LLM Training Module: TDD Strategy

## Testing Framework Setup
```
# conftest.py
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
```

## Sample Inputs and Expected Outputs
Before any coding or testing begins, we must define:

### Input: QA JSON Export
```
{
  "qa_pairs": [
    {
      "question": "What is the capital of France?",
      "answer": "The capital of France is Paris.",
      "reasoning": "Paris has been the capital of France for centuries. Oh wait?! It's also the largest city in the country."
    }
  ]
}
```

### Expected Output: Trained Adapter
```
{
  "model_info": {
    "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "adapter_checksum": "sha256:abc123...",
    "training_timestamp": "2025-03-21T10:30:00Z"
  },
  "training_metrics": {
    "final_loss": 0.0023,
    "perplexity": 1.15,
    "qa_accuracy": 0.92
  },
  "compliance": {
    "ear99_valid": true,
    "license": "apache-2.0"
  },
  "signature": {
    "gpg_fingerprint": "A1B2C3D4...",
    "rotation_schedule": "2025-Q2"
  }
}
```

## 1. Key Rotation CI Tests
```
def test_ci_rotation_schedule():
    with open(".github/workflows/rotate_keys.yml") as f:
        content = f.read()
    assert "0 0 1 */3 *" in content  # Quarterly schedule
    assert "rotate_encryption_key" in content
    assert "OLD_TOKEN_FILE" in content and "NEW_TOKEN_FILE" in content

def test_key_rotation_implementation():
    with patch("keyring.set_password") as mock_set, \
         patch("keyring.delete_password") as mock_del, \
         patch("os.remove") as mock_rm:
        rotate_encryption_key("old.enc", "new.enc")
        assert mock_del.called_with("dualipa_hf", "old.enc")
        assert mock_rm.called_with("old.enc")
```

## 2. Checksum Automation Tests
```
def test_checksum_fetch_real():
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "abc123"
        assert fetch_model_checksum("test-model") == "abc123"

def test_checksum_validation_failure():
    with patch("requests.get") as mock_get, \
         pytest.raises(ValueError):
        mock_get.return_value.text = "bad_checksum"
        load_and_configure_model("unsloth/test-model")
```

## 3. Enhanced Compliance Tests
```
def test_ear99_compliance():
    with pytest.raises(ValueError) as e:
        check_compliance("military-optimized-model")
    assert "EAR99" in str(e.value)

def test_license_fetch_real():
    with patch("huggingface_hub.HfApi.model_info") as mock_info:
        mock_info.return_value.card_data = {"license": "apache-2.0"}
        check_compliance("valid-model")
        assert mock_info.called
```

## 4. Signature Verification Tests
```
def test_signature_generation():
    with patch("gnupg.GPG.sign") as mock_sign:
        upload_to_hf(model, tokenizer, "adapter", "merged", "user", "token.enc")
        assert mock_sign.called_with(any, keyid="default_key_id")

def test_verification_failure_handling():
    with patch("gnupg.GPG.verify_data") as mock_verify, \
         pytest.raises(ValueError):
        mock_verify.return_value.valid = False
        verify_hf_download("invalid_model")
```

## 5. Drift Prevention Tests
```
def test_inference_distribution():
    train_dist = {"low": 0.2, "medium": 0.5, "high": 0.3}
    outputs = ["low"]*20 + ["medium"]*50 + ["high"]*30
    inference_dist = calculate_distribution(outputs)
    assert js_divergence(train_dist, inference_dist) < 0.1
```

## Updated Validation Checklist
```
[
    ("Checksum Validation", checksum == fetch_model_checksum()),
    ("License Compliance", license in allowed_licenses),
    ("Key Rotation", "rotate_keys.yml" in CI and old_key_deleted),
    ("Signature Verification", verify_hf_download passes),
    ("Drift Prevention", entropy < 0.1 threshold)
]
```

## Enhanced Multi-Layer Testing Strategy
| Layer | Coverage | New Tests |
|-------|----------|-----------|
| **Security** | Key rotation, GPG signing | `test_ci_rotation_schedule`, `test_signature_generation` |
| **Compliance** | Real license checks, EAR99 | `test_ear99_compliance`, `test_license_fetch_real` |
| **Integrity** | Checksum validation | `test_checksum_fetch_real`, `test_checksum_validation_failure` |
| **CI/CD** | Scheduled jobs | `test_ci_rotation_schedule`, `test_ci_regression` |

## New Best Practices
1. **Rotation Testing**: Test key rotation manually and via CI schedule mock
2. **Signature Management**: Store GPG keys in GitHub Secrets, test signing with test key in CI
3. **Checksum Validation**: Test checksum fetch failure fallback, add checksum to model card metadata

## Final Test Coverage Requirements
```
# pytest.ini
[pytest]
asyncio_mode = auto
min_cov = 95
addopts = 
    --cov=dualipa.training 
    --cov-report=term-missing
    --cov-fail-under=95
filterwarnings =
    ignore::DeprecationWarning
```

This updated TDD strategy now includes sample inputs and expected outputs as a prerequisite for coding and testing. It also maintains the key improvements from the previous version, ensuring comprehensive coverage of security, compliance, and performance aspects of the DuaLipa LLM Training Module.

