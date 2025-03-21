# Task Plan for DuaLipa LLM Q&A Generation Module

This plan ensures a secure, scalable, production-ready implementation, adhering to TDD, documentation-first principles, and lessons from `.cursor/TESTING_LESSONS.md` and `.cursor/LESSONS_LEARNED.md`. It targets 90% success with human oversight, building iteratively from an MVP.

**Current Date:** March 19, 2025

---

## Phase 0: Preparation and Documentation Foundation

### Task 0.1: Review Official Documentation
- [x] Read and link documentation:
  - `pytest`: https://docs.pytest.org/en/stable/
  - `pydantic`: https://docs.pydantic.dev/latest/
  - `asyncio`: https://docs.python.org/3/library/asyncio.html
  - `litellm`: [Verify actual URL]
  - `json`: https://docs.python.org/3/library/json.html
  - `loguru`: https://loguru.readthedocs.io/
  - `bleach`: https://bleach.readthedocs.io/
  - `sentence-transformers`: https://sbert.net/docs/
  - `prometheus-client`: https://github.com/prometheus/client_python
- [x] Write docstring in `conftest.py` summarizing behaviors.
- [x] Verify methods with `method_validator`.

### Task 0.2: Establish Real-World Baseline (MVP)
- [x] Obtain real Extraction Module JSON sample.
- [x] Write smoke test in `test_processor.py`: `test_minimal_pipeline_real_data`
  - Input: Real JSON.
  - Expect: One Q&A pair written to file.
- [x] Run test and fail.
- [x] Implement minimal `process_extraction_json` (use `asyncio.to_thread()`).
- [x] Run test and pass; verify output manually.

---

## Phase 1: Setup and Core Model Validation

### Task 1.1: Set Up Project Structure and Testing Framework
- [x] Review existing tests from related modules.
- [x] Create structure:
  - `src/agent_tools/dualipa/qa/` (all files as before + `retry_llm_call.py`)
  - `tests/qa/` (all test files as before)
- [x] Add doc links to all files.
- [x] Configure `conftest.py` with `asyncio` fixture and path setup.
- [x] Write test: `test_structure_exists`, implement, ensure pass.

### Task 1.2: Implement and Test Pydantic Models with Constraints
- [x] Write test: `test_qapair_model_validation`, implement `QAPair`, ensure pass.
- [x] Write test: `test_qapair_model_failure`, ensure pass.
- [x] Write test: `test_qaresponse_model_validation`, implement `QAResponse`, ensure pass.
- [x] Write test: `test_constraint_tracking_phase1`
  - Expect: Temperature in `QAPair` within 0.0-1.0.
- [x] Update `QAPair` with constraint, ensure all tests pass.

### Task 1.3: Configuration Management
- [x] Write test: `test_load_config_real`, implement `load_config` (YAML with env support), ensure pass.
- [x] Write test: `test_validate_temperature_range`, update, ensure pass.

### Task 1.4: Security Validation
- [x] Write test: `test_sanitize_input_json`, implement `sanitize_input_json`, ensure pass.
- [x] Write test: `test_sanitize_prompt_injection`, update, ensure pass.

---

## Phase 2: Core Business Logic Components

### Task 2.1: Input Validation and Normalization
- [x] Write test: `test_validate_input_json_real`, implement, ensure pass.
- [x] Write test: `test_validate_input_json_invalid`, update, ensure pass.

### Task 2.2: Temperature Iteration Logic with Rate-Limiting
- [x] Write test: `test_iterate_temperatures_real`
  - Input: Real section, config temps.
  - Expect: Isolated responses per temp.
- [x] Implement `iterate_temperatures`.
- [x] Write test: `test_iterate_temperatures_deadlock`
  - Expect: No context overlap (clear `messages` per temp).
- [x] Update `iterate_temperatures` with context reset, ensure pass.
- [x] Write test: `test_iterate_temperatures_rate_limit`, update with semaphore, ensure pass.
- [ ] Human Review: Verify temperature context isolation.

### Task 2.3: Bidirectional Generation
- [x] Write test: `test_generate_reversed_qa_pairs_real`, implement, ensure pass.
- [x] Write test: `test_generate_reversed_qa_pairs_quality`, update, ensure pass.

### Task 2.4: Reasoning Enrichment with Enhanced Error Recovery
- [x] Write test: `test_generate_markdown_qa_pairs_real`, implement with `retry_llm_call`, ensure pass.
- [x] Write test: `test_generate_retry_persistent_failure`
  - Expect: 3 retries, 5min circuit break.
- [x] Update `retry_llm_call` with circuit breaker, ensure pass.
- [x] Write test: `test_generate_dead_letter_queue`
  - Expect: Failed pairs in `dead_letter.json`.
- [x] Update `retry_llm_call` with file-based queue, ensure pass.
- [ ] Fix: Address issue with dead letter queue test patching (test currently skipped)

### Task 2.5: Advanced Deduplication with Tuning
- [x] Write test: `test_deduplicate_qa_pairs_real`, implement basic deduplication, ensure pass.
- [x] Write test: `test_deduplicate_semantic`
  - Expect: Paraphrased pairs deduplicated (threshold TBD).
- [x] Update `deduplicate_qa_pairs` with `sentence-transformers`, ensure pass.
- [ ] Human Tuning: Set similarity threshold (e.g., 0.85) via real-world validation; update and retest.

### Task 2.6: Output Validation
- [x] Write test: `test_validate_qa_pair_real`, implement, ensure pass.
- [x] Write test: `test_validate_qa_pair_failure`, update, ensure pass.

### Task 2.7: Model Fallback Strategy
- [x] Write test: `test_fallback_model`, update `retry_llm_call`, ensure pass.
- [x] Write test: `test_cost_aware_routing`
  - Simulate cost (e.g., $0.01 vs. $0.03).
  - Expect: Cheaper model chosen.
- [x] Update `retry_llm_call`, ensure pass.
- [ ] Human Review: Validate cost logic with real pricing data.

### Task 2.8: Constraint Tracking for Phase 2
- [x] Write test: `test_constraint_tracking_phase2`
  - Expect: Config temps respected, sanitized inputs used.
- [x] Update relevant functions, ensure all tests pass.

---

## Phase 3: Infrastructure and Integration

### Task 3.1: Cache Implementation
- [x] Write test: `test_cache_hit_real` to document cache/LLM generation relationship.
- [x] Write test: `test_cache_hit_rate` to illustrate statistics and direct cache API usage.
- [x] Added test: `test_cache_key_consistency` to demonstrate config/key relationship.
- [x] Implemented multi-level caching (memory + disk) with graceful fallbacks.
- [x] Integrated caching with retry logic in a non-intrusive way.
- [x] Updated documentation to reflect proper test philosophy.

### Task 3.2: Async Processing with Scalability
- [x] Write test: `test_batch_process_sections_real`, implement, ensure pass.
- [x] Write test: `test_batch_process_worker_pool`
  - Expect: 4 workers handle 10 sections.
- [x] Added test: `test_semaphore_rate_limiting` to verify concurrency control.
- [x] Implemented `batch_processing.py` with configurable worker pools.
- [x] Update processor.py to use batch processing for section handling.
- [x] Added detailed performance tracking and metrics.

### Task 3.3: Full Pipeline Integration with Monitoring
- [x] Write test: `test_process_extraction_json_full_real`, implement, ensure pass.
- [x] Write test: `test_process_metadata_with_metrics`, implement `monitoring.py`, ensure pass.
- [x] Write test: `test_error_alerting`
  - Expect: Alert on `error_count > 5`.
- [x] Update `monitoring.py`, ensure pass.
- [x] Implemented additional test: `test_cache_metrics_integration` to verify cache monitoring.
- [ ] Human Tuning: Set alert threshold via real-world data; retest.

### Task 3.4: Constraint Tracking for Phase 3
- [x] Write test: `test_constraint_tracking_phase3`
  - Expect: Worker pool respects config, metrics logged.
- [x] Write additional tests: `test_worker_pool_constraint_with_metrics` and `test_max_qa_pairs_constraint`
- [x] Updated `process_section` to enforce the `max_qa_pairs_per_section` constraint and respect the `bidirectional_ratio`
- [x] Ensured all tests pass with implemented constraints

---

## Phase 4: Optimization and CLI

### Task 4.1: Performance Optimization with Load Testing
- [x] Write test: `test_pipeline_performance_real`, optimize, ensure pass.
- [x] Write test: `test_load_scalability`
  - Expect: 100 sections < 120s.
- [x] Optimize `batch_process_sections`, ensure pass.
- [ ] Human Review: Validate scalability with real load.

### Task 4.2: CLI Implementation
- [x] Write test: `test_cli_execution_real`, implement with `__main__`, ensure pass.

---

## Phase 5: Final Verification and CI

### Task 5.1: Continuous Integration
- [ ] Write `.github/workflows/ci.yml`:
  - Steps: `pytest --cov`, coverage > 95%, regression test (baseline time < 120s).
- [ ] Write test: `test_ci_regression`, implement, ensure pass.
- [ ] Human Calibration: Set performance gate (e.g., 120s) based on real runs.

### Task 5.2: Comprehensive Validation
- [ ] Run full suite: `pytest --cov=dualipa.qa --cov-report=term-missing`.
- [ ] Verify coverage: 100% models, 95% LLM, 100% logic, full integration.
- [ ] Verify real-world output with `cli.py`.

---

### Improvements from Critique
1. **Constraint Tracking:** Added per-phase tests (1.2, 2.8, 3.4) to enforce consistency.
2. **Semantic Deduplication:** Task 2.5 now includes human tuning for thresholds.
3. **Cost-Aware Routing:** Task 2.7 adds cost simulation and review.
4. **CI/CD Complexity:** Task 5.1 details YAML steps with calibration.
5. **Temperature Deadlocks:** Task 2.2 tests context isolation with human review.
6. **Human Oversight:** Explicit tuning/validation steps added (2.5, 2.7, 3.3, 4.1, 5.1).

### Success Likelihood
- With these changes, the plan rises from ~70% to ~90% success probability by mitigating LLM limitations through human validation and robust testing.

