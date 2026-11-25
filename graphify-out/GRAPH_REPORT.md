# Graph Report - RAHMAD  (2026-11-25)

## Corpus Check
- 35 files · ~150,003 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: .csv 3, (none) 2, .ipynb 2)

## Summary
- 533 nodes · 1111 edges · 20 communities (11 shown, 9 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 52 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d958fe48`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_phase3_evaluation.py
- Judge
- analyze_text
- KnowledgeRetriever
- OrchestratorAgent
- analyzer_new.py
- analyzer.py
- TestJudgeModule
- EarlyPredictor
- train_early_predictor.py
- TestEarlyPredictor
- test_phase1_unit.py
- script.js
- input_retrieval.py
- JudgeResponseError
- NewsArticleSpider

## God Nodes (most connected - your core abstractions)
1. `analyze_text()` - 43 edges
2. `Judge` - 32 edges
3. `OrchestratorAgent` - 30 edges
4. `EarlyPredictor` - 26 edges
5. `PipelineLogger` - 24 edges
6. `KnowledgeRetriever` - 24 edges
7. `ReverseImageSearchPipeline` - 22 edges
8. `analyze_text()` - 22 edges
9. `TestPipelineFlowIntegration` - 19 edges
10. `run_ablation_and_pipeline()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `TestJudgeModule` --uses--> `Judge`  [INFERRED]
  testing/test_phase1_unit.py → model/judge.py
- `run_ablation_and_pipeline()` --uses--> `Judge`  [INFERRED]
  testing/test_phase3_evaluation.py → model/judge.py
- `TestEarlyPredictor` --uses--> `EarlyPredictor`  [INFERRED]
  testing/test_phase1_unit.py → model/early_predictor.py
- `run_ablation_and_pipeline()` --uses--> `KnowledgeRetriever`  [INFERRED]
  testing/test_phase3_evaluation.py → model/knowledge_retrieval.py
- `run_ablation_and_pipeline()` --uses--> `OrchestratorAgent`  [INFERRED]
  testing/test_phase3_evaluation.py → model/mad_orchestrator.py

## Import Cycles
- None detected.

## Communities (20 total, 9 thin omitted)

### Community 0 - "test_phase3_evaluation.py"
Cohesion: 0.09
Nodes (6): ReverseImageSearchPipeline, main(), generate_report(), load_test_split(), main(), run_ablation_and_pipeline()

### Community 2 - "analyze_text"
Cohesion: 0.08
Nodes (6): analyze(), generate(), analyze_text(), analyze_text(), PipelineLogger, TestPipelineFlowIntegration

### Community 3 - "KnowledgeRetriever"
Cohesion: 0.05
Nodes (10): _get_retriever(), _get_retriever(), domain_tier(), _get_tier_for_netloc(), is_domain_whitelisted(), registrable_domain(), build_publisher_stats(), is_social_media_url() (+2 more)

### Community 4 - "OrchestratorAgent"
Cohesion: 0.08
Nodes (6): LLMClient, DebaterAgent, strip_non_english(), OrchestratorAgent, SharedMemoryBank, TestMultiAgentDebate

### Community 5 - "analyzer_new.py"
Cohesion: 0.09
Nodes (10): _decode_image(), _get_judge(), _get_mad_orchestrator(), _get_ris_pipeline(), clean_for_llm(), extract_and_clean_lead_text(), preprocess_inputs(), process_image() (+2 more)

### Community 6 - "analyzer.py"
Cohesion: 0.05
Nodes (7): _decode_image(), _get_mad_orchestrator(), _get_ris_pipeline(), _preflight_local_llm(), _tcp_reachable(), generate_sample_base64_png(), TestApiEndpointsIntegration

### Community 8 - "EarlyPredictor"
Cohesion: 0.10
Nodes (4): _get_predictor(), _get_predictor(), EarlyPredictor, _MLPHead

### Community 9 - "train_early_predictor.py"
Cohesion: 0.11
Nodes (10): build_lora_config(), clean_text(), compute_metrics(), EarlyPredictor, evaluate(), FakeNewsDataset, get_args(), get_device() (+2 more)

### Community 11 - "test_phase1_unit.py"
Cohesion: 0.05
Nodes (12): build_news_clippings_dataset(), download_image(), fetch_image_url(), resolve_article_url(), prepare_dataset(), format_image_path(), download_image(), get_mediaproxy_url() (+4 more)

### Community 12 - "script.js"
Cohesion: 0.42
Nodes (6): createLogLine(), createPendingLine(), displayLabel(), getCurrentTime(), setStatus(), simulateProcess()

### Community 13 - "input_retrieval.py"
Cohesion: 0.17
Nodes (5): serve_frontend(), serve_res(), load_image(), load_inputs(), load_text()

## Knowledge Gaps
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Judge` connect `Judge` to `test_phase3_evaluation.py`, `analyzer_new.py`, `analyzer.py`, `TestJudgeModule`, `test_phase1_unit.py`, `JudgeResponseError`, `._build_user_message`, `._aggregate_verdicts`, `.save_judge_log`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `analyze_text()` (e.g. with `.log_clip_output()` and `.log_ep_output()`) actually correct?**
  _`analyze_text()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_phase3_evaluation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0873440285204991 - nodes in this community are weakly interconnected._
- **Why does `analyze_text()` connect `analyze_text` to `Judge`, `KnowledgeRetriever`, `analyzer_new.py`, `analyzer.py`, `EarlyPredictor`, `test_phase1_unit.py`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Judge` (e.g. with `TestJudgeModule` and `run_ablation_and_pipeline()`) actually correct?**
  _`Judge` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Judge` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._
- **Why does `EarlyPredictor` connect `EarlyPredictor` to `test_phase3_evaluation.py`, `analyzer_new.py`, `analyzer.py`, `TestEarlyPredictor`, `test_phase1_unit.py`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._