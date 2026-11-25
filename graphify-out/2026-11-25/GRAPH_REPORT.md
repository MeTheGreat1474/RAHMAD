# Graph Report - RAHMAD  (2026-11-24)

## Corpus Check
- 35 files · ~149,769 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: .csv 3, (none) 2, .ipynb 2)

## Summary
- 530 nodes · 1103 edges · 18 communities (11 shown, 7 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 52 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `14d977d9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_phase3_evaluation.py
- Judge
- TestPipelineFlowIntegration
- KnowledgeRetriever
- OrchestratorAgent
- test_phase1_unit.py
- analyzer.py
- analyzer_new.py
- EarlyPredictor
- train_early_predictor.py
- TestEarlyPredictor
- knowledge_retrieval.py
- script.js
- app.py
- PipelineLogger
- analyze_text

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
- `run_ablation_and_pipeline()` --uses--> `Judge`  [INFERRED]
  testing/test_phase3_evaluation.py → model/judge.py
- `TestEarlyPredictor` --uses--> `EarlyPredictor`  [INFERRED]
  testing/test_phase1_unit.py → model/early_predictor.py
- `run_ablation_and_pipeline()` --uses--> `KnowledgeRetriever`  [INFERRED]
  testing/test_phase3_evaluation.py → model/knowledge_retrieval.py
- `run_ablation_and_pipeline()` --uses--> `OrchestratorAgent`  [INFERRED]
  testing/test_phase3_evaluation.py → model/mad_orchestrator.py
- `evaluate_early_predictor()` --uses--> `EarlyPredictor`  [INFERRED]
  testing/test_phase3_evaluation.py → model/early_predictor.py

## Import Cycles
- None detected.

## Communities (18 total, 7 thin omitted)

### Community 0 - "test_phase3_evaluation.py"
Cohesion: 0.07
Nodes (13): build_news_clippings_dataset(), download_image(), fetch_image_url(), resolve_article_url(), clean_for_llm(), ReverseImageSearchPipeline, main(), evaluate_early_predictor() (+5 more)

### Community 1 - "Judge"
Cohesion: 0.05
Nodes (5): _clean_user_facing_text(), Judge, JudgeResponseError, JudgeTruncatedResponse, TestJudgeModule

### Community 3 - "KnowledgeRetriever"
Cohesion: 0.07
Nodes (5): _get_retriever(), _get_retriever(), is_social_media_url(), KnowledgeRetriever, TestKnowledgeRetrieval

### Community 4 - "OrchestratorAgent"
Cohesion: 0.08
Nodes (6): LLMClient, DebaterAgent, strip_non_english(), OrchestratorAgent, SharedMemoryBank, TestMultiAgentDebate

### Community 5 - "test_phase1_unit.py"
Cohesion: 0.06
Nodes (11): prepare_dataset(), format_image_path(), extract_and_clean_lead_text(), preprocess_inputs(), process_image(), tokenize_text(), load_image(), load_inputs() (+3 more)

### Community 7 - "analyzer_new.py"
Cohesion: 0.24
Nodes (6): analyze_text(), _decode_image(), _get_judge(), _get_mad_orchestrator(), _get_predictor(), _get_ris_pipeline()

### Community 9 - "train_early_predictor.py"
Cohesion: 0.09
Nodes (10): build_lora_config(), clean_text(), compute_metrics(), EarlyPredictor, evaluate(), FakeNewsDataset, get_args(), get_device() (+2 more)

### Community 11 - "knowledge_retrieval.py"
Cohesion: 0.06
Nodes (9): domain_tier(), _get_tier_for_netloc(), is_domain_whitelisted(), registrable_domain(), NewsArticleSpider, build_publisher_stats(), download_image(), get_mediaproxy_url() (+1 more)

### Community 12 - "script.js"
Cohesion: 0.47
Nodes (3): createLogLine(), getCurrentTime(), simulateProcess()

### Community 13 - "app.py"
Cohesion: 0.22
Nodes (4): analyze(), generate(), serve_frontend(), serve_res()

### Community 16 - "analyze_text"
Cohesion: 0.16
Nodes (8): analyze_text(), _decode_image(), _get_judge(), _get_mad_orchestrator(), _get_predictor(), _get_ris_pipeline(), _preflight_local_llm(), _tcp_reachable()

## Knowledge Gaps
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Judge` connect `Judge` to `test_phase3_evaluation.py`, `test_phase1_unit.py`, `analyzer_new.py`, `knowledge_retrieval.py`, `analyze_text`?**
  _High betweenness centrality (0.146) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `analyze_text()` (e.g. with `.log_clip_output()` and `.log_ep_output()`) actually correct?**
  _`analyze_text()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_phase3_evaluation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06545879602571596 - nodes in this community are weakly interconnected._
- **Why does `analyze_text()` connect `analyze_text` to `test_phase3_evaluation.py`, `TestPipelineFlowIntegration`, `KnowledgeRetriever`, `test_phase1_unit.py`, `app.py`, `PipelineLogger`, `.log_clip_output`?**
  _High betweenness centrality (0.137) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Judge` (e.g. with `TestJudgeModule` and `run_ablation_and_pipeline()`) actually correct?**
  _`Judge` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Judge` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Why does `EarlyPredictor` connect `EarlyPredictor` to `test_phase3_evaluation.py`, `test_phase1_unit.py`, `analyzer_new.py`, `TestEarlyPredictor`, `analyze_text`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._