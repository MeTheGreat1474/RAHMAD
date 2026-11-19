# Graph Report - RAHMAD  (2026-11-18)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 473 nodes · 1001 edges · 15 communities (11 shown, 4 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 55 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `934085b8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13

## God Nodes (most connected - your core abstractions)
1. `analyze_text()` - 43 edges
2. `EarlyPredictor` - 26 edges
3. `Judge` - 25 edges
4. `PipelineLogger` - 24 edges
5. `ReverseImageSearchPipeline` - 22 edges
6. `OrchestratorAgent` - 22 edges
7. `analyze_text()` - 22 edges
8. `KnowledgeRetriever` - 20 edges
9. `TestPipelineFlowIntegration` - 19 edges
10. `run_ablation_and_pipeline()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `TestJudgeModule` --uses--> `Judge`  [INFERRED]
  testing/test_phase1_unit.py → model/judge.py
- `TestEarlyPredictor` --uses--> `EarlyPredictor`  [INFERRED]
  testing/test_phase1_unit.py → model/early_predictor.py
- `run_ablation_and_pipeline()` --uses--> `KnowledgeRetriever`  [INFERRED]
  testing/test_phase3_evaluation.py → model/knowledge_retrieval.py
- `evaluate_early_predictor()` --uses--> `EarlyPredictor`  [INFERRED]
  testing/test_phase3_evaluation.py → model/early_predictor.py
- `run_ablation_and_pipeline()` --uses--> `EarlyPredictor`  [INFERRED]
  testing/test_phase3_evaluation.py → model/early_predictor.py

## Import Cycles
- None detected.

## Communities (15 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (11): build_news_clippings_dataset(), clean_for_llm(), Judge, ReverseImageSearchPipeline, main(), evaluate_early_predictor(), generate_report(), load_test_split() (+3 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (4): analyze_text(), analyze_text(), PipelineLogger, TestPipelineFlowIntegration

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (10): download_image(), fetch_image_url(), resolve_article_url(), prepare_dataset(), format_image_path(), is_domain_whitelisted(), download_image(), get_mediaproxy_url() (+2 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (8): extract_and_clean_lead_text(), preprocess_inputs(), process_image(), tokenize_text(), load_image(), load_inputs(), load_text(), TestInputProcessing

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (7): _decode_image(), _get_judge(), _get_ris_pipeline(), _preflight_local_llm(), _tcp_reachable(), generate_sample_base64_png(), TestApiEndpointsIntegration

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (9): _get_retriever(), _decode_image(), _get_judge(), _get_mad_orchestrator(), _get_retriever(), _get_ris_pipeline(), is_social_media_url(), KnowledgeRetriever (+1 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (4): _get_predictor(), _get_predictor(), EarlyPredictor, _MLPHead

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (6): _get_mad_orchestrator(), LLMClient, DebaterAgent, OrchestratorAgent, SharedMemoryBank, TestMultiAgentDebate

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (10): build_lora_config(), clean_text(), compute_metrics(), EarlyPredictor, evaluate(), FakeNewsDataset, get_args(), get_device() (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (4): analyze(), generate(), serve_frontend(), serve_res()

### Community 13 - "Community 13"
Cohesion: 0.47
Nodes (3): createLogLine(), getCurrentTime(), simulateProcess()

## Knowledge Gaps
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `analyze_text()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 11`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `analyze_text()` (e.g. with `.log_clip_output()` and `.log_ep_output()`) actually correct?**
  _`analyze_text()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05706760316066725 - nodes in this community are weakly interconnected._
- **Why does `EarlyPredictor` connect `Community 6` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 10`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `EarlyPredictor` (e.g. with `TestEarlyPredictor` and `evaluate_early_predictor()`) actually correct?**
  _`EarlyPredictor` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.07966101694915254 - nodes in this community are weakly interconnected._
- **Why does `Judge` connect `Community 0` to `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 9`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._