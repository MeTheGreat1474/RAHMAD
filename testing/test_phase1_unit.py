"""
testing/test_phase1_unit.py

Phase 1 Unit Tests for RAHMAD Fake News Detection System.
Covers:
  - 1.1 Input Processing (backend/input_processing.py)
  - 1.2 Early Predictor (model/early_predictor.py)
  - 1.3 Judge Module (model/judge.py)
  - 1.4 Knowledge Retrieval (model/knowledge_retrieval.py)
  - 1.5 Multi-Agent Debate (model/mad_orchestrator.py, model/mad_agent.py, model/shared_memory.py)

Designed to run in isolation with deterministic assertions and mocked external dependencies where appropriate.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# SECTION 1.1: INPUT PROCESSING TESTS
# ---------------------------------------------------------------------------
from backend.input_processing import (
    clean_for_llm,
    extract_and_clean_lead_text,
    tokenize_text,
    process_image,
    preprocess_inputs,
)

class TestInputProcessing:
    """Unit tests for backend/input_processing.py"""

    def test_clean_for_llm_normal_article(self):
        raw = (
            "Breaking news reported by John Doe on https://example.com/news?id=123. "
            "Contact us at editor@fakemail.com or call +1-800-555-0199 for more details."
        )
        cleaned = clean_for_llm(raw)
        assert "<URL>" in cleaned, "Raw URLs should be replaced with <URL> placeholder"
        assert "<EMAIL>" in cleaned, "Emails should be replaced with <EMAIL> placeholder"
        assert "https://" not in cleaned, "No raw URLs should remain in LLM text"
        assert "editor@fakemail.com" not in cleaned, "No raw emails should remain"

    def test_clean_for_llm_unicode_handling(self):
        raw = "Café résumé with international figures: ñ, ü, and Chinese characters 中文."
        cleaned = clean_for_llm(raw)
        # Verify transliteration / ASCII normalization
        assert "Cafe" in cleaned or "resume" in cleaned
        # Verify all characters in output are standard ASCII
        assert all(ord(c) < 128 for c in cleaned), "All characters should be transliterated to ASCII"

    def test_clean_for_llm_empty_string(self):
        assert clean_for_llm("") == "", "Empty string should return empty string"

    def test_extract_and_clean_lead_text_normal(self):
        raw = (
            "MASSIVE WILDFIRE BREAKS OUT NEAR METROPOLIS\n\n"
            "Firefighters are struggling to contain the blaze spreading across 500 acres.\n"
            "Evacuation orders were issued Tuesday morning."
        )
        lead = extract_and_clean_lead_text(raw)
        assert lead.islower(), "Lead text must be lowercased for CLIP consistency"
        assert "\n" not in lead, "Lead text must be collapsed to a single line"
        assert "massive wildfire" in lead
        assert "firefighters are struggling" in lead
        assert "evacuation orders" not in lead, "Should only extract the first two non-empty lines"

    def test_extract_and_clean_lead_text_single_line(self):
        raw = "BREAKING: SCIENTISTS DISCOVER NEW PLANET"
        lead = extract_and_clean_lead_text(raw)
        assert lead.islower()
        assert "scientists discover new planet" in lead

    def test_tokenize_text_output_shape(self):
        sample_text = "breaking news scientists discover new ancient ruins in the desert"
        input_ids, attention_mask = tokenize_text(sample_text)
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert tuple(input_ids.shape) == (1, 77), f"Expected [1, 77], got {input_ids.shape}"
        assert tuple(attention_mask.shape) == (1, 77), f"Expected [1, 77], got {attention_mask.shape}"

    def test_tokenize_text_long_truncation(self):
        long_text = "word " * 500
        input_ids, attention_mask = tokenize_text(long_text)
        assert input_ids.shape[1] == 77, "Sequences longer than 77 tokens must be truncated to 77"

    def test_process_image_output_shape(self):
        img = Image.new("RGB", (400, 300), color=(73, 109, 137))
        pv = process_image(img)
        assert isinstance(pv, torch.Tensor)
        assert tuple(pv.shape) == (1, 3, 224, 224), f"Expected [1, 3, 224, 224], got {pv.shape}"

    def test_process_image_grayscale_input(self):
        gray_img = Image.new("L", (320, 240), color=128)
        pv = process_image(gray_img)
        assert tuple(pv.shape) == (1, 3, 224, 224), "Grayscale image should be processed into 3-channel tensor"

    def test_preprocess_inputs_full_pipeline(self):
        img = Image.new("RGB", (250, 250), color=(20, 40, 60))
        raw_text = "Headline Alert\nFirst line details about the event.\nSecond line extra."
        result = preprocess_inputs(raw_text, img)

        expected_keys = {"llm_text", "lead_text", "input_ids", "attention_mask", "pixel_values"}
        assert expected_keys.issubset(result.keys()), f"Result missing keys: {expected_keys - set(result.keys())}"
        assert isinstance(result["llm_text"], str)
        assert isinstance(result["lead_text"], str)
        assert result["input_ids"].shape == (1, 77)
        assert result["attention_mask"].shape == (1, 77)
        assert result["pixel_values"].shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# SECTION 1.2: EARLY PREDICTOR TESTS
# ---------------------------------------------------------------------------
from model.early_predictor import (
    _clean_text_for_clip,
    EarlyPredictor,
    FAKE_THRESHOLD,
    REAL_THRESHOLD,
    LORA_ADAPTER_PATH,
    MLP_HEAD_PATH,
    PROCESSOR_PATH,
)

class TestEarlyPredictor:
    """Unit tests for model/early_predictor.py"""

    def test_clean_text_for_clip_html_removal(self):
        raw = "<b>Breaking News:</b> Major event occurred <p>details follow</p>."
        cleaned = _clean_text_for_clip(raw)
        assert "<b>" not in cleaned and "</b>" not in cleaned and "<p>" not in cleaned
        assert "breaking news" in cleaned
        assert cleaned.islower()

    def test_clean_text_for_clip_url_removal(self):
        raw = "Check this link https://misinfo-source.com/hoax?id=99 for the truth"
        cleaned = _clean_text_for_clip(raw)
        assert "https" not in cleaned
        assert "misinfo-source" not in cleaned
        assert "check this link" in cleaned

    def test_clean_text_for_clip_emoji_removal(self):
        raw = "Shocking development! 🚨📰💥 Must read now!"
        cleaned = _clean_text_for_clip(raw)
        assert "🚨" not in cleaned and "📰" not in cleaned and "💥" not in cleaned
        assert "shocking development" in cleaned

    def test_clean_text_for_clip_non_string(self):
        assert _clean_text_for_clip(None) == ""
        assert _clean_text_for_clip(123) == ""
        assert _clean_text_for_clip([]) == ""

    def test_build_verdict_high_fake(self):
        verdict = EarlyPredictor._build_verdict(0.85, FAKE_THRESHOLD, REAL_THRESHOLD)
        assert verdict["label"] == "FAKE"
        assert verdict["color_class"] == "manipulated"
        assert verdict["pass_to_agents"] is True
        assert verdict["confidence"] == "85.0%"

    def test_build_verdict_high_real(self):
        verdict = EarlyPredictor._build_verdict(0.15, FAKE_THRESHOLD, REAL_THRESHOLD)
        assert verdict["label"] == "REAL"
        assert verdict["color_class"] == "real"
        assert verdict["pass_to_agents"] is True
        assert verdict["confidence"] == "15.0%"

    def test_build_verdict_uncertain(self):
        verdict = EarlyPredictor._build_verdict(0.50, FAKE_THRESHOLD, REAL_THRESHOLD)
        assert verdict["label"] == "UNCERTAIN"
        assert verdict["color_class"] == "inconclusive"
        assert verdict["pass_to_agents"] is True
        assert verdict["confidence"] == "50.0%"

    def test_build_verdict_boundary_fake(self):
        verdict = EarlyPredictor._build_verdict(0.60, FAKE_THRESHOLD, REAL_THRESHOLD)
        assert verdict["label"] == "FAKE", "Threshold >= 0.60 must classify as FAKE"

    def test_build_verdict_boundary_real(self):
        verdict = EarlyPredictor._build_verdict(0.40, FAKE_THRESHOLD, REAL_THRESHOLD)
        assert verdict["label"] == "REAL", "Threshold <= 0.40 must classify as REAL"

    def test_model_loading_artifacts_exist(self):
        # Validate that the trained artifacts exist on disk
        assert LORA_ADAPTER_PATH.exists(), f"Missing LoRA adapter at {LORA_ADAPTER_PATH}"
        assert MLP_HEAD_PATH.exists(), f"Missing MLP head at {MLP_HEAD_PATH}"
        assert PROCESSOR_PATH.exists(), f"Missing Processor at {PROCESSOR_PATH}"
        # Validate that _validate_artifacts() executes without error
        EarlyPredictor._validate_artifacts()

    def test_model_loading_missing_artifacts(self, monkeypatch):
        fake_path = Path("/nonexistent/directory/model_file")
        monkeypatch.setattr("model.early_predictor.LORA_ADAPTER_PATH", fake_path)
        with pytest.raises(FileNotFoundError):
            EarlyPredictor._validate_artifacts()

    def test_predict_mocked_forward(self):
        """Test EarlyPredictor.predict logic with mocked forward pass for rapid testing."""
        ep = object.__new__(EarlyPredictor)
        ep.device = torch.device("cpu")
        ep.fake_threshold = FAKE_THRESHOLD
        ep.real_threshold = REAL_THRESHOLD
        ep._prepare_inputs = MagicMock(return_value={
            "input_ids": torch.zeros((1, 77), dtype=torch.long),
            "attention_mask": torch.ones((1, 77), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 224, 224)),
        })
        ep._forward = MagicMock(return_value=0.88)

        img = Image.new("RGB", (100, 100))
        result = ep.predict("Sample fake headline", img)
        assert result["label"] == "FAKE"
        assert result["probability"] == 0.88
        assert result["confidence"] == "88.0%"
        assert result["pass_to_agents"] is True


# ---------------------------------------------------------------------------
# SECTION 1.3: JUDGE MODULE TESTS
# ---------------------------------------------------------------------------
from model.judge import Judge, _clean_user_facing_text

class TestJudgeModule:
    """Unit tests for model/judge.py"""

    def test_parse_response_clean_json(self):
        raw = """{
            "verdict": "CREDIBLE",
            "confidence": "92.5%",
            "justification": "Verified by multiple reputable sources.",
            "reasoning": "Step 1: Check facts. Step 2: Compare with Reuters.",
            "key_facts": ["Fact 1", "Fact 2"]
        }"""
        parsed = Judge._parse_response(raw)
        assert parsed["verdict"] == "CREDIBLE"
        assert parsed["color_class"] == "credible"
        assert parsed["confidence"] == "92.5%"
        assert len(parsed["key_facts"]) == 2

    def test_parse_response_markdown_fences(self):
        raw = """```json
        {
            "verdict": "MISLEADING",
            "confidence": "78.0%",
            "justification": "Headline distorts the study findings.",
            "reasoning": "Step 1: Analyzed text.",
            "key_facts": ["Author misquoted"]
        }
        ```"""
        parsed = Judge._parse_response(raw)
        assert parsed["verdict"] == "MISLEADING CONTENT"
        assert parsed["color_class"] == "misleading"

    def test_parse_response_think_blocks(self):
        raw = """<think>
        The user wants to evaluate whether this news is fake.
        After reviewing claims, I notice fabricated quotes.
        </think>
        {
            "verdict": "FABRICATED",
            "confidence": "95.0%",
            "justification": "The speech never occurred.",
            "reasoning": "Fabricated quotes.",
            "key_facts": ["No official transcript"]
        }"""
        parsed = Judge._parse_response(raw)
        assert parsed["verdict"] == "FABRICATED CONTENT"
        assert parsed["confidence"] == "95.0%"

    def test_parse_response_truncated_json(self):
        # Truncated midway through key_facts
        raw = '{"verdict": "MISLEADING", "confidence": "80.0%", "justification": "Incomplete sentence'
        parsed = Judge._parse_response(raw)
        assert parsed["verdict"] == "MISLEADING CONTENT"
        assert "justification" in parsed

    def test_parse_response_verdict_normalization_fabricated(self):
        raw = '{"verdict": "FABRICATED", "confidence": "90%"}'
        parsed = Judge._parse_response(raw)
        assert parsed["verdict"] == "FABRICATED CONTENT"

    def test_parse_response_verdict_normalization_credible(self):
        raw = '{"verdict": "CREDIBLE", "confidence": "85%"}'
        parsed = Judge._parse_response(raw)
        assert parsed["verdict"] == "CREDIBLE"

    def test_parse_response_verdict_normalization_real_news(self):
        raw = '{"verdict": "REAL NEWS", "confidence": "88%"}'
        parsed = Judge._parse_response(raw)
        assert parsed["verdict"] == "CREDIBLE"

    def test_parse_response_unknown_verdict(self):
        raw = '{"verdict": "BOGUS_UNSEEN_CATEGORY", "confidence": "50%"}'
        parsed = Judge._parse_response(raw)
        assert parsed["verdict"] == "INCONCLUSIVE"
        assert parsed["color_class"] == "inconclusive"

    def test_clean_user_facing_text_mad_acronym(self):
        text = "MAD shows that the report contains internal contradictions. Also the MAD module analyzed bias."
        cleaned = _clean_user_facing_text(text)
        assert "MAD shows" not in cleaned
        assert "Debate shows" in cleaned
        assert "the debate analysis" in cleaned

    def test_clean_user_facing_text_ris_acronym(self):
        text = "RIS indicates that the image was taken in 2018. The RIS module confirmed location."
        cleaned = _clean_user_facing_text(text)
        assert "Reverse Image Search indicates" in cleaned
        assert "Reverse Image Search confirmed" in cleaned

    def test_clean_user_facing_text_kr_acronym(self):
        text = "According to KR, the event happened yesterday. KR shows corroboration."
        cleaned = _clean_user_facing_text(text)
        assert "According to external news reports" in cleaned
        assert "External knowledge retrieval shows" in cleaned

    def test_parse_confidence_float(self):
        assert Judge._parse_confidence_float("87.0%") == 87.0
        assert Judge._parse_confidence_float(" 95.5 % ") == 95.5
        assert Judge._parse_confidence_float("invalid") == 50.0

    def test_aggregate_verdicts_majority_vote(self):
        results = [
            {"verdict": "CREDIBLE", "color_class": "credible", "confidence": "80.0%", "justification": "J1", "reasoning": "R1", "key_facts": ["F1"]},
            {"verdict": "CREDIBLE", "color_class": "credible", "confidence": "90.0%", "justification": "J2", "reasoning": "R2", "key_facts": ["F2"]},
            {"verdict": "FABRICATED CONTENT", "color_class": "fabricated", "confidence": "70.0%", "justification": "J3", "reasoning": "R3", "key_facts": ["F3"]},
        ]
        agg = Judge._aggregate_verdicts(results)
        assert agg["verdict"] == "CREDIBLE", "Majority vote (2 vs 1) should select CREDIBLE"
        assert agg["confidence"] == "85.0%", "Averaged confidence should be (80 + 90) / 2 = 85.0%"
        assert "F1" in agg["key_facts"] and "F2" in agg["key_facts"]
        assert agg["sc_votes"] == {"CREDIBLE": 2, "FABRICATED CONTENT": 1}

    def test_fallback_response(self):
        fallback = Judge._fallback_response("API timeout")
        assert fallback["verdict"] == "INCONCLUSIVE"
        assert fallback["confidence"] == "0.0%"
        assert "API timeout" in fallback["justification"]


# ---------------------------------------------------------------------------
# SECTION 1.4: KNOWLEDGE RETRIEVAL TESTS
# ---------------------------------------------------------------------------
from model.knowledge_retrieval import is_social_media_url, KnowledgeRetriever

class TestKnowledgeRetrieval:
    """Unit tests for model/knowledge_retrieval.py"""

    def test_is_social_media_url_facebook(self):
        assert is_social_media_url("https://facebook.com/posts/9812739") is True
        assert is_social_media_url("https://m.twitter.com/user/status/1234") is True
        assert is_social_media_url("https://www.instagram.com/p/B98xyz") is True

    def test_is_social_media_url_reuters(self):
        assert is_social_media_url("https://www.reuters.com/world/article-slug") is False
        assert is_social_media_url("https://apnews.com/article/politics") is False
        assert is_social_media_url("https://bbc.com/news/world-1234") is False

    def test_filter_domains_mixed_urls(self):
        urls = [
            "https://www.reuters.com/article/tech-update",
            "https://random-unverified-blog.xyz/scam-post",
            "https://apnews.com/article/world-news",
            "https://dubious-clickbait.net/shocking",
            "https://bbc.com/news/science",
        ]
        kr = object.__new__(KnowledgeRetriever)
        filtered = kr.filter_domains(urls)
        assert len(filtered) == 3
        assert "https://www.reuters.com/article/tech-update" in filtered
        assert "https://apnews.com/article/world-news" in filtered
        assert "https://bbc.com/news/science" in filtered
        assert "https://random-unverified-blog.xyz/scam-post" not in filtered

    def test_extract_title_from_url_slug(self):
        kr = object.__new__(KnowledgeRetriever)
        url = "https://www.reuters.com/world/asia-flood-disaster-emergency-aid-2024"
        title = kr.extract_title_from_url_slug(url)
        assert len(title) > 0
        assert "Asia" in title
        assert "Flood" in title
        assert "Disaster" in title

    def test_semantic_filter_embedder_not_loaded(self):
        kr = object.__new__(KnowledgeRetriever)
        kr.embedder = None
        articles = [
            {"title": "Article 1", "text": "This is a paragraph about an economic summit happening in Tokyo."},
            {"title": "Article 2", "text": "Another report detailing inflation figures and trade agreements."},
        ]
        snippets = kr.semantic_filter("Tokyo economic summit", articles)
        assert len(snippets) == 2, "Should return raw text chunks when embedder is None"

    def test_semantic_filter_mocked_similarity(self):
        kr = object.__new__(KnowledgeRetriever)
        kr.similarity_threshold = 0.35
        kr.embedder = MagicMock()
        
        # Mock encode returns
        # claim embedding shape: (1, 4)
        kr.embedder.encode.side_effect = [
            np.array([[0.9, 0.1, 0.0, 0.0]]),  # claim
            np.array([                          # chunks for article 1
                [0.85, 0.15, 0.0, 0.0],         # high similarity
                [0.0, 0.0, 0.9, 0.1],          # low similarity
            ])
        ]
        articles = [{
            "title": "Verified News",
            "text": "First chunk with highly relevant facts.\n\nSecond chunk with totally unrelated information."
        }]
        snippets = kr.semantic_filter("Verified event claim", articles)
        assert len(snippets) == 1, "Only chunks >= 0.35 threshold should pass"
        assert "[Verified News]" in snippets[0]


# ---------------------------------------------------------------------------
# SECTION 1.5: MULTI-AGENT DEBATE TESTS
# ---------------------------------------------------------------------------
from model.mad_agent import DebaterAgent
from model.shared_memory import SharedMemoryBank
from model.mad_orchestrator import OrchestratorAgent

class TestMultiAgentDebate:
    """Unit tests for Multi-Agent Debate module."""

    def test_debater_agent_initialization(self):
        agent = DebaterAgent(
            name="Agent_1",
            side="Real News",
            profile="Investigative Journalist",
            speaker_role="First Speaker"
        )
        assert agent.name == "Agent_1"
        assert agent.side == "Real News"
        assert agent.profile == "Investigative Journalist"
        assert agent.speaker_role == "First Speaker"
        assert agent.llm is not None

    def test_shared_memory_bank_add_and_format(self):
        sm = SharedMemoryBank("Initial news article content", "Context summary from orchestrator")
        sm.add_argument("Round 1: First Speakers", "Agent_1 (Investigative Journalist)", "Real News", "Argument for authenticity.")
        sm.add_argument("Round 1: First Speakers", "Agent_4 (Forensic Data Analyst)", "Fake News", "Argument against authenticity.")

        formatted = sm.get_formatted_history()
        assert "Initial news article content" in formatted
        assert "Context summary from orchestrator" in formatted
        assert "--- Round 1: First Speakers ---" in formatted
        assert "[Real News] Agent_1 (Investigative Journalist): Argument for authenticity." in formatted
        assert "[Fake News] Agent_4 (Forensic Data Analyst): Argument against authenticity." in formatted

    def test_assign_profiles_valid_response(self):
        orch = OrchestratorAgent()
        valid_json = (
            '["Investigative Journalist", "Forensic Data Analyst", "Political Fact-Checker", '
            '"Media Context Analyst", "Public Records Specialist", "Disinformation Researcher"]'
        )
        with patch.object(orch.llm, "generate", return_value=valid_json):
            profiles = orch.assign_profiles("Sample news topic summary")
            assert len(profiles) == 6
            assert "Investigative Journalist" in profiles
            assert "Disinformation Researcher" in profiles

    def test_assign_profiles_llm_failure_fallback(self):
        orch = OrchestratorAgent()
        with patch.object(orch.llm, "generate", side_effect=Exception("API connection refused")):
            profiles = orch.assign_profiles("Sample topic")
            assert len(profiles) == 6
            assert profiles[0] == "Investigative Journalist"
            assert profiles[1] == "Forensic Data Analyst"

    def test_mad_debate_structure_and_rounds(self):
        orch = OrchestratorAgent()
        # Mock LLM generation across all stages to run fast without external dependencies
        with patch.object(orch.llm, "generate", return_value="Mocked analytical argument."), \
             patch.object(DebaterAgent, "generate_argument", return_value="Mocked debate speech."), \
             patch.object(orch, "save_debate_log", return_value=Path("output/mock_log.txt")) as mock_save:

            consolidated, agents_info = orch.run_debate("Article on election outcome", "Verified external knowledge reports")

            # Verify return values
            assert isinstance(consolidated, str)
            assert len(consolidated) > 0
            assert len(agents_info) == 6

            # Verify that save_debate_log received shared_memory with all 3 rounds and 6 total arguments
            assert mock_save.called
            shared_memory_arg = mock_save.call_args[0][0]
            assert isinstance(shared_memory_arg, SharedMemoryBank)

            history = shared_memory_arg.get_formatted_history()
            assert "Round 1: First Speakers" in history
            assert "Round 2: Second Speakers" in history
            assert "Round 3: Third Speakers" in history

            # Check 2 arguments per round (total 6 arguments)
            assert len(shared_memory_arg.argument_history["Round 1: First Speakers"]) == 2
            assert len(shared_memory_arg.argument_history["Round 2: Second Speakers"]) == 2
            assert len(shared_memory_arg.argument_history["Round 3: Third Speakers"]) == 2
