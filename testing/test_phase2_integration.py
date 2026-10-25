"""
testing/test_phase2_integration.py

Phase 2 Integration Tests for RAHMAD Fake News Detection System.
Covers:
  - 2.1 Pipeline Flow Integration (backend/analyzer.py)
      * Full pipeline flow (text + image) across all 6 stages
      * Text-only pipeline flow (no image provided, graceful skips)
      * Short text guard rejection (< 10 chars)
      * Empty / whitespace text guard rejection
      * Invalid / corrupted image data graceful fallback
      * Judge execution mode: single (JUDGE_MODE=single)
      * Judge execution mode: panel (JUDGE_MODE=panel)
      * Judge execution mode: self_consistency (JUDGE_MODE=self_consistency)
      * Knowledge Retrieval (KR) failure resilience
      * Reverse Image Search (RIS) failure resilience
      * Multi-Agent Debate (MAD) failure resilience
      * Early Predictor (EP) failure resilience
      * Judge module unhandled exception resilience
      * Image decoding (data URI and raw formats)
      * Link formatting and domain extraction from KR sources
      * Structured pipeline logger integration
  - 2.2 Flask API Endpoint Integration (app.py)
      * POST /api/analyze valid multimodal request (200, NDJSON stream)
      * POST /api/analyze valid text-only request (200, NDJSON stream)
      * POST /api/analyze missing text field (400 Bad Request)
      * POST /api/analyze empty JSON payload (400 Bad Request)
      * POST /api/analyze non-JSON payload (400 Bad Request)
      * GET / frontend dashboard index route (200 HTML)
      * GET /res/<filename> existing static resource (200 OK)
      * GET /res/<filename> nonexistent resource (404 Not Found)
      * Cross-Origin Resource Sharing (CORS) header verification
"""

import base64
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from backend.analyzer import analyze_text, _decode_image, _NO_IMAGE_EP


# ---------------------------------------------------------------------------
# Helper fixtures & test utilities
# ---------------------------------------------------------------------------

def generate_sample_base64_png(width: int = 64, height: int = 64, color: tuple = (255, 0, 0)) -> str:
    """Generate a valid base64 data-URL string for a PNG image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{raw_b64}"


SAMPLE_ARTICLE_TEXT = (
    "GLOBAL HEALTH SUMMIT ANNOUNCES BREAKTHROUGH VACCINE STRATEGY\n\n"
    "Health authorities from over forty countries concluded a landmark three-day summit today, "
    "unveiling a multilateral framework for rapid pandemic response and medical distribution.\n\n"
    "Independent observers and international delegates praised the cooperative consensus, "
    "affirming adherence to stringent scientific peer-review standards."
)

SAMPLE_KR_ARTICLES = [
    {
        "headline": "Global Health Delegates Reach Agreement on Pandemic Protocol",
        "source": "https://www.reuters.com/world/health-summit-2026-agreement",
        "url": "https://www.reuters.com/world/health-summit-2026-agreement",
        "similarity": 0.88,
    },
    {
        "headline": "International Medical Accord Finalized in Geneva",
        "source": "https://apnews.com/article/geneva-health-accord-101",
        "url": "https://apnews.com/article/geneva-health-accord-101",
        "similarity": 0.81,
    },
]

SAMPLE_EP_RESULT = {
    "probability": 0.12,
    "label": "REAL",
    "confidence": "88.0%",
    "pass_to_agents": False,
    "color_class": "real",
}

SAMPLE_JUDGE_RESULT = {
    "verdict": "CREDIBLE",
    "color_class": "credible",
    "confidence": "92.0%",
    "justification": "The claim is substantiated by multiple independent international news agencies.",
    "key_facts": [
        "Forty nations ratified the public health accord.",
        "Consensus corroborated by Reuters and Associated Press reports."
    ],
    "reasoning": "External reports align with the submitted text; no manipulation detected.",
}


@pytest.fixture
def mock_pipeline_components():
    """
    Provide mocked pipeline components for deterministic integration testing.
    Patches module-level getters in backend.analyzer.
    """
    mock_ep = MagicMock()
    mock_ep.predict_from_tensors_verbose.return_value = (
        SAMPLE_EP_RESULT,
        torch.zeros((1, 512)),
        torch.zeros((1, 512)),
        torch.zeros((1, 1024)),
    )

    mock_kr = MagicMock()
    mock_kr.retrieve_detailed.return_value = {
        "claim": SAMPLE_ARTICLE_TEXT[:300],
        "articles": SAMPLE_KR_ARTICLES,
        "summary": "Verified through corroborating international news sources.",
    }

    mock_ris = MagicMock()
    mock_ris.execute.return_value = {
        "status": "success",
        "category": "authentic",
        "is_out_of_context": False,
        "confidence": "High",
        "judge_summary": "Original photograph matches official press briefings.",
    }
    mock_ris.log_skipped = MagicMock()

    mock_mad = MagicMock()
    mock_mad.run_debate.return_value = (
        "Consensus reached: The facts presented align with official documentation.",
        [
            {"name": "Agent_1", "side": "Real News", "profile": "Investigative Journalist"},
            {"name": "Agent_2", "side": "Fake News", "profile": "Forensic Data Analyst"},
        ],
    )

    mock_judge = MagicMock()
    mock_judge.analyze.return_value = dict(SAMPLE_JUDGE_RESULT)
    mock_judge.panel_analyze.return_value = dict(SAMPLE_JUDGE_RESULT)
    mock_judge.self_consistency_analyze.return_value = dict(SAMPLE_JUDGE_RESULT)
    mock_judge.last_user_message = "Mocked User Message"
    mock_judge.last_raw_response = json.dumps(SAMPLE_JUDGE_RESULT)
    mock_judge.save_judge_log = MagicMock()

    with patch.dict(os.environ, {"JUDGE_MODE": "single"}), \
         patch("backend.analyzer._get_predictor", return_value=mock_ep), \
         patch("backend.analyzer._get_retriever", return_value=mock_kr), \
         patch("backend.analyzer._get_ris_pipeline", return_value=mock_ris), \
         patch("backend.analyzer._get_mad_orchestrator", return_value=mock_mad), \
         patch("backend.analyzer._get_judge", return_value=mock_judge):
        yield {
            "ep": mock_ep,
            "kr": mock_kr,
            "ris": mock_ris,
            "mad": mock_mad,
            "judge": mock_judge,
        }


# ===========================================================================
# SECTION 2.1: PIPELINE FLOW INTEGRATION TESTS
# ===========================================================================

class TestPipelineFlowIntegration:
    """Integration tests verifying end-to-end dataflow through analyzer.py."""

    def test_full_pipeline_flow_all_modules_success(self, mock_pipeline_components):
        """
        Verify that a complete input with text and image triggers all 6 pipeline stages
        and returns a well-formed structured result.
        """
        b64_image = generate_sample_base64_png(width=100, height=100)
        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, b64_image))

        # Separate status events from result event
        status_events = [c for c in chunks if c.get("type") == "status"]
        result_events = [c for c in chunks if c.get("type") == "result"]

        assert len(result_events) == 1, "Pipeline must yield exactly one final result"
        final_result = result_events[0]["data"]

        # Verify all expected status steps were reported
        expected_steps = [
            "PRE-PROCESSING INPUT...",
            "RUNNING EARLY PREDICTOR...",
            "FETCHING EVIDENCE...",
            "SEARCHING FOR SIMILAR IMAGES...",
            "DEBATING...",
            "SYNTHESIZING RESULTS...",
            "OUTPUTTING PREDICTION...",
        ]
        reported_steps = [event["step"] for event in status_events]
        for step in expected_steps:
            assert step in reported_steps, f"Pipeline stream missing status step: {step}"

        # Verify OK transitions
        ok_steps = [e["step"] for e in status_events if e.get("status") == "OK"]
        for step in expected_steps:
            assert step in ok_steps, f"Step did not reach 'OK' state: {step}"

        # Verify result schema
        assert final_result["verdict"] == "CREDIBLE"
        assert final_result["color_class"] == "credible"
        assert final_result["confidence"] == "92.0%"
        assert isinstance(final_result["key_facts"], list)
        assert len(final_result["key_facts"]) > 0
        assert "ep_signal" in final_result
        assert final_result["ep_signal"]["label"] == "REAL"
        assert final_result["ep_signal"]["probability"] == 0.12

        # Verify links populated from Knowledge Retrieval
        assert isinstance(final_result["links"], list)
        assert len(final_result["links"]) == 2
        assert final_result["links"][0]["source"] == "REUTERS.COM"
        assert final_result["links"][1]["source"] == "APNEWS.COM"

    def test_pipeline_text_only_no_image(self, mock_pipeline_components):
        """
        Verify that when no image is provided, Early Predictor and RIS display FAIL
        cleanly without hard crashing, and the Judge still receives appropriate neutral signals.
        """
        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
        status_events = [c for c in chunks if c.get("type") == "status"]
        result_events = [c for c in chunks if c.get("type") == "result"]

        assert len(result_events) == 1
        final_result = result_events[0]["data"]

        # EP and RIS must be marked as FAIL in status updates when no image is provided
        failed_steps = [e["step"] for e in status_events if e.get("status") == "FAIL"]
        assert "RUNNING EARLY PREDICTOR..." in failed_steps
        assert "SEARCHING FOR SIMILAR IMAGES..." in failed_steps

        # Judge must have been called with the neutral _NO_IMAGE_EP signal
        mock_judge = mock_pipeline_components["judge"]
        mock_judge.analyze.assert_called_once()
        called_ep = mock_judge.analyze.call_args.args[1]
        assert called_ep["label"] == "UNKNOWN"
        assert called_ep["probability"] == 0.5
        assert "no image provided" in called_ep["note"].lower()

        # Final result should be valid
        assert final_result["verdict"] == "CREDIBLE"
        assert final_result["ep_signal"]["label"] == "UNKNOWN"

    def test_pipeline_short_text_rejection(self, mock_pipeline_components):
        """
        Verify that input text shorter than 10 characters is immediately rejected
        as INCONCLUSIVE without invoking any heavy downstream modules.
        """
        short_inputs = ["Too short", "Hi", "123456789"]
        for short_text in short_inputs:
            chunks = list(analyze_text(short_text, image_data=None))
            assert len(chunks) == 1, "Short input must yield only the single immediate result"
            result = chunks[0]["data"]
            assert result["verdict"] == "INCONCLUSIVE"
            assert result["color_class"] == "inconclusive"
            assert result["confidence"] == "0.0%"
            assert "too short" in result["justification"].lower()

        # Verify no heavy modules were executed
        mock_pipeline_components["ep"].predict_from_tensors_verbose.assert_not_called()
        mock_pipeline_components["kr"].retrieve_detailed.assert_not_called()
        mock_pipeline_components["ris"].execute.assert_not_called()
        mock_pipeline_components["mad"].run_debate.assert_not_called()
        mock_pipeline_components["judge"].analyze.assert_not_called()

    def test_pipeline_empty_or_whitespace_text_rejection(self, mock_pipeline_components):
        """Verify empty and whitespace-only text inputs are immediately rejected."""
        whitespace_inputs = ["", "   ", "\n\t  \n"]
        for text in whitespace_inputs:
            chunks = list(analyze_text(text, image_data=None))
            assert len(chunks) == 1
            assert chunks[0]["data"]["verdict"] == "INCONCLUSIVE"

        mock_pipeline_components["judge"].analyze.assert_not_called()

    def test_pipeline_invalid_image_data_graceful_fallback(self, mock_pipeline_components):
        """
        Verify that malformed or unsupported image formats trigger graceful fallback
        to text-only analysis without crashing the pipeline.
        """
        malformed_image_str = "not_a_valid_data_url_or_http"
        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=malformed_image_str))

        result_events = [c for c in chunks if c.get("type") == "result"]
        assert len(result_events) == 1
        assert result_events[0]["data"]["verdict"] == "CREDIBLE"

        # EP should have been skipped due to invalid image
        mock_pipeline_components["ep"].predict_from_tensors_verbose.assert_not_called()

    def test_pipeline_judge_mode_single(self, mock_pipeline_components, monkeypatch):
        """Verify JUDGE_MODE='single' invokes standard Judge.analyze()."""
        monkeypatch.setenv("JUDGE_MODE", "single")
        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
        assert len(chunks) > 0
        mock_pipeline_components["judge"].analyze.assert_called_once()
        mock_pipeline_components["judge"].panel_analyze.assert_not_called()
        mock_pipeline_components["judge"].self_consistency_analyze.assert_not_called()

    def test_pipeline_judge_mode_panel(self, mock_pipeline_components, monkeypatch):
        """Verify JUDGE_MODE='panel' invokes Judge.panel_analyze()."""
        monkeypatch.setenv("JUDGE_MODE", "panel")
        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
        assert len(chunks) > 0
        mock_pipeline_components["judge"].panel_analyze.assert_called_once()
        mock_pipeline_components["judge"].analyze.assert_not_called()
        mock_pipeline_components["judge"].self_consistency_analyze.assert_not_called()

    def test_pipeline_judge_mode_self_consistency(self, mock_pipeline_components, monkeypatch):
        """Verify JUDGE_MODE='self_consistency' invokes Judge.self_consistency_analyze() with n=3."""
        monkeypatch.setenv("JUDGE_MODE", "self_consistency")
        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
        assert len(chunks) > 0
        mock_pipeline_components["judge"].self_consistency_analyze.assert_called_once()
        mock_pipeline_components["judge"].analyze.assert_not_called()
        mock_pipeline_components["judge"].panel_analyze.assert_not_called()
        args, kwargs = mock_pipeline_components["judge"].self_consistency_analyze.call_args
        assert kwargs.get("n") == 3

    def test_pipeline_kr_failure_resilience(self, mock_pipeline_components):
        """
        Verify that if Knowledge Retrieval raises an exception, the pipeline marks
        the step as FAIL, passes fallback context to the Judge, and finishes successfully.
        """
        mock_pipeline_components["kr"].retrieve_detailed.side_effect = ConnectionError("Search API timeout")

        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
        status_events = [c for c in chunks if c.get("type") == "status"]
        result_events = [c for c in chunks if c.get("type") == "result"]

        assert len(result_events) == 1
        assert result_events[0]["data"]["verdict"] == "CREDIBLE"

        # Evidence fetching step should be marked FAIL
        failed_steps = [e["step"] for e in status_events if e.get("status") == "FAIL"]
        assert "FETCHING EVIDENCE..." in failed_steps

        # Judge must still execute
        mock_pipeline_components["judge"].analyze.assert_called_once()

    def test_pipeline_ris_failure_resilience(self, mock_pipeline_components):
        """
        Verify that if Reverse Image Search raises an exception, the pipeline marks
        the step as FAIL, informs the Judge, and finishes successfully.
        """
        b64_image = generate_sample_base64_png()
        mock_pipeline_components["ris"].execute.side_effect = RuntimeError("SerpAPI quota exceeded")

        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=b64_image))
        status_events = [c for c in chunks if c.get("type") == "status"]
        result_events = [c for c in chunks if c.get("type") == "result"]

        assert len(result_events) == 1
        assert result_events[0]["data"]["verdict"] == "CREDIBLE"

        # Image search step should be marked FAIL
        failed_steps = [e["step"] for e in status_events if e.get("status") == "FAIL"]
        assert "SEARCHING FOR SIMILAR IMAGES..." in failed_steps

        # Judge receives error notification
        mock_judge = mock_pipeline_components["judge"]
        called_ris = mock_judge.analyze.call_args.kwargs.get("ris_summary", "")
        assert "encountered an error" in called_ris

    def test_pipeline_mad_failure_resilience(self, mock_pipeline_components):
        """
        Verify that if Multi-Agent Debate raises an exception, the pipeline marks
        the step as FAIL, passes fallback context, and finishes successfully.
        """
        mock_pipeline_components["mad"].run_debate.side_effect = TimeoutError("Local LLM inference timeout")

        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
        status_events = [c for c in chunks if c.get("type") == "status"]
        result_events = [c for c in chunks if c.get("type") == "result"]

        assert len(result_events) == 1
        assert result_events[0]["data"]["verdict"] == "CREDIBLE"

        # Debating step should be marked FAIL
        failed_steps = [e["step"] for e in status_events if e.get("status") == "FAIL"]
        assert "DEBATING..." in failed_steps

    def test_pipeline_early_predictor_failure_resilience(self, mock_pipeline_components):
        """
        Verify that if Early Predictor crashes during inference, the pipeline marks
        the step as FAIL, falls back to neutral EP signal, and continues.
        """
        b64_image = generate_sample_base64_png()
        mock_pipeline_components["ep"].predict_from_tensors_verbose.side_effect = RuntimeError("CUDA OOM")

        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=b64_image))
        status_events = [c for c in chunks if c.get("type") == "status"]
        result_events = [c for c in chunks if c.get("type") == "result"]

        assert len(result_events) == 1
        failed_steps = [e["step"] for e in status_events if e.get("status") == "FAIL"]
        assert "RUNNING EARLY PREDICTOR..." in failed_steps

        # Judge still receives a neutral EP signal
        mock_judge = mock_pipeline_components["judge"]
        mock_judge.analyze.assert_called_once()

    def test_pipeline_judge_unhandled_exception_fallback(self, mock_pipeline_components):
        """
        Verify that if the Judge module raises an unexpected exception, the pipeline
        catches it gracefully and returns an INCONCLUSIVE error verdict.
        """
        mock_pipeline_components["judge"].analyze.side_effect = Exception("OpenRouter authentication error")
        mock_pipeline_components["judge"].panel_analyze.side_effect = Exception("OpenRouter authentication error")
        mock_pipeline_components["judge"].self_consistency_analyze.side_effect = Exception("OpenRouter authentication error")

        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
        result_events = [c for c in chunks if c.get("type") == "result"]

        assert len(result_events) == 1
        res = result_events[0]["data"]
        assert res["verdict"] == "INCONCLUSIVE"
        assert res["confidence"] == "0.0%"
        assert "unexpected error" in res["justification"].lower()

    def test_decode_image_valid_base64(self):
        """Verify _decode_image successfully parses valid base64 PNG data."""
        b64 = generate_sample_base64_png(width=50, height=50, color=(0, 255, 0))
        img = _decode_image(b64)
        assert isinstance(img, Image.Image)
        assert img.size == (50, 50)
        assert img.mode == "RGB"

    def test_decode_image_empty_and_corrupt(self):
        """Verify _decode_image returns None on empty or corrupted input."""
        assert _decode_image(None) is None
        assert _decode_image("") is None
        assert _decode_image("data:image/png;base64,invalid_corrupt_data!!!") is None
        assert _decode_image("invalid_format_string") is None

    def test_pipeline_link_formatting_and_domain_parsing(self, mock_pipeline_components):
        """Verify that external URLs in KR are converted into clean upper-case domains."""
        custom_articles = [
            {"source": "https://www.bbc.com/news/article-99", "title": "BBC News Report"},
            {"source": "http://nytimes.com/world/update", "title": "NYT Update"},
            {"source": "https://theguardian.com/world/accord", "title": "Guardian Accord"},
            {"source": "ftp://custom.org/resource", "title": "Custom Org"},
            {"source": "non_url_source", "title": "Raw Source"},
        ]
        mock_pipeline_components["kr"].retrieve_detailed.return_value = {
            "claim": "Test claim",
            "articles": custom_articles,
            "summary": "Summary",
        }

        chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
        result = [c for c in chunks if c.get("type") == "result"][0]["data"]
        links = result["links"]

        assert len(links) == 5
        assert links[0]["source"] == "BBC.COM"
        assert links[1]["source"] == "NYTIMES.COM"
        assert links[2]["source"] == "THEGUARDIAN.COM"
        assert links[3]["source"] == "EXTERNAL SOURCE"
        assert links[4]["source"] == "EXTERNAL SOURCE"

    def test_pipeline_logging_execution(self, mock_pipeline_components):
        """Verify that PipelineLogger and Judge log mechanisms are invoked during analysis."""
        with patch("backend.pipeline_logger.PipelineLogger.save") as mock_log_save:
            chunks = list(analyze_text(SAMPLE_ARTICLE_TEXT, image_data=None))
            assert len(chunks) > 0
            mock_log_save.assert_called_once()
            mock_pipeline_components["judge"].save_judge_log.assert_called_once()


# ===========================================================================
# SECTION 2.2: FLASK API ENDPOINT INTEGRATION TESTS (app.py)
# ===========================================================================

class TestApiEndpointsIntegration:
    """Integration tests verifying HTTP endpoints, headers, and NDJSON streaming."""

    @pytest.fixture
    def client(self):
        """Create a Flask test client with testing mode enabled."""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_api_analyze_valid_multimodal_request(self, client, mock_pipeline_components):
        """
        Verify POST /api/analyze with text + image payload returns HTTP 200,
        streaming application/x-ndjson, status events, and final result.
        """
        b64_image = generate_sample_base64_png(width=80, height=80)
        payload = {
            "text": SAMPLE_ARTICLE_TEXT,
            "image": b64_image,
        }

        response = client.post(
            "/api/analyze",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers["Content-Type"]

        # Parse NDJSON lines
        lines = [line.strip() for line in response.get_data(as_text=True).split("\n") if line.strip()]
        assert len(lines) > 0, "Response must contain NDJSON lines"

        parsed_items = [json.loads(line) for line in lines]
        types = [item.get("type") for item in parsed_items]
        assert "status" in types
        assert "result" in types

        result_item = [item for item in parsed_items if item.get("type") == "result"][0]
        assert result_item["data"]["verdict"] == "CREDIBLE"
        assert result_item["data"]["confidence"] == "92.0%"

    def test_api_analyze_valid_text_only_request(self, client, mock_pipeline_components):
        """Verify POST /api/analyze with text-only payload runs successfully."""
        payload = {"text": SAMPLE_ARTICLE_TEXT}

        response = client.post(
            "/api/analyze",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers["Content-Type"]

        lines = [line.strip() for line in response.get_data(as_text=True).split("\n") if line.strip()]
        parsed_items = [json.loads(line) for line in lines]
        result_item = [item for item in parsed_items if item.get("type") == "result"][0]
        assert result_item["data"]["verdict"] == "CREDIBLE"

    def test_api_analyze_missing_text_400(self, client):
        """Verify POST /api/analyze without 'text' key returns HTTP 400."""
        b64_image = generate_sample_base64_png()
        payload = {"image": b64_image}

        response = client.post(
            "/api/analyze",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data == {"error": "No text provided"}

    def test_api_analyze_empty_body_400(self, client):
        """Verify POST /api/analyze with empty JSON object returns HTTP 400."""
        response = client.post(
            "/api/analyze",
            data=json.dumps({}),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data == {"error": "No text provided"}

    def test_api_analyze_non_json_payload_400(self, client):
        """Verify POST /api/analyze with non-JSON body returns HTTP 400 or 415 client error."""
        response = client.post(
            "/api/analyze",
            data="Not valid json string",
            content_type="text/plain",
        )
        assert response.status_code in (400, 415)

        # Also test malformed JSON with application/json
        malformed_response = client.post(
            "/api/analyze",
            data="{invalid_json_string",
            content_type="application/json",
        )
        assert malformed_response.status_code == 400

    def test_frontend_serve_index_html(self, client):
        """Verify GET / serves frontend index.html with HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
        html_content = response.get_data(as_text=True)
        assert "<title>" in html_content
        assert "RAHMAD" in html_content

    def test_res_serve_existing_file(self, client):
        """Verify GET /res/<filename> serves existing static files from res/."""
        response = client.get("/res/news1.txt")
        assert response.status_code == 200
        content = response.get_data(as_text=True)
        assert len(content) > 0

    def test_res_serve_nonexistent_file_404(self, client):
        """Verify GET /res/<filename> returns 404 for nonexistent files."""
        response = client.get("/res/file_does_not_exist_987654321.xyz")
        assert response.status_code == 404

    def test_cors_headers_support(self, client):
        """Verify that Flask-CORS headers are active on responses."""
        response = client.get("/")
        # Flask-CORS adds Access-Control-Allow-Origin header
        assert "Access-Control-Allow-Origin" in response.headers
