# Reverse Image Search (RIS) Module Complete

The Reverse Image Search module has been successfully built and integrated into the RAHMAD pipeline! Here's a breakdown of the new features.

## What Was Added

### 1. New RIS Component ([reverse_image_search.py](file:///c:/Users/User/Documents/Programming/fyp/RAHMAD/model/reverse_image_search.py))
We created a standalone RIS component containing:
- **`RISFetcher`**: Connects to the SerpApi Google Lens engine to retrieve the top 10 visual matches for the original claim's image URL.
- **`Visual Filter`**: Uses the existing CLIP model (`openai/clip-vit-base-patch32`) to compute the visual cosine similarity between the original image and the retrieved images. Discards images falling below `0.80`.
- **`Semantic Filter`**: Uses `all-MiniLM-L6-v2` to compute semantic similarity between the original claim text and the contextual text surrounding the surviving images. Discards contexts falling below `0.35`.
- **`Visual Agent`**: Packages the surviving high-fidelity multimodal pairings into a prompt and queries OpenRouter (`nvidia/nemotron-3-ultra-550b-a55b:free`). It determines if the visual image has been decontextualized, outputting a strict categorization (`Identical Source` or `Identical Image, Different Context`) and a condensed summary.

### 2. Main Pipeline Integration ([analyzer.py](file:///c:/Users/User/Documents/Programming/fyp/RAHMAD/backend/analyzer.py))
We integrated the RIS pipeline into the main `analyzer.py` execution flow:
- The RIS pipeline executes after the initial `Knowledge Retrieval` step.
- The `analyzer.py` checks if the image provided is an external URL (as required by SerpApi) and triggers the RIS module.
- The findings from the RIS module (the LLM visual summary and categorization) are combined with the standard text-based Knowledge Retrieval summary.
- The final, synthesized output of **both** modules is fed to the top-level **Judge Module** for its final verdict!

### 3. Dependencies Added
- `google-search-results` (SerpApi client library) was added to `requirements.txt` and installed in the backend.

> [!TIP]
> The backend `app.py` has been running in the background for a while. You should restart the backend server so it can pick up the new `.env` variables and `google-search-results` dependency! You can run a test on the frontend using an image URL to see the new RIS logging in action.
