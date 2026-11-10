# RAHMAD
Retrieval-Augmented Hybrid Multi-Agent Debate

## Project Structure
```
RAHMAD/
├── app.py              # Flask entry point
├── requirements.txt    # Python dependencies
├── README.md
├── frontend/           # UI assets (HTML, CSS, JS)
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── backend/            # Python backend logic / API modules
│   └── __init__.py
└── model/              # AI model code
    └── __init__.py
```

## Running the App

**Step 1** — Install dependencies:
```powershell
pip install -r requirements.txt
```

**Step 2** — Start the Flask server:
```powershell
python app.py
```

**Step 3** — Open your browser and go to:
```
Run tailscale to connect to port in PC
http://127.0.0.1:5000
```

## Multilingual Support & Cross-Lingual Fallback

RAHMAD natively handles non-English news claims (e.g. Bahasa Melayu, Indonesian) through defense-in-depth architectural mechanisms:
- **Pixel-Level Visual Provenance:** Reverse Image Search (`filter_visually`) operates purely on raw CLIP image features, making visual verification 100% language-agnostic.
- **Automatic Semantic Fallback:** If the text transformer (`all-MiniLM-L6-v2`) produces depressed cosine scores on non-English text, RAHMAD automatically bypasses the threshold (`candidate_contexts = semantically_filtered if semantically_filtered else visually_filtered`) so no visual evidence is lost.
- **Cross-Lingual Reasoning:** Multilingual LLMs (Qwen 2.5 / DeepSeek) in the Visual Agent, Multi-Agent Debate, and Chief Judicial Magistrate evaluate local vernacular claims against global news wire provenance.
- **Regional News Whitelist:** Verified news publishers across Malaysia (`bernama.com`, `astroawani.com`, `malaysiakini.com`, `thestar.com.my`, `nst.com.my`, `theedgemalaysia.com`) and Southeast Asia are pre-whitelisted for Knowledge Retrieval.

