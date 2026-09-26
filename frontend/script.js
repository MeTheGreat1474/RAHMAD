document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const processStream = document.getElementById('process-stream');
    const resultsSection = document.getElementById('results-section');

    // ── Image upload ────────────────────────────────────────────
    const uploadZone = document.getElementById('image-upload');
    const fileInput = document.getElementById('image-file-input');
    const placeholder = document.getElementById('image-placeholder');
    const previewWrap = document.getElementById('image-preview-wrap');
    const previewImg = document.getElementById('image-preview');
    const removeBtn = document.getElementById('image-remove-btn');

    // Mobile elements
    const mobileImgBtn = document.getElementById('mobile-img-btn');
    const mobileFilename = document.getElementById('mobile-filename');
    const mobileRemoveBtn = document.getElementById('mobile-remove-btn');

    function loadImageFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            // Desktop update
            previewImg.src = e.target.result;
            placeholder.hidden = true;
            previewWrap.hidden = false;

            // Mobile update
            if (mobileFilename) mobileFilename.textContent = file.name;
            if (mobileRemoveBtn) mobileRemoveBtn.hidden = false;
        };
        reader.readAsDataURL(file);
    }

    function resetImageUpload() {
        // Desktop reset
        previewImg.src = '';
        fileInput.value = '';
        previewWrap.hidden = true;
        placeholder.hidden = false;

        // Mobile reset
        if (mobileFilename) mobileFilename.textContent = 'Choose image…';
        if (mobileRemoveBtn) mobileRemoveBtn.hidden = true;
    }

    // Click anywhere on the zone to open file picker
    uploadZone.addEventListener('click', (e) => {
        if (e.target === removeBtn) return; // handled below
        fileInput.click();
    });

    // Mobile button trigger
    if (mobileImgBtn) {
        mobileImgBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }

    fileInput.addEventListener('change', () => {
        loadImageFile(fileInput.files[0]);
    });

    // Remove / reset
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetImageUpload();
    });

    if (mobileRemoveBtn) {
        mobileRemoveBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetImageUpload();
        });
    }

    // Drag-and-drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        loadImageFile(file);
    });
    // ────────────────────────────────────────────────────────────

    const streamSteps = [
        { text: "PRE-PROCESSING INPUT...", status: "OK", delay: 500 },
        { text: "RUNNING EARLY PREDICTOR...", status: "OK", delay: 1000 },
        { text: "FETCHING EVIDENCE...", status: "OK", delay: 1000 },
        { text: "SEARCHING FOR SIMILAR IMAGES...", status: "OK", delay: 1000 },
        { text: "DEBATING...", status: "OK", delay: 1000 },
        { text: "SYNTHESIZING RESULTS...", status: "OK", delay: 1000 },
        { text: "OUTPUTTING PREDICTION...", status: "OK", delay: 100 }
    ];

    function getCurrentTime() {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        return `${hours}:${minutes}:${seconds}`;
    }

    function createLogLine(text, status, isFinal = false) {
        const line = document.createElement('div');
        line.className = 'log-line';

        const timeSpan = document.createElement('span');
        timeSpan.className = 'log-time';
        timeSpan.textContent = `[${getCurrentTime()}]`;

        const textSpan = document.createElement('span');
        textSpan.className = 'log-text';
        textSpan.textContent = ` ${text}`;

        const statusSpan = document.createElement('span');
        const displayStatus = isFinal ? status : 'IN PROGRESS';
        const statusClass = displayStatus === 'OK' ? 'log-status--ok'
            : displayStatus === 'IN PROGRESS' ? 'log-status--in-progress'
                : 'log-status--fail';
        statusSpan.className = `log-status ${statusClass}`;
        statusSpan.textContent = displayStatus;

        line.appendChild(timeSpan);
        line.appendChild(textSpan);
        line.appendChild(statusSpan);

        return line;
    }

    async function simulateProcess() {
        // Get input text
        const textInput = document.getElementById('text-input').value;
        
        // Get image if it exists and is visible
        const previewWrap = document.getElementById('image-preview-wrap');
        const imageInput = (!previewWrap.hidden) ? document.getElementById('image-preview').src : null;

        // Reset state
        runBtn.disabled = true;
        runBtn.style.opacity = '0.5';
        runBtn.textContent = 'ANALYZING...';
        processStream.innerHTML = '';
        resultsSection.classList.remove('visible');
        resultsSection.classList.add('hidden');

        // Start the fetch request to the Python backend
        const fetchPromise = fetch('http://127.0.0.1:5000/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: textInput, image: imageInput })
        }).then(res => res.json()).catch(err => {
            console.error('Error fetching from backend:', err);
            return null;
        });

        // We will execute the steps sequentially
        for (let i = 0; i < streamSteps.length; i++) {
            const step = streamSteps[i];

            // Create the line initially with IN PROGRESS
            const line = createLogLine(step.text, step.status, false);
            processStream.appendChild(line);

            // Show the remaining future steps as [ ]
            const futureLines = [];
            for (let j = i + 1; j < streamSteps.length; j++) {
                const fLine = document.createElement('div');
                fLine.className = 'log-line';
                fLine.style.color = 'var(--border-light)';
                fLine.innerHTML = `<span class="log-time">[ ]</span><span class="log-text"> ${streamSteps[j].text}</span>`;
                processStream.appendChild(fLine);
                futureLines.push(fLine);
            }

            // Wait for the simulated delay
            await new Promise(resolve => setTimeout(resolve, step.delay));

            // Update to final status (text + colour class)
            const statusEl = line.querySelector('.log-status');
            statusEl.textContent = step.status;
            statusEl.classList.remove('log-status--in-progress', 'log-status--ok', 'log-status--fail');
            const finalClass = step.status === 'OK' ? 'log-status--ok' : 'log-status--fail';
            statusEl.classList.add(finalClass);

            // Remove future placeholder lines so we can redraw them next tick
            futureLines.forEach(l => l.remove());
        }

        // Wait for the backend to finish if it hasn't already
        const responseData = await fetchPromise;

        // Update DOM with backend data if successful
        if (responseData && !responseData.error) {
            const verdictText = document.getElementById('verdict-text');
            verdictText.innerHTML = responseData.verdict.replace(' ', '<br>');
            verdictText.className = 'verdict-text ' + responseData.color_class;

            const confEl = document.getElementById('confidence-score');
            confEl.textContent = responseData.confidence;
            const confValue = parseFloat(responseData.confidence);
            confEl.className = 'confidence-score ' + (
                confValue < 50  ? 'conf-low' :
                confValue < 70  ? 'conf-mid' :
                                  'conf-high'
            );
            // Helper for HTML escaping and markdown bold conversion
            const formatMarkdownText = (text) => {
                if (!text) return '';
                let formatted = String(text)
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;");
                
                // Convert **text**, __text__, [bold]text[/bold], [b]text[/b], <b>text</b> into <strong>text</strong>
                formatted = formatted.replace(/\*\*([\s\S]*?)\*\*/g, '<strong>$1</strong>');
                formatted = formatted.replace(/__([\s\S]*?)__/g, '<strong>$1</strong>');
                formatted = formatted.replace(/\[bold\]([\s\S]*?)\[\/bold\]/gi, '<strong>$1</strong>');
                formatted = formatted.replace(/\[b\]([\s\S]*?)\[\/b\]/gi, '<strong>$1</strong>');
                formatted = formatted.replace(/&lt;b&gt;([\s\S]*?)&lt;\/b&gt;/gi, '<strong>$1</strong>');
                formatted = formatted.replace(/&lt;strong&gt;([\s\S]*?)&lt;\/strong&gt;/gi, '<strong>$1</strong>');
                return formatted;
            };

            const justificationEl = document.getElementById('justification-text');
            if (justificationEl) {
                justificationEl.innerHTML = formatMarkdownText(responseData.justification);
            }

            // Update key facts
            const keyFactsList = document.getElementById('key-facts-list');
            keyFactsList.innerHTML = '';
            responseData.key_facts.forEach(fact => {
                const li = document.createElement('li');
                let formattedFact = formatMarkdownText(fact);
                // Simple parsing for bold colon parts if they exist (e.g., "Location:") and weren't already bolded
                if (!formattedFact.includes('<strong>')) {
                    const splitIndex = formattedFact.indexOf(':');
                    if (splitIndex !== -1) {
                        formattedFact = `<strong>${formattedFact.substring(0, splitIndex + 1)}</strong>${formattedFact.substring(splitIndex + 1)}`;
                    }
                }
                li.innerHTML = formattedFact;
                keyFactsList.appendChild(li);
            });

            // Update Links
            const linksList = document.getElementById('links-list');
            linksList.innerHTML = '';
            const links = (responseData.links && responseData.links.length > 0)
                ? responseData.links
                : [];

            if (links.length === 0) {
                linksList.innerHTML = `
                    <div class="link-item">
                        <div class="link-number">--</div>
                        <div class="link-details">
                            <span class="link-source">NO EXTERNAL LINKS</span>
                            <span class="link-title" style="opacity: 0.6;">No verified news links found for this query.</span>
                        </div>
                    </div>
                `;
            } else {
                links.forEach((link, index) => {
                    const numStr = String(index + 1).padStart(2, '0');
                    let sourceStr = "NEWS OUTLET";
                    let titleStr = "Article Link";
                    let urlStr = "#";

                    if (typeof link === 'string') {
                        urlStr = link;
                        titleStr = link;
                        try {
                            sourceStr = new URL(link).hostname.replace('www.', '').toUpperCase();
                        } catch(e) {
                            sourceStr = "EXTERNAL SOURCE";
                        }
                    } else if (typeof link === 'object' && link !== null) {
                        sourceStr = link.source || link.domain || "EXTERNAL SOURCE";
                        titleStr = link.title || link.headline || link.url || "Article Link";
                        urlStr = link.url || link.link || "#";

                        if (sourceStr.startsWith('HTTP://') || sourceStr.startsWith('HTTPS://')) {
                            try {
                                sourceStr = new URL(sourceStr).hostname.replace('www.', '').toUpperCase();
                            } catch(e) {}
                        }
                    }

                    const linkItem = document.createElement('div');
                    linkItem.className = 'link-item';
                    if (urlStr && urlStr !== '#') {
                        linkItem.addEventListener('click', () => {
                            window.open(urlStr, '_blank', 'noopener,noreferrer');
                        });
                    }

                    const numDiv = document.createElement('div');
                    numDiv.className = 'link-number';
                    numDiv.textContent = numStr;

                    const detailsDiv = document.createElement('div');
                    detailsDiv.className = 'link-details';

                    const sourceSpan = document.createElement('span');
                    sourceSpan.className = 'link-source';
                    sourceSpan.textContent = sourceStr;

                    const titleLink = document.createElement('a');
                    titleLink.href = urlStr;
                    titleLink.target = '_blank';
                    titleLink.rel = 'noopener noreferrer';
                    titleLink.className = 'link-title';
                    titleLink.textContent = titleStr;
                    titleLink.addEventListener('click', (e) => e.stopPropagation());

                    detailsDiv.appendChild(sourceSpan);
                    detailsDiv.appendChild(titleLink);
                    linkItem.appendChild(numDiv);
                    linkItem.appendChild(detailsDiv);

                    linksList.appendChild(linkItem);
                });
            }

            // Update Reasoning
            const reasoningText = document.getElementById('reasoning-text');
            if (reasoningText) {
                let rawReasoning = responseData.reasoning || '';
                
                // Format markdown bold first
                let formattedReasoning = formatMarkdownText(rawReasoning);
                    
                // Format "STEP X" or "STEP X — SOMETHING" as headers
                // This regex finds STEP followed by a number, captures the title up to a colon or newline, and consumes the colon/spacing
                formattedReasoning = formattedReasoning.replace(/(STEP\s+\d+[^:\n]*?)(?::\s*|\n|$)/gi, function(match, p1) {
                    return `<h4 style="margin-top: 18px; margin-bottom: 6px; font-size: 0.95rem; color: var(--text-primary); font-weight: 600; border-bottom: 1px solid var(--border-light); padding-bottom: 4px;">${p1}</h4>`;
                });
                
                // Convert remaining newlines to <br> to preserve paragraphs
                formattedReasoning = formattedReasoning.replace(/\n/g, '<br>');
                
                // Clean up any stray <br> tags immediately following a heading
                formattedReasoning = formattedReasoning.replace(/<\/h4><br>/g, '</h4>');
                
                reasoningText.innerHTML = formattedReasoning;
                // Reset accordion state on each new result
                const body = document.getElementById('reasoning-body');
                const toggle = document.getElementById('reasoning-toggle');
                if (body && toggle) {
                    body.classList.remove('open');
                    toggle.setAttribute('aria-expanded', 'false');
                }
            }
        }

        // Finish process
        runBtn.disabled = false;
        runBtn.style.opacity = '1';
        runBtn.textContent = 'RUN NEWS DETECTION';

        // Reveal results
        resultsSection.classList.remove('hidden');
        resultsSection.classList.add('visible');
    }

    runBtn.addEventListener('click', simulateProcess);

    // Reasoning accordion toggle
    const reasoningToggle = document.getElementById('reasoning-toggle');
    const reasoningBody = document.getElementById('reasoning-body');
    if (reasoningToggle && reasoningBody) {
        reasoningToggle.addEventListener('click', () => {
            const isOpen = reasoningBody.classList.toggle('open');
            reasoningToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        });
    }
});
