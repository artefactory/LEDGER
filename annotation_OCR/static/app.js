const state = {
    sessionId: window.OCR_ANNOTATION_SESSION_ID
        || new URLSearchParams(window.location.search).get('session')
        || window.OCR_ANNOTATION_DEFAULT_SESSION_ID,
    index: 0,
    itemCount: 0,
    item: null,
    overallStatus: 'unreviewed',
    startedAt: null,
    zoom: 1,
    showingRaw: false,
    showInlineImages: true,
    saving: false,
    prefetchImage: null,
};

const els = {
    sessionTitle: document.getElementById('sessionTitle'),
    sessionMeta: document.getElementById('sessionMeta'),
    progressText: document.getElementById('progressText'),
    progressBar: document.getElementById('progressBar'),
    prevButton: document.getElementById('prevButton'),
    nextButton: document.getElementById('nextButton'),
    skipReviewedButton: document.getElementById('skipReviewedButton'),
    helpButton: document.getElementById('helpButton'),
    imageCanvas: document.getElementById('imageCanvas'),
    rawImage: document.getElementById('rawImage'),
    imageMissing: document.getElementById('imageMissing'),
    imageSubtitle: document.getElementById('imageSubtitle'),
    markdownSubtitle: document.getElementById('markdownSubtitle'),
    markdownPreview: document.getElementById('markdownPreview'),
    rawMarkdown: document.getElementById('rawMarkdown'),
    inlineImagesToggle: document.getElementById('inlineImagesToggle'),
    toggleRawButton: document.getElementById('toggleRawButton'),
    zoomOutButton: document.getElementById('zoomOutButton'),
    zoomResetButton: document.getElementById('zoomResetButton'),
    zoomInButton: document.getElementById('zoomInButton'),
    reportName: document.getElementById('reportName'),
    industryValue: document.getElementById('industryValue'),
    tickerValue: document.getElementById('tickerValue'),
    pageValue: document.getElementById('pageValue'),
    signalsValue: document.getElementById('signalsValue'),
    mappingValue: document.getElementById('mappingValue'),
    notesInput: document.getElementById('notesInput'),
    saveButton: document.getElementById('saveButton'),
    saveStatus: document.getElementById('saveStatus'),
    summaryCsvLink: document.getElementById('summaryCsvLink'),
    summaryMdLink: document.getElementById('summaryMdLink'),
    helpDialog: document.getElementById('helpDialog'),
};

function apiJson(url, options = {}) {
    return fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    }).then(async (response) => {
        if (!response.ok) {
            const text = await response.text();
            throw new Error(text || `${response.status} ${response.statusText}`);
        }
        return response.json();
    });
}

function statusMessage(message, tone = 'neutral') {
    els.saveStatus.textContent = message;
    els.saveStatus.dataset.tone = tone;
}

function formatList(values) {
    if (!values || values.length === 0) return 'none';
    return values.join(', ');
}

function updateProgress(progress) {
    const metadata = progress.metadata || {};
    state.itemCount = progress.item_count || 0;
    els.sessionTitle.textContent = metadata.session_name || metadata.session_id || 'Session';
    els.sessionMeta.textContent = `${metadata.annotator || 'anonymous'} · ${metadata.session_id || state.sessionId}`;
    const reviewed = progress.reviewed_count || 0;
    const total = progress.item_count || 0;
    els.progressText.textContent = `${reviewed} / ${total} reviewed`;
    els.progressBar.style.width = `${total ? Math.round((reviewed / total) * 100) : 0}%`;
    els.summaryCsvLink.href = `/api/session/${state.sessionId}/summary.csv`;
    els.summaryMdLink.href = `/api/session/${state.sessionId}/summary.md`;
}

function setOverall(status) {
    state.overallStatus = status;
    document.querySelectorAll('.status-button').forEach((button) => {
        button.classList.toggle('active', button.dataset.status === status);
    });
}

function loadAnnotation(annotation) {
    setOverall(annotation?.overall_status || 'unreviewed');
    els.notesInput.value = annotation?.notes || '';
}

function fittedImageWidth() {
    const stage = els.imageCanvas.parentElement;
    const availableWidth = Math.max(240, stage.clientWidth - 32);
    const availableHeight = Math.max(240, stage.clientHeight - 32);
    const naturalWidth = els.rawImage.naturalWidth || availableWidth;
    const naturalHeight = els.rawImage.naturalHeight || naturalWidth * 1.414;
    const fitScale = Math.min(availableWidth / naturalWidth, availableHeight / naturalHeight);
    return Math.max(120, Math.floor(naturalWidth * fitScale));
}

function applyZoom() {
    els.imageCanvas.style.setProperty('--image-width', `${Math.round(fittedImageWidth() * state.zoom)}px`);
    els.zoomResetButton.textContent = `${Math.round(state.zoom * 100)}%`;
}

function setZoom(value) {
    state.zoom = Math.min(3, Math.max(0.35, value));
    applyZoom();
}

async function loadProgress() {
    const progress = await apiJson(`/api/session/${state.sessionId}`);
    updateProgress(progress);
    return progress;
}

function prefetchNextImage(url) {
    if (!url) return;
    state.prefetchImage = new Image();
    state.prefetchImage.decoding = 'async';
    state.prefetchImage.src = url;
}

function resetExtractedContentScroll() {
    els.markdownPreview.scrollTop = 0;
    els.rawMarkdown.scrollTop = 0;
}

async function loadItem(index) {
    const safeIndex = Math.max(0, Math.min(index, Math.max(0, state.itemCount - 1)));
    const inlineFlag = state.showInlineImages ? '1' : '0';
    const data = await apiJson(`/api/session/${state.sessionId}/item/${safeIndex}?inline_images=${inlineFlag}`);
    state.index = safeIndex;
    state.item = data.item;
    state.itemCount = data.item_count;
    state.startedAt = new Date();

    els.reportName.textContent = data.item.report_name;
    els.industryValue.textContent = data.item.industry_slug;
    els.tickerValue.textContent = `${data.item.exchange}:${data.item.ticker} · ${data.item.year}`;
    els.pageValue.textContent = `${data.item.page_number} / ${data.item.mmd_page_count}`;
    els.signalsValue.textContent = formatList(data.item.candidate_reasons);
    els.mappingValue.textContent = [data.item.mapping_status, ...data.item.mapping_warnings].filter(Boolean).join(' · ');
    els.imageSubtitle.textContent = data.item.raw_png_path || 'No raw image path';
    els.markdownSubtitle.textContent = `${data.item.page_text_chars} chars · ${data.item.page_text_sha256.slice(0, 12)}`;

    els.markdownPreview.innerHTML = data.markdown_html || '';
    els.rawMarkdown.textContent = data.page_text || '';
    resetExtractedContentScroll();

    if (data.item.raw_png_path) {
        els.rawImage.hidden = false;
        els.imageMissing.hidden = true;
        els.rawImage.src = `${data.image_url}?v=${encodeURIComponent(data.item.page_text_sha256)}`;
        prefetchNextImage(data.next_image_url);
    } else {
        els.rawImage.hidden = true;
        els.imageMissing.hidden = false;
        els.rawImage.removeAttribute('src');
    }

    loadAnnotation(data.annotation);
    setZoom(1);
    statusMessage(`Loaded item ${safeIndex + 1} of ${data.item_count}`);
    els.prevButton.disabled = safeIndex === 0;
    els.nextButton.disabled = safeIndex >= data.item_count - 1;
}

function annotationPayload(source = 'manual') {
    return {
        item_id: state.item.item_id,
        overall_status: state.overallStatus,
        notes: els.notesInput.value,
        annotation_source: source,
        review_duration_ms: state.startedAt ? new Date() - state.startedAt : null,
        client_started_at_utc: state.startedAt ? state.startedAt.toISOString() : null,
        client_updated_at_utc: new Date().toISOString(),
    };
}

async function saveAnnotation(source = 'manual', advance = false) {
    if (!state.item || state.saving) return;
    state.saving = true;
    els.saveButton.disabled = true;
    statusMessage('Saving...');
    try {
        const data = await apiJson(`/api/session/${state.sessionId}/annotation`, {
            method: 'POST',
            body: JSON.stringify(annotationPayload(source)),
        });
        updateProgress(data.progress);
        statusMessage('Saved', 'ok');
        if (advance && state.index < state.itemCount - 1) {
            await loadItem(state.index + 1);
            await loadProgress();
        }
    } catch (error) {
        statusMessage(`Save failed: ${error.message}`, 'error');
    } finally {
        state.saving = false;
        els.saveButton.disabled = false;
    }
}

function quickMark(status, source = 'shortcut') {
    setOverall(status);
    saveAnnotation(`${source}:${status}`, true);
}

async function go(delta) {
    const target = state.index + delta;
    if (target < 0 || target >= state.itemCount) return;
    await loadItem(target);
    await loadProgress();
}

async function goNextOpen() {
    const progress = await loadProgress();
    if (progress.next_unreviewed_index === null || progress.next_unreviewed_index === undefined) {
        statusMessage('No open items');
        return;
    }
    await loadItem(progress.next_unreviewed_index);
}

function toggleRawMarkdown() {
    state.showingRaw = !state.showingRaw;
    els.rawMarkdown.hidden = !state.showingRaw;
    els.markdownPreview.hidden = state.showingRaw;
    els.toggleRawButton.textContent = state.showingRaw ? 'Rendered' : 'Raw Markdown';
}

function inputHasFocus() {
    const active = document.activeElement;
    return active && ['TEXTAREA', 'INPUT', 'SELECT'].includes(active.tagName);
}

function setupEvents() {
    els.prevButton.addEventListener('click', () => go(-1));
    els.nextButton.addEventListener('click', () => go(1));
    els.skipReviewedButton.addEventListener('click', goNextOpen);
    els.saveButton.addEventListener('click', () => saveAnnotation('manual', false));
    els.inlineImagesToggle.addEventListener('change', () => {
        state.showInlineImages = els.inlineImagesToggle.checked;
        loadItem(state.index);
    });
    els.toggleRawButton.addEventListener('click', toggleRawMarkdown);
    els.zoomOutButton.addEventListener('click', () => setZoom(state.zoom - 0.15));
    els.zoomInButton.addEventListener('click', () => setZoom(state.zoom + 0.15));
    els.zoomResetButton.addEventListener('click', () => setZoom(1));
    els.helpButton.addEventListener('click', () => els.helpDialog.showModal());
    els.rawImage.addEventListener('load', () => setZoom(1));
    window.addEventListener('resize', applyZoom);
    document.querySelectorAll('.status-button').forEach((button) => {
        button.addEventListener('click', () => quickMark(button.dataset.status, 'button'));
    });

    document.addEventListener('keydown', (event) => {
        if (inputHasFocus()) return;
        if (event.key === '?') {
            event.preventDefault();
            els.helpDialog.showModal();
        } else if (event.key.toLowerCase() === 'a') {
            event.preventDefault();
            quickMark('ok');
        } else if (event.key.toLowerCase() === 'r') {
            event.preventDefault();
            quickMark('not_ok');
        } else if (event.key.toLowerCase() === 'u') {
            event.preventDefault();
            quickMark('uncertain');
        } else if (event.key === 'ArrowRight' || event.key.toLowerCase() === 'j') {
            event.preventDefault();
            go(1);
        } else if (event.key === 'ArrowLeft' || event.key.toLowerCase() === 'k') {
            event.preventDefault();
            go(-1);
        } else if (event.key === '+' || event.key === '=') {
            event.preventDefault();
            setZoom(state.zoom + 0.15);
        } else if (event.key === '-') {
            event.preventDefault();
            setZoom(state.zoom - 0.15);
        } else if (event.key === '0') {
            event.preventDefault();
            setZoom(1);
        }
    });
}

async function init() {
    setupEvents();
    try {
        const progress = await loadProgress();
        const startIndex = progress.next_unreviewed_index ?? 0;
        if (progress.item_count > 0) {
            await loadItem(startIndex);
        } else {
            statusMessage('Session has no queued items', 'error');
        }
    } catch (error) {
        statusMessage(`Startup failed: ${error.message}`, 'error');
    }
}

init();