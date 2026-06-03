const state = {
    sessionId: window.QRELS_ANNOTATION_SESSION_ID
        || new URLSearchParams(window.location.search).get('session')
        || window.QRELS_ANNOTATION_DEFAULT_SESSION_ID,
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

const IMAGE_STAGE_PADDING = 16;

const els = {
    sessionTitle: document.getElementById('sessionTitle'),
    sessionMeta: document.getElementById('sessionMeta'),
    progressText: document.getElementById('progressText'),
    progressBar: document.getElementById('progressBar'),
    prevButton: document.getElementById('prevButton'),
    nextButton: document.getElementById('nextButton'),
    skipReviewedButton: document.getElementById('skipReviewedButton'),
    helpButton: document.getElementById('helpButton'),
    imageStage: document.getElementById('imageStage'),
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
    kpiName: document.getElementById('kpiName'),
    kpiTargetValue: document.getElementById('kpiTargetValue'),
    kpiExactValue: document.getElementById('kpiExactValue'),
    kpiQuickRef: document.getElementById('kpiQuickRef'),
    kpiMatchType: document.getElementById('kpiMatchType'),
    kpiAliasMatched: document.getElementById('kpiAliasMatched'),
    kpiRawValue: document.getElementById('kpiRawValue'),
    kpiRelError: document.getElementById('kpiRelError'),
    kpiUnitSource: document.getElementById('kpiUnitSource'),
    kpiSnippet: document.getElementById('kpiSnippet'),
    reportName: document.getElementById('reportName'),
    queryId: document.getElementById('queryId'),
    industryValue: document.getElementById('industryValue'),
    tickerValue: document.getElementById('tickerValue'),
    pageValue: document.getElementById('pageValue'),
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

function updateProgress(progress) {
    const metadata = progress.metadata || {};
    state.itemCount = progress.item_count || 0;
    els.sessionTitle.textContent = metadata.session_name || metadata.session_id || 'Session';
    els.sessionMeta.textContent = `${metadata.annotator || 'anonymous'} \u00b7 ${metadata.session_id || state.sessionId}`;
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

function baseFittedImageSize() {
    const width = fittedImageWidth();
    const naturalWidth = els.rawImage.naturalWidth || 1;
    const naturalHeight = els.rawImage.naturalHeight || Math.max(1, naturalWidth * 1.414);
    return {
        width,
        height: width * (naturalHeight / naturalWidth),
    };
}

function scaledImageSize() {
    const baseSize = baseFittedImageSize();
    return {
        width: baseSize.width * state.zoom,
        height: baseSize.height * state.zoom,
    };
}

function imagePlacement() {
    const { width, height } = scaledImageSize();
    const stageWidth = Math.max(1, els.imageStage.clientWidth);
    const stageHeight = Math.max(1, els.imageStage.clientHeight);
    const paddedWidth = width + (IMAGE_STAGE_PADDING * 2);
    const paddedHeight = height + (IMAGE_STAGE_PADDING * 2);
    const canvasWidth = Math.max(stageWidth, paddedWidth);
    const canvasHeight = Math.max(stageHeight, paddedHeight);

    return {
        width,
        height,
        canvasWidth,
        canvasHeight,
        left: IMAGE_STAGE_PADDING + Math.max(0, (canvasWidth - paddedWidth) / 2),
        top: IMAGE_STAGE_PADDING + Math.max(0, (canvasHeight - paddedHeight) / 2),
    };
}

function applyZoom() {
    const placement = imagePlacement();
    els.imageCanvas.style.setProperty('--canvas-width', `${Math.round(placement.canvasWidth)}px`);
    els.imageCanvas.style.setProperty('--canvas-height', `${Math.round(placement.canvasHeight)}px`);
    els.imageCanvas.style.setProperty('--image-left', `${Math.round(placement.left)}px`);
    els.imageCanvas.style.setProperty('--image-top', `${Math.round(placement.top)}px`);
    els.imageCanvas.style.setProperty('--image-width', `${Math.round(placement.width)}px`);
    els.imageCanvas.style.setProperty('--image-height', `${Math.round(placement.height)}px`);
    els.zoomResetButton.textContent = `${Math.round(state.zoom * 100)}%`;
}

function setZoom(value) {
    state.zoom = Math.min(3, Math.max(0.35, value));
    applyZoom();
}

function scheduleAfterLayout(callback) {
    window.requestAnimationFrame(() => {
        callback();
    });
}

function viewportCenterPoint() {
    if (!els.rawImage.naturalWidth || !els.rawImage.naturalHeight) {
        return { x: 0.5, y: 0.5 };
    }
    const placement = imagePlacement();
    return {
        x: (els.imageStage.scrollLeft + (els.imageStage.clientWidth / 2) - placement.left) / placement.width,
        y: (els.imageStage.scrollTop + (els.imageStage.clientHeight / 2) - placement.top) / placement.height,
    };
}

function centerViewportOnPoint(point) {
    if (!point || !els.rawImage.naturalWidth || !els.rawImage.naturalHeight) return;
    const placement = imagePlacement();
    els.imageStage.scrollLeft = Math.max(
        0,
        placement.left + (point.x * placement.width) - (els.imageStage.clientWidth / 2),
    );
    els.imageStage.scrollTop = Math.max(
        0,
        placement.top + (point.y * placement.height) - (els.imageStage.clientHeight / 2),
    );
}

function zoomAroundPoint(value, point) {
    setZoom(value);
    scheduleAfterLayout(() => centerViewportOnPoint(point));
}

function adjustZoom(delta) {
    const nextZoom = state.zoom + delta;
    const anchorPoint = viewportCenterPoint();
    zoomAroundPoint(nextZoom, anchorPoint);
}

function focusCurrentItem({ resetZoom = true } = {}) {
    if (resetZoom) setZoom(1);
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

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

function formatExactValue(value) {
    if (value === null || value === undefined) return '';
    const absVal = Math.abs(value);
    const sign = value < 0 ? '-' : '';
    if (absVal >= 1e9) {
        return `${sign}$${(absVal / 1e9).toFixed(3).replace(/\.?0+$/, '')}B (${absVal.toLocaleString('en-US')})`;
    }
    if (absVal >= 1e6) {
        return `${sign}$${(absVal / 1e6).toFixed(3).replace(/\.?0+$/, '')}M (${absVal.toLocaleString('en-US')})`;
    }
    if (absVal >= 1e3) {
        return `${sign}$${(absVal / 1e3).toFixed(3).replace(/\.?0+$/, '')}K (${absVal.toLocaleString('en-US')})`;
    }
    return `${sign}$${absVal.toLocaleString('en-US')}`;
}

function renderKpiContext(item) {
    const kpiName = (item.kpi || '').replace(/_/g, ' ');
    els.kpiName.textContent = kpiName || '-';
    els.kpiTargetValue.textContent = item.target_value_display || '-';
    els.kpiExactValue.textContent = formatExactValue(item.target_value);
    els.kpiQuickRef.textContent = item.target_value_display
        ? `Find ${item.target_value_display} in ${item.exchange || ''}:${item.ticker || ''} ${item.year || ''}`
        : '';
    els.kpiMatchType.textContent = item.match_type || '-';
    els.kpiAliasMatched.textContent = item.alias_matched || '-';
    els.kpiRawValue.textContent = item.raw_value || '-';
    els.kpiUnitSource.textContent = item.unit_source || '-';
    if (item.rel_error !== null && item.rel_error !== undefined) {
        els.kpiRelError.textContent = `${(item.rel_error * 100).toFixed(3)}%`;
    } else {
        els.kpiRelError.textContent = '-';
    }
    if (item.snippet) {
        els.kpiSnippet.innerHTML = `<code>${escapeHtml(item.snippet)}</code>`;
    } else {
        els.kpiSnippet.textContent = '';
    }
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
    els.queryId.textContent = data.item.query_id;
    els.industryValue.textContent = data.item.industry_slug;
    els.tickerValue.textContent = `${data.item.exchange}:${data.item.ticker} \u00b7 ${data.item.year}`;
    els.pageValue.textContent = `Page ${data.item.page_idx + 1}`;
    els.imageSubtitle.textContent = data.item.raw_png_path || 'No raw image path';
    els.markdownSubtitle.textContent = `${data.item.page_text_chars} chars \u00b7 ${data.item.page_text_sha256.slice(0, 12)}`;

    renderKpiContext(data.item);

    els.markdownPreview.innerHTML = data.markdown_html || '';
    els.rawMarkdown.innerHTML = data.page_text_highlighted || escapeHtml(data.page_text || '');
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
        setZoom(1);
    }

    loadAnnotation(data.annotation);
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
    els.zoomOutButton.addEventListener('click', () => adjustZoom(-0.15));
    els.zoomInButton.addEventListener('click', () => adjustZoom(0.15));
    els.zoomResetButton.addEventListener('click', () => focusCurrentItem({ resetZoom: true }));
    els.helpButton.addEventListener('click', () => els.helpDialog.showModal());
    els.rawImage.addEventListener('load', () => focusCurrentItem({ resetZoom: true }));
    window.addEventListener('resize', () => {
        const anchorPoint = viewportCenterPoint();
        applyZoom();
        scheduleAfterLayout(() => centerViewportOnPoint(anchorPoint));
    });
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
            quickMark('2');
        } else if (event.key.toLowerCase() === 'u') {
            event.preventDefault();
            quickMark('1');
        } else if (event.key.toLowerCase() === 'r') {
            event.preventDefault();
            quickMark('0');
        } else if (event.key === 'ArrowRight' || event.key.toLowerCase() === 'j') {
            event.preventDefault();
            go(1);
        } else if (event.key === 'ArrowLeft' || event.key.toLowerCase() === 'k') {
            event.preventDefault();
            go(-1);
        } else if (event.key === '+' || event.key === '=') {
            event.preventDefault();
            adjustZoom(0.15);
        } else if (event.key === '-') {
            event.preventDefault();
            adjustZoom(-0.15);
        } else if (event.key === '0') {
            event.preventDefault();
            focusCurrentItem({ resetZoom: true });
        } else if (event.key.toLowerCase() === 'f') {
            event.preventDefault();
            focusCurrentItem({ resetZoom: true });
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
