const state = {
    sessionId: window.OCR_ANNOTATION_DEFAULT_SESSION_ID,
    index: 0,
    itemCount: 0,
    item: null,
    overallStatus: 'unreviewed',
    startedAt: null,
    zoom: 1,
    showingRaw: false,
    saving: false,
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
    rawImage: document.getElementById('rawImage'),
    imageMissing: document.getElementById('imageMissing'),
    imageSubtitle: document.getElementById('imageSubtitle'),
    markdownSubtitle: document.getElementById('markdownSubtitle'),
    markdownPreview: document.getElementById('markdownPreview'),
    rawMarkdown: document.getElementById('rawMarkdown'),
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
    issueGrid: document.getElementById('issueGrid'),
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
    const pct = total ? Math.round((reviewed / total) * 100) : 0;
    els.progressBar.style.width = `${pct}%`;
    els.summaryCsvLink.href = `/api/session/${state.sessionId}/summary.csv`;
    els.summaryMdLink.href = `/api/session/${state.sessionId}/summary.md`;
}

function setOverall(status) {
    state.overallStatus = status;
    document.querySelectorAll('.status-button').forEach((button) => {
        button.classList.toggle('active', button.dataset.status === status);
    });
}

function setSubchecks(values = {}) {
    document.querySelectorAll('[data-subcheck]').forEach((select) => {
        select.value = values[select.dataset.subcheck] || 'unreviewed';
    });
}

function setIssues(values = []) {
    const selected = new Set(values);
    els.issueGrid.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
        checkbox.checked = selected.has(checkbox.value);
    });
}

function getSubchecks() {
    const subchecks = {};
    document.querySelectorAll('[data-subcheck]').forEach((select) => {
        subchecks[select.dataset.subcheck] = select.value;
    });
    return subchecks;
}

function getIssues() {
    return Array.from(els.issueGrid.querySelectorAll('input[type="checkbox"]:checked'))
        .map((checkbox) => checkbox.value)
        .sort();
}

function loadAnnotation(annotation) {
    setOverall(annotation?.overall_status || 'unreviewed');
    setSubchecks(annotation?.subchecks || {});
    setIssues(annotation?.issue_tags || []);
    els.notesInput.value = annotation?.notes || '';
}

function applyZoom() {
    els.rawImage.style.transform = `scale(${state.zoom})`;
    els.rawImage.style.marginBottom = `${Math.max(0, (state.zoom - 1) * 100)}%`;
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

async function loadItem(index) {
    const safeIndex = Math.max(0, Math.min(index, Math.max(0, state.itemCount - 1)));
    const data = await apiJson(`/api/session/${state.sessionId}/item/${safeIndex}`);
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

    if (data.item.raw_png_path) {
        els.rawImage.hidden = false;
        els.imageMissing.hidden = true;
        els.rawImage.src = `${data.image_url}?v=${encodeURIComponent(data.item.page_text_sha256)}`;
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
        subchecks: getSubchecks(),
        issue_tags: getIssues(),
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

function quickMark(status) {
    setOverall(status);
    if (status === 'ok') {
        setSubchecks({
            text_content: 'ok',
            table_content: 'ok',
            table_structure: 'ok',
            page_alignment: 'ok',
        });
        setIssues([]);
    } else if (status === 'not_ok') {
        const subchecks = getSubchecks();
        if (Object.values(subchecks).every((value) => value === 'unreviewed')) {
            setSubchecks({
                text_content: 'uncertain',
                table_content: 'uncertain',
                table_structure: 'not_ok',
                page_alignment: 'uncertain',
            });
        }
    } else if (status === 'uncertain') {
        setSubchecks({
            text_content: 'uncertain',
            table_content: 'uncertain',
            table_structure: 'uncertain',
            page_alignment: 'uncertain',
        });
    }
    saveAnnotation(`shortcut:${status}`, true);
}

function toggleIssue(tag) {
    const checkbox = els.issueGrid.querySelector(`input[value="${tag}"]`);
    if (checkbox) checkbox.checked = !checkbox.checked;
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
    els.toggleRawButton.addEventListener('click', toggleRawMarkdown);
    els.zoomOutButton.addEventListener('click', () => setZoom(state.zoom - 0.15));
    els.zoomInButton.addEventListener('click', () => setZoom(state.zoom + 0.15));
    els.zoomResetButton.addEventListener('click', () => setZoom(1));
    els.helpButton.addEventListener('click', () => els.helpDialog.showModal());
    document.querySelectorAll('.status-button').forEach((button) => {
        button.addEventListener('click', () => setOverall(button.dataset.status));
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
        } else if (event.key.toLowerCase() === 't') {
            event.preventDefault();
            toggleIssue('broken_table');
        } else if (event.key.toLowerCase() === 'c') {
            event.preventDefault();
            toggleIssue('merged_columns');
        } else if (event.key.toLowerCase() === 'm') {
            event.preventDefault();
            toggleIssue('missing_text');
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