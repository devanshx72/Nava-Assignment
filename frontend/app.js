const BASE = 'http://localhost:8000/api';

/* ───────────────────────────────────────────
   Navigation
─────────────────────────────────────────── */
function showTask(id) {
  document.querySelectorAll('.task-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('#nav-list button').forEach(b => b.classList.remove('active'));
  document.getElementById('task' + id).classList.add('active');
  document.getElementById('nav-btn-' + id).classList.add('active');
}

/* ───────────────────────────────────────────
   Health check
─────────────────────────────────────────── */
async function checkHealth() {
  const el = document.getElementById('health-dot');
  try {
    const r = await fetch(BASE + '/health');
    if (r.ok) {
      el.classList.add('ok');
      el.innerHTML = '<span class="dot"></span> Backend online';
    } else { throw new Error(); }
  } catch {
    el.classList.add('err');
    el.innerHTML = '<span class="dot"></span> Backend offline';
  }
}

/* ───────────────────────────────────────────
   Helpers
─────────────────────────────────────────── */
function setLoading(btn, loading) {
  if (loading) {
    btn.disabled = true;
    btn.dataset.orig = btn.innerHTML;
    btn.innerHTML = '<span class="spinner"></span> Running…';
  } else {
    btn.disabled = false;
    btn.innerHTML = btn.dataset.orig;
  }
}

function showError(panelId, msg) {
  const el = document.getElementById(panelId);
  el.classList.remove('empty');
  el.innerHTML = '<span class="error-msg">Error: ' + escHtml(msg) + '</span>';
}

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function colorJson(obj) {
  const json = JSON.stringify(obj, null, 2);
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g, match => {
    if (/^"/.test(match)) {
      if (/:$/.test(match)) return '<span class="json-key">' + match + '</span>';
      return '<span class="json-string">' + match + '</span>';
    }
    if (/true|false/.test(match)) return '<span class="json-bool">' + match + '</span>';
    if (/null/.test(match)) return '<span class="json-null">' + match + '</span>';
    return '<span class="json-number">' + match + '</span>';
  });
}

function setJson(panelId, data) {
  const el = document.getElementById(panelId);
  el.classList.remove('empty');
  el.innerHTML = colorJson(data);
}

async function apiFetch(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

/* ───────────────────────────────────────────
   TASK 1 — RAG Pipeline
─────────────────────────────────────────── */
async function task1Ingest() {
  const btn = document.getElementById('t1-ingest-btn');
  const raw = document.getElementById('t1-docs').value.trim();
  if (!raw) return;
  const docs = raw.split('\n').filter(l => l.trim());
  setLoading(btn, true);
  try {
    const data = await apiFetch(BASE + '/task1/ingest', { documents: docs });
    setJson('t1-ingest-out', data);
  } catch (e) { showError('t1-ingest-out', e.message); }
  setLoading(btn, false);
}

async function task1Query() {
  const btn = document.getElementById('t1-query-btn');
  const query = document.getElementById('t1-query').value.trim();
  const top_k = parseInt(document.getElementById('t1-topk').value) || 3;
  if (!query) return;
  setLoading(btn, true);
  try {
    const data = await apiFetch(BASE + '/task1/query', { query, top_k });
    // Rich display
    const el = document.getElementById('t1-query-out');
    el.classList.remove('empty');
    let html = '<b style="color:var(--accent-h)">Answer</b><br>' + escHtml(data.answer) + '<br><br>';
    html += '<b style="color:var(--accent-h)">Retrieved Chunks</b><br>';
    (data.retrieved_chunks || []).forEach((c, i) => {
      const dist = data.distances ? data.distances[i] : '';
      html += '<div class="chunk-item">' + escHtml(c) +
        '<div class="chunk-meta">Distance: ' + (dist !== '' ? Number(dist).toFixed(4) : 'N/A') + '</div></div>';
    });
    el.innerHTML = html;
  } catch (e) { showError('t1-query-out', e.message); }
  setLoading(btn, false);
}

/* ───────────────────────────────────────────
   TASK 1 — PDF Upload
─────────────────────────────────────────── */
function task1PdfSelected(input) {
  const label = document.getElementById('t1-pdf-label-text');
  const btn = document.getElementById('t1-pdf-btn');
  if (input.files && input.files[0]) {
    label.textContent = input.files[0].name;
    btn.disabled = false;
  } else {
    label.textContent = 'Click to choose a PDF file';
    btn.disabled = true;
  }
}

async function task1IngestPdf() {
  const btn = document.getElementById('t1-pdf-btn');
  const input = document.getElementById('t1-pdf-input');
  if (!input.files || !input.files[0]) return;

  const formData = new FormData();
  formData.append('file', input.files[0]);

  setLoading(btn, true);
  try {
    const r = await fetch(BASE + '/task1/ingest_pdf', {
      method: 'POST',
      body: formData
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    const data = await r.json();
    setJson('t1-pdf-out', data);
  } catch (e) { showError('t1-pdf-out', e.message); }
  setLoading(btn, false);
}

/* ───────────────────────────────────────────
   TASK 2 — AI Agent
─────────────────────────────────────────── */
async function task2Run() {
  const btn = document.getElementById('t2-btn');
  const query = document.getElementById('t2-query').value.trim();
  if (!query) return;
  setLoading(btn, true);
  try {
    const data = await apiFetch(BASE + '/task2/run', { query });
    const el = document.getElementById('t2-out');
    el.classList.remove('empty');
    let html = '';
    (data.steps || []).forEach((s, i) => {
      html += '<div class="step-item">';
      if (s.tool) html += '<div class="step-tool">Tool: ' + escHtml(s.tool) + '</div>';
      html += '<div>' + escHtml(s.thought || '') + '</div>';
      if (s.result) html += '<div class="step-result">Result: ' + escHtml(s.result) + '</div>';
      html += '</div>';
    });
    html += '<div style="margin-top:14px;padding:14px;background:var(--accent-dim);border-radius:8px;">' +
      '<b style="color:var(--accent-h)">Final Answer</b><br>' + escHtml(data.final_answer || '') + '</div>';
    el.innerHTML = html;
  } catch (e) { showError('t2-out', e.message); }
  setLoading(btn, false);
}

/* ───────────────────────────────────────────
   TASK 3 — LLM Judge
─────────────────────────────────────────── */
async function task3Run() {
  const btn = document.getElementById('t3-btn');
  const question = document.getElementById('t3-question').value.trim();
  const answer = document.getElementById('t3-answer').value.trim();
  const ref = document.getElementById('t3-ref').value.trim();
  if (!question) return;
  setLoading(btn, true);
  try {
    const data = await apiFetch(BASE + '/task3/evaluate', { question, answer, reference_answer: ref });
    const el = document.getElementById('t3-out');
    el.classList.remove('empty');
    const s = data.scores || {};
    const criteria = ['accuracy', 'relevance', 'completeness', 'clarity', 'overall'];
    let html = '<div class="score-grid">';
    criteria.forEach(k => {
      const v = s[k] || 0;
      html += '<div class="score-card">' +
        '<div class="score-label">' + k + '</div>' +
        '<div class="score-value">' + v + '</div>' +
        '<div class="score-bar"><div class="score-bar-fill" style="width:' + (v * 10) + '%"></div></div>' +
        '</div>';
    });
    html += '</div>';
    const verdict = (data.verdict || '').toUpperCase();
    const badgeCls = verdict === 'PASS' ? 'badge-pass' : 'badge-fail';
    html += '<div style="margin-top:16px;">' +
      '<span class="badge ' + badgeCls + '">' + verdict + '</span>' +
      '<span style="color:var(--text-muted);font-size:13px;margin-left:12px;">' + escHtml(data.reasoning || '') + '</span>' +
      '</div>';
    if (data.generated_answer) {
      html += '<br><b style="font-size:12px;color:var(--text-muted)">Generated Answer</b><br>' +
        '<div style="font-size:13px;margin-top:6px;line-height:1.7;">' + escHtml(data.generated_answer) + '</div>';
    }
    el.innerHTML = html;
  } catch (e) { showError('t3-out', e.message); }
  setLoading(btn, false);
}

/* ───────────────────────────────────────────
   TASK 4 — Hallucination Detection
─────────────────────────────────────────── */
async function task4Run() {
  const btn = document.getElementById('t4-btn');
  const context = document.getElementById('t4-context').value.trim();
  const answer = document.getElementById('t4-answer').value.trim();
  if (!context || !answer) return;
  setLoading(btn, true);
  try {
    const data = await apiFetch(BASE + '/task4/detect', { context, answer });
    const el = document.getElementById('t4-out');
    el.classList.remove('empty');
    const verdict = (data.verdict || '').toUpperCase();
    const badgeCls = verdict === 'GROUNDED' ? 'badge-grounded' : 'badge-hallucinated';
    let html = '<div style="margin-bottom:14px;display:flex;align-items:center;gap:12px;">' +
      '<span class="badge ' + badgeCls + '">' + verdict + '</span>' +
      '<span style="color:var(--text-muted);font-size:13px;">Hallucination Score: ' +
      '<b style="color:var(--text)">' + (data.hallucination_score * 100).toFixed(1) + '%</b></span>' +
      '</div>';
    const sum = data.summary || {};
    html += '<div style="display:flex;gap:16px;margin-bottom:16px;font-size:13px;">' +
      '<span style="color:var(--green)">Supported: ' + (sum.supported || 0) + '</span>' +
      '<span style="color:var(--red)">Contradicted: ' + (sum.contradicted || 0) + '</span>' +
      '<span style="color:var(--yellow)">Not Mentioned: ' + (sum.not_mentioned || 0) + '</span>' +
      '</div>';
    (data.sentence_analysis || []).forEach(item => {
      const v = item.verdict || 'NOT_MENTIONED';
      html += '<div class="sentence-item ' + v + '">' +
        '<span class="sentence-verdict verdict-' + v + '">' + v + '</span>' +
        escHtml(item.sentence) +
        '<div style="font-size:11px;color:var(--text-muted);margin-top:6px;">' + escHtml(item.reason || '') + ' (conf: ' + (item.confidence || 0).toFixed(2) + ')</div>' +
        '</div>';
    });
    el.innerHTML = html;
  } catch (e) { showError('t4-out', e.message); }
  setLoading(btn, false);
}

/* ───────────────────────────────────────────
   TASK 5 — Re-Ranker
─────────────────────────────────────────── */
async function task5Run() {
  const btn = document.getElementById('t5-btn');
  const query = document.getElementById('t5-query').value.trim();
  const raw = document.getElementById('t5-results').value.trim();
  if (!query || !raw) return;
  const results = raw.split('\n').filter(l => l.trim());
  setLoading(btn, true);
  try {
    const data = await apiFetch(BASE + '/task5/rerank', { query, results });
    const el = document.getElementById('t5-out');
    el.classList.remove('empty');
    let html = '<b style="color:var(--accent-h);font-size:12px;">Re-ranked Results</b><br><br>';
    (data.reranked_results || []).forEach(item => {
      html += '<div class="rank-item">' +
        '<div class="rank-num">' + item.rank + '</div>' +
        '<div style="flex:1;">' +
        '<div>' + escHtml(item.document) + '</div>' +
        '<div class="rank-score">Score: ' + (item.score || 0).toFixed(3) + '</div>' +
        '<div class="rank-reason">' + escHtml(item.reason || '') + '</div>' +
        '</div></div>';
    });
    el.innerHTML = html;
  } catch (e) { showError('t5-out', e.message); }
  setLoading(btn, false);
}

/* ───────────────────────────────────────────
   TASK 6 — Guardrails
─────────────────────────────────────────── */
async function task6Run() {
  const btn = document.getElementById('t6-btn');
  const topic = document.getElementById('t6-topic').value.trim();
  const llm_output = document.getElementById('t6-output').value.trim();
  if (!topic || !llm_output) return;
  setLoading(btn, true);
  try {
    const data = await apiFetch(BASE + '/task6/check', { topic, llm_output });
    const el = document.getElementById('t6-out');
    el.classList.remove('empty');
    const status = (data.overall_status || '').toUpperCase();
    const badgeCls = status === 'ALLOWED' ? 'badge-allowed' : 'badge-blocked';
    let html = '<div style="margin-bottom:16px;"><span class="badge ' + badgeCls + '">' + status + '</span></div>';
    const checks = data.checks || {};

    // Off-topic
    const ot = checks.off_topic || {};
    const otVerdict = (ot.verdict || '').toUpperCase();
    const otOk = otVerdict === 'ON_TOPIC';
    html += '<div class="check-row">' +
      '<div><div class="check-label">Off-topic Check</div><div class="check-reason">' + escHtml(ot.reason || '') + '</div></div>' +
      '<span class="badge ' + (otOk ? 'badge-on-topic' : 'badge-off-topic') + '">' + (ot.verdict || '—') + '</span>' +
      '</div>';

    // PII
    const pii = checks.pii || {};
    const piiOk = !pii.found;
    html += '<div class="check-row">' +
      '<div><div class="check-label">PII Detection</div>' +
      '<div class="check-reason">' + (pii.found ? 'Found: ' + (pii.types || []).join(', ') : 'No PII detected') + '</div></div>' +
      '<span class="badge ' + (piiOk ? 'badge-allowed' : 'badge-blocked') + '">' + (piiOk ? 'CLEAN' : 'PII FOUND') + '</span>' +
      '</div>';
    if (pii.found) {
      html += '<div style="font-size:12px;color:var(--text-muted);padding:8px 14px;background:var(--surface2);border-radius:8px;margin-bottom:8px;">' +
        'Redacted: ' + escHtml(pii.redacted_text || '') + '</div>';
    }

    // Toxicity
    const tox = checks.toxicity || {};
    const toxOk = !tox.is_toxic;
    html += '<div class="check-row">' +
      '<div><div class="check-label">Toxicity Check</div><div class="check-reason">' + escHtml(tox.reason || '') + '</div></div>' +
      '<span class="badge ' + (toxOk ? 'badge-allowed' : 'badge-blocked') + '">' + (toxOk ? 'CLEAN' : 'TOXIC') + '</span>' +
      '</div>';

    el.innerHTML = html;
  } catch (e) { showError('t6-out', e.message); }
  setLoading(btn, false);
}

/* ───────────────────────────────────────────
   TASK 7 — Multi-Agent
─────────────────────────────────────────── */
async function task7Run() {
  const btn = document.getElementById('t7-btn');
  const task = document.getElementById('t7-task').value.trim();
  if (!task) return;
  setLoading(btn, true);
  try {
    const data = await apiFetch(BASE + '/task7/run', { task });
    const el = document.getElementById('t7-out');
    el.classList.remove('empty');
    const icons = ['R', 'A', 'W'];
    let html = '<div>';
    (data.pipeline || []).forEach((step, i) => {
      html += '<div class="pipeline-step">' +
        '<div class="pipeline-icon">' + (icons[i] || i + 1) + '</div>' +
        '<div class="pipeline-content">' +
        '<div class="pipeline-agent-name">' + escHtml(step.agent) + '</div>' +
        '<div class="pipeline-output-box">' + escHtml(step.output || '') + '</div>' +
        '</div></div>';
    });
    html += '</div>';
    el.innerHTML = html;
  } catch (e) { showError('t7-out', e.message); }
  setLoading(btn, false);
}

/* ───────────────────────────────────────────
   Init
─────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  showTask(1);
});
