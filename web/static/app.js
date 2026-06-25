'use strict';

const FIELD_LABELS = {
  brand_name: 'Brand Name',
  class_type: 'Class / Type',
  abv_percent: 'Alcohol Content',
  net_contents: 'Net Contents',
  government_warning: 'Government Warning',
};

function showTab(name, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(name).style.display = 'block';
  btn.classList.add('active');
}

function previewImage(input) {
  const preview = document.getElementById('preview');
  const placeholder = document.getElementById('upload-placeholder');
  if (!input.files[0]) return;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.style.display = 'block';
    placeholder.style.display = 'none';
  };
  reader.readAsDataURL(input.files[0]);
}

// Pre-fill the form with a known PASS example so reviewers can try immediately.
// The image is fetched from /static/example_label.jpg served alongside the app.
async function loadExample() {
  document.getElementById('brand_name').value  = 'JIM BEAM';
  document.getElementById('class_type').value  = 'STRAIGHT BOURBON WHISKY';
  document.getElementById('abv_percent').value = '62.5';
  document.getElementById('net_contents').value = '375ml';

  // Fetch the example image and inject it into the file input via a DataTransfer
  try {
    const resp = await fetch('/static/example_label.jpg');
    const blob = await resp.blob();
    const file = new File([blob], 'example_label.jpg', { type: 'image/jpeg' });
    const dt = new DataTransfer();
    dt.items.add(file);
    const input = document.getElementById('image-input');
    input.files = dt.files;
    previewImage(input);
  } catch {
    // If static file unavailable, form fields are still pre-filled
  }
}

async function verifyLabel() {
  const image = document.getElementById('image-input').files[0];
  if (!image) { alert('Please upload a label image first.'); return; }

  const fd = new FormData();
  fd.append('image', image);
  fd.append('brand_name', document.getElementById('brand_name').value);
  fd.append('class_type', document.getElementById('class_type').value);
  fd.append('abv_percent', document.getElementById('abv_percent').value);
  fd.append('net_contents', document.getElementById('net_contents').value);

  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '<p class="spinner">Processing&hellip;</p>';

  try {
    const resp = await fetch('/verify', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    renderResult(data, resultDiv);
  } catch (e) {
    resultDiv.innerHTML = `<p class="error-msg">Error: ${e.message}</p>`;
  }
}

function renderResult(result, container) {
  const verdictClass = result.verdict === 'PASS' ? 'pass' : 'fail';

  let rows = '';
  for (const [key, v] of Object.entries(result.fields)) {
    const statusClass = `status-${v.status.toLowerCase()}`;
    rows += `<tr class="row-${v.status.toLowerCase()}">
      <td>${FIELD_LABELS[key] || key}</td>
      <td class="${statusClass}">${v.status}</td>
      <td>${trunc(v.expected, 90)}</td>
      <td>${trunc(v.extracted, 90)}</td>
      <td>${v.score != null ? v.score.toFixed(1) : '&mdash;'}</td>
    </tr>`;
  }

  const lowConf = result.confidence < 0.6
    ? `<div class="low-confidence">Low extraction confidence (${(result.confidence * 100).toFixed(0)}%) &mdash; review manually.</div>`
    : '';

  container.innerHTML = `
    <hr>
    ${lowConf}
    <div class="verdict ${verdictClass}">${result.verdict}</div>
    <p class="result-notes">${result.notes}</p>
    <table>
      <thead>
        <tr><th>Field</th><th>Status</th><th>Expected</th><th>Extracted</th><th>Score</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <button class="btn-download" onclick='downloadJSON(${JSON.stringify(JSON.stringify(result))})'>
      Download Report (JSON)
    </button>`;
}

async function runBatch() {
  const csvFile = document.getElementById('batch-csv').files[0];
  const images = document.getElementById('batch-images').files;
  if (!csvFile) { alert('Please upload a CSV file.'); return; }
  if (!images.length) { alert('Please upload label images.'); return; }

  const fd = new FormData();
  fd.append('csv', csvFile);
  for (const img of images) fd.append('images', img);

  const resultDiv = document.getElementById('batch-result');
  resultDiv.innerHTML = '<p class="spinner">Processing batch&hellip;</p>';

  try {
    const resp = await fetch('/batch', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    renderBatch(data, resultDiv);
  } catch (e) {
    resultDiv.innerHTML = `<p class="error-msg">Error: ${e.message}</p>`;
  }
}

function renderBatch(results, container) {
  let rows = results.map(r => {
    const cls = r.verdict === 'PASS' ? 'row-pass' : r.verdict === 'FAIL' ? 'row-fail' : 'row-unknown';
    const sCls = `status-${r.verdict.toLowerCase()}`;
    return `<tr class="${cls}">
      <td>${r.filename}</td>
      <td class="${sCls}">${r.verdict}</td>
      <td>${r.notes}</td>
    </tr>`;
  }).join('');

  const csv = 'filename,verdict,notes\n' +
    results.map(r => `${r.filename},${r.verdict},"${r.notes}"`).join('\n');

  container.innerHTML = `
    <hr>
    <table>
      <thead><tr><th>Filename</th><th>Verdict</th><th>Notes</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <button class="btn-download" onclick='downloadCSV(${JSON.stringify(csv)})'>
      Download Results (CSV)
    </button>`;
}

function trunc(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n) + '…' : s;
}

function downloadJSON(jsonStr) {
  download(jsonStr, 'ttb_report.json', 'application/json');
}

function downloadCSV(csvStr) {
  download(csvStr, 'batch_results.csv', 'text/csv');
}

function download(content, filename, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
