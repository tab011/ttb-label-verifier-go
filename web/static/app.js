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
  autoPopulateFields(input.files[0]);
}

// autoPopulateFields sends the image to /extract and fills in the form fields.
// The user can then correct anything before clicking Verify.
async function autoPopulateFields(file) {
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '<p class="spinner">Reading label&hellip;</p>';

  try {
    const { blob } = await prepareImageForUpload(file);
    const fd = new FormData();
    fd.append('image', blob, file.name);

    const resp = await fetch('/extract', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) { resultDiv.innerHTML = ''; return; }

    // Populate form fields with extracted values (don't overwrite if already filled)
    const fields = ['brand_name', 'class_type', 'abv_percent', 'net_contents'];
    let filled = 0;
    for (const f of fields) {
      const el = document.getElementById(f);
      const val = f === 'abv_percent'
        ? (data[f] > 0 ? String(data[f]) : '')
        : (data[f] || '');
      if (val) { el.value = val; filled++; }
    }

    const conf = Math.round((data.confidence || 0) * 100);
    const cat = data.spirit_category && data.spirit_category !== 'UNKNOWN'
      ? ` &nbsp;·&nbsp; <strong>${data.spirit_category}</strong>`
      : '';
    resultDiv.innerHTML = filled > 0
      ? `<p class="extract-hint">Fields auto-populated from image (${conf}% confidence)${cat} &mdash; correct anything that looks wrong, then click <strong>Verify Label</strong>.</p>`
      : `<p class="extract-hint low">Could not read label text. Fill in the fields manually and click <strong>Verify Label</strong>.</p>`;
  } catch {
    resultDiv.innerHTML = '';
  }
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

// prepareImageForUpload resizes the image to max 1500px on the longest side
// and converts it to grayscale with contrast stretching. If the label region
// is predominantly dark (white text on dark background — common on bourbon
// bottles), it inverts the image so Tesseract reads light text on white.
// Returns a Blob ready to append to FormData.
async function prepareImageForUpload(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const MAX = 1500;
      let w = img.width, h = img.height;
      if (w > MAX || h > MAX) {
        const scale = MAX / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }

      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);

      const imageData = ctx.getImageData(0, 0, w, h);
      const d = imageData.data;

      // Convert to grayscale + contrast stretch
      const gray = new Uint8Array(w * h);
      for (let i = 0, p = 0; i < d.length; i += 4, p++) {
        gray[p] = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
      }

      // Measure average brightness of the center label region (middle 50% of image)
      let sum = 0, count = 0;
      const cx0 = Math.round(w * 0.25), cx1 = Math.round(w * 0.75);
      const cy0 = Math.round(h * 0.20), cy1 = Math.round(h * 0.80);
      for (let y = cy0; y < cy1; y++) {
        for (let x = cx0; x < cx1; x++) {
          sum += gray[y * w + x];
          count++;
        }
      }
      const avgBrightness = count > 0 ? sum / count : 128;

      // Dark label (avg < 110): invert so Tesseract sees black text on white.
      // Bourbon labels are almost always dark background with light text.
      const invert = avgBrightness < 110;

      // Write back grayscale (inverted if needed) with mild contrast boost
      for (let i = 0, p = 0; i < d.length; i += 4, p++) {
        let v = gray[p];
        if (invert) v = 255 - v;
        // Contrast: stretch toward extremes (mid-point ±40%)
        v = Math.min(255, Math.max(0, Math.round((v - 128) * 1.4 + 128)));
        d[i] = d[i + 1] = d[i + 2] = v;
        d[i + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);

      canvas.toBlob(blob => resolve({ blob, invert, avgBrightness }), 'image/jpeg', 0.92);
    };
    img.src = URL.createObjectURL(file);
  });
}

async function verifyLabel() {
  const image = document.getElementById('image-input').files[0];
  if (!image) { alert('Please upload a label image first.'); return; }

  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '<p class="spinner">Preprocessing image&hellip;</p>';

  const { blob, invert, avgBrightness } = await prepareImageForUpload(image);

  resultDiv.innerHTML = `<p class="spinner">Verifying&hellip; ${invert ? '(inverted dark label)' : '(light label)'}</p>`;

  const fd = new FormData();
  fd.append('image', blob, image.name);
  fd.append('brand_name', document.getElementById('brand_name').value);
  fd.append('class_type', document.getElementById('class_type').value);
  fd.append('abv_percent', document.getElementById('abv_percent').value);
  fd.append('net_contents', document.getElementById('net_contents').value);

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

  const spiritBadge = result.spirit_category && result.spirit_category !== 'UNKNOWN'
    ? `<div class="spirit-badge">Visual classification: <strong>${result.spirit_category}</strong> &nbsp;<span class="spirit-conf">${(result.spirit_confidence * 100).toFixed(0)}% confidence</span></div>`
    : '';

  container.innerHTML = `
    <hr>
    ${spiritBadge}
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

// ── Camera capture ──────────────────────────────────────────────────────────

let cameraStream = null;

async function toggleCamera(e) {
  e.preventDefault();
  const panel = document.getElementById('camera-panel');
  const video = document.getElementById('camera-video');

  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
    panel.style.display = 'none';
    e.target.textContent = 'Use Camera';
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = cameraStream;
    panel.style.display = 'block';
    e.target.textContent = 'Close Camera';
  } catch (err) {
    alert('Camera access denied or unavailable: ' + err.message);
  }
}

function captureFrame() {
  const video  = document.getElementById('camera-video');
  const canvas = document.getElementById('camera-canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
    const dt = new DataTransfer();
    dt.items.add(file);
    const input = document.getElementById('image-input');
    input.files = dt.files;
    previewImage(input);

    // Stop camera after capture
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }
    document.getElementById('camera-panel').style.display = 'none';
    document.querySelector('button[onclick^="toggleCamera"]').textContent = 'Use Camera';
  }, 'image/jpeg', 0.92);
}

// ── Batch annotation template ───────────────────────────────────────────────

function downloadTemplate() {
  const csv = [
    'filename,brand_name,class_type,abv_percent,net_contents',
    'label_001.jpg,MAKER\'S MARK,STRAIGHT BOURBON WHISKY,45.0,750 mL',
    'label_002.jpg,BUFFALO TRACE,KENTUCKY STRAIGHT BOURBON WHISKY,40.0,1750 mL',
    'label_003.jpg,WILD TURKEY 101,STRAIGHT BOURBON WHISKY,50.5,750 mL',
  ].join('\n');
  download(csv, 'ttb_annotation_template.csv', 'text/csv');
}
