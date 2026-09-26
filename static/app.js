'use strict';
const types = {
  REFLECTION: ['Reflection', 'Make sense of your own learning, decisions, and experiences.'],
  RESEARCH: ['Research', 'Gather, compare, and assess evidence to build understanding.'],
  COLLABORATE: ['Collaborate', 'Work with other people toward a shared decision or outcome.'],
  CREATE: ['Create', 'Produce an original artifact, composition, or solution.'],
  PRACTICE: ['Practice', 'Improve a skill through repeated performance and feedback.'],
  EXPERIENCE: ['Experience', 'Learn through firsthand exposure or participation.'],
  TEACH: ['Teach', 'Help someone else learn through explanation and guidance.']
};
const examples = {
  create: ['Make a birthday song', 'Produce an original song for a friend.', 'Invent new lyrics and compose a melody.'],
  practice: ['Rehearse a birthday song', 'Improve your performance of an existing song.', 'Repeat the song several times, correcting missed notes and timing.'],
  research: ['Investigate neighborhood temperatures', 'Understand why some streets are warmer than others.', 'Compare published temperature and tree-cover records. Summarize the evidence and note limitations.']
};
const form = document.querySelector('form');
const fields = ['title', 'description', 'instructions'].map(id => document.getElementById(id));
const result = document.getElementById('result');
const empty = document.getElementById('empty');
const status = document.getElementById('status');
const error = document.getElementById('error');
const submit = document.getElementById('classify');
let controller;
let requestId = 0;
function busy(value) {
  submit.disabled = value;
  submit.textContent = value ? 'Classifying…' : 'Classify activity ↗';
  document.querySelector('.result-card').setAttribute('aria-busy', String(value));
}
function clearResult() {
  requestId++;
  if (controller) controller.abort();
  busy(false);
  result.hidden = true;
  empty.hidden = false;
  error.hidden = true;
  status.textContent = 'Your result will appear here.';
}
fields.forEach(field => field.addEventListener('input', clearResult));
form.addEventListener('reset', clearResult);
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  clearResult();
  examples[button.dataset.example].forEach((value, i) => { fields[i].value = value; });
  fields[0].focus();
}));
form.addEventListener('submit', async event => {
  event.preventDefault();
  clearResult();
  const payload = Object.fromEntries(fields.map(field => [field.id, field.value.trim()]));
  if (!Object.values(payload).some(Boolean)) {
    error.textContent = 'Add a title, description, or instructions before classifying.';
    error.hidden = false;
    fields[0].focus();
    return;
  }
  const current = requestId;
  controller = new AbortController();
  const activeController = controller;
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; activeController.abort(); }, 60000);
  busy(true);
  status.textContent = 'Reading your activity…';
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload), signal: activeController.signal
    });
    if (!response.ok) throw new Error(response.status === 422
      ? 'Please check your activity fields and try again.'
      : 'The model could not complete this request. Please try again.');
    const data = await response.json();
    if (current !== requestId) return;
    if (!Object.hasOwn(types, data.activityType) || !Number.isFinite(data.confidence) ||
        data.confidence < 0 || data.confidence > 1 ||
        Object.keys(types).some(key => !Number.isFinite(data.scores?.[key]) || data.scores[key] < 0 || data.scores[key] > 1)) {
      throw new Error('The model returned an unexpected result. Please try again.');
    }
    document.getElementById('predicted-label').textContent = types[data.activityType][0];
    document.getElementById('definition').textContent = types[data.activityType][1];
    document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}% score`;
    const chart = document.getElementById('chart');
    chart.replaceChildren();
    for (const [key, [name]] of Object.entries(types)) {
      const percent = (data.scores[key] * 100).toFixed(1);
      const row = document.createElement('div');
      row.className = `score-row${key === data.activityType ? ' winner' : ''}`;
      row.setAttribute('role', 'listitem');
      const label = document.createElement('div'); label.className = 'score-label';
      const title = document.createElement('span'); title.textContent = name;
      const value = document.createElement('span'); value.textContent = `${percent}%`;
      label.append(title, value);
      const track = document.createElement('div'); track.className = 'track'; track.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${percent}%`;
      track.append(fill); row.append(label, track); chart.append(row);
    }
    empty.hidden = true; result.hidden = false;
    status.textContent = `Prediction ready: ${types[data.activityType][0]}.`;
  } catch (failure) {
    if (current !== requestId) return;
    error.textContent = timedOut ? 'This request took too long. Please try again.' :
      failure instanceof TypeError ? 'Could not reach the model. Check that the local server is running.' : failure.message;
    error.hidden = false;
    status.textContent = 'No prediction available. Please try again.';
  } finally {
    clearTimeout(timeout);
    if (current === requestId) busy(false);
  }
});
