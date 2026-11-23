/**
 * ACADEMIX ANALYTICS — Core Reactive Controller
 */

// Model linear coefficients learned from Ridge Regressor
const COEFFICIENTS = {
  intercept: 5.388,
  study_hours: 2.385,
  attendance: 0.248,
  previous_marks: 0.576
};

// DOM References
const hoursSlider = document.getElementById('hoursSlider');
const hoursInput = document.getElementById('hoursInput');
const attendanceSlider = document.getElementById('attendanceSlider');
const attendanceInput = document.getElementById('attendanceInput');
const previousSlider = document.getElementById('previousSlider');
const previousInput = document.getElementById('previousInput');
const resetBtn = document.getElementById('resetBtn');

// Outcomes
const predictedScoreEl = document.getElementById('predictedScore');
const gradeBadgeEl = document.getElementById('gradeBadge');
const tierLabelEl = document.getElementById('tierLabel');
const meterFillEl = document.getElementById('meterFill');

// Attribution
const ptPreviousEl = document.getElementById('ptPrevious');
const ptHoursEl = document.getElementById('ptHours');
const ptAttendanceEl = document.getElementById('ptAttendance');
const ptBaselineEl = document.getElementById('ptBaseline');
const barPreviousEl = document.getElementById('barPrevious');
const barHoursEl = document.getElementById('barHours');
const barAttendanceEl = document.getElementById('barAttendance');
const barBaselineEl = document.getElementById('barBaseline');

// Quick Sim
const valPlus1h = document.getElementById('valPlus1h');
const valPlus2h = document.getElementById('valPlus2h');
const valPlusAtt = document.getElementById('valPlusAtt');

// Goal Planner & Advisory
const plannerGrid = document.getElementById('plannerGrid');
const advisoryText = document.getElementById('advisoryText');
const actionItemsList = document.getElementById('actionItemsList');
const presetsGrid = document.getElementById('presetsGrid');
const sensitivityChart = document.getElementById('sensitivityChart');

let debounceTimer = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  initDualSync();
  fetchPresets();
  runPrediction();
});

function initDualSync() {
  const syncPairs = [
    { slider: hoursSlider, input: hoursInput },
    { slider: attendanceSlider, input: attendanceInput },
    { slider: previousSlider, input: previousInput }
  ];

  syncPairs.forEach(({ slider, input }) => {
    slider.addEventListener('input', () => {
      input.value = slider.value;
      debouncedPredict();
    });

    input.addEventListener('input', () => {
      let val = parseFloat(input.value);
      if (!isNaN(val)) {
        slider.value = val;
        debouncedPredict();
      }
    });

    input.addEventListener('change', () => {
      let min = parseFloat(input.min) || 0;
      let max = parseFloat(input.max) || 100;
      let val = parseFloat(input.value);
      if (isNaN(val) || val < min) input.value = min;
      if (val > max) input.value = max;
      slider.value = input.value;
      runPrediction();
    });
  });

  resetBtn.addEventListener('click', () => {
    hoursSlider.value = 6.0;
    hoursInput.value = 6.0;
    attendanceSlider.value = 88.0;
    attendanceInput.value = 88.0;
    previousSlider.value = 78.0;
    previousInput.value = 78.0;
    clearPresetSelection();
    runPrediction();
  });
}

function debouncedPredict() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    runPrediction();
  }, 40);
}

async function runPrediction() {
  const hours = parseFloat(hoursInput.value) || 0;
  const attendance = parseFloat(attendanceInput.value) || 0;
  const previous = parseFloat(previousInput.value) || 0;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        study_hours: hours,
        attendance: attendance,
        previous_marks: previous
      })
    });

    if (!res.ok) throw new Error('Network error');
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    // Immediate clientside fallback computation
    const fallback = computeClientPrediction(hours, attendance, previous);
    renderResults(fallback);
  }
}

function computeClientPrediction(hours, attendance, previous) {
  const rawScore = COEFFICIENTS.intercept +
    (COEFFICIENTS.study_hours * hours) +
    (COEFFICIENTS.attendance * attendance) +
    (COEFFICIENTS.previous_marks * previous);

  const finalScore = Math.max(0.0, Math.min(100.0, Math.round(rawScore * 10) / 10));

  let grade = "C";
  let tier = "Satisfactory";
  let status = "warning";
  let feedback = "Solid effort. Raising daily study or attendance will strengthen retention.";
  let actions = ["Attend all scheduled lectures.", "Add 1 hour of active problem solving."];

  if (finalScore >= 90) {
    grade = "A+";
    tier = "Outstanding Academic Standing";
    status = "success";
    feedback = "Exceptional performance trajectory across all curriculum indicators.";
    actions = ["Maintain current structured daily review routine.", "Explore honors-tier problem sets."];
  } else if (finalScore >= 80) {
    grade = "A";
    tier = "Distinction Standing";
    status = "success";
    feedback = "Strong performance. Well positioned for honors honors with slight study boost.";
    actions = ["Dedicate 45 minutes extra daily to high-weight questions.", "Keep attendance >90%."];
  } else if (finalScore >= 70) {
    grade = "B";
    tier = "Good Standing";
    status = "info";
    feedback = "Good foundation with steady progression.";
    actions = ["Increase self-study to 5.5+ hrs/day.", "Review midterm mock tests."];
  } else if (finalScore >= 50) {
    grade = "D";
    tier = "Academic Risk Deficit";
    status = "danger";
    feedback = "Borderline status; high vulnerability on comprehensive final examinations.";
    actions = ["Schedule immediate tutoring session.", "Commit to mandatory 3+ hrs study."];
  } else {
    grade = "F";
    tier = "Critical Remedial Required";
    status = "critical";
    feedback = "Severe risk of course failure. Immediate academic intervention recommended.";
    actions = ["Consult faculty advisor.", "Enact daily structured remediation."];
  }

  const c_base = COEFFICIENTS.intercept;
  const c_hours = Math.round(COEFFICIENTS.study_hours * hours * 100) / 100;
  const c_att = Math.round(COEFFICIENTS.attendance * attendance * 100) / 100;
  const c_prev = Math.round(COEFFICIENTS.previous_marks * previous * 100) / 100;

  const sim_1h = Math.min(100.0, Math.round((finalScore + COEFFICIENTS.study_hours * 1.0) * 10) / 10);
  const sim_2h = Math.min(100.0, Math.round((finalScore + COEFFICIENTS.study_hours * 2.0) * 10) / 10);
  const sim_att = Math.min(100.0, Math.round((finalScore + COEFFICIENTS.attendance * 10.0) * 10) / 10);

  const goal_plans = {};
  [
    { name: "A+ (90+)", val: 90.0 },
    { name: "A (80+)", val: 80.0 },
    { name: "B (70+)", val: 70.0 }
  ].forEach(t => {
    if (finalScore >= t.val) {
      goal_plans[t.name] = { achieved: true };
    } else {
      const neededHours = (t.val - c_base - c_att - c_prev) / COEFFICIENTS.study_hours;
      const neededAtt = (t.val - c_base - c_hours - c_prev) / COEFFICIENTS.attendance;
      goal_plans[t.name] = {
        achieved: false,
        needed_hours_daily: neededHours <= 14.0 ? Math.round(Math.max(0.5, neededHours) * 10) / 10 : null,
        needed_attendance_pct: neededAtt <= 100.0 ? Math.round(Math.max(0.0, neededAtt) * 10) / 10 : null
      };
    }
  });

  const curve = [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0].map(h => {
    const raw = c_base + (COEFFICIENTS.study_hours * h) + c_att + c_prev;
    return { hours: h, predicted_score: Math.max(0.0, Math.min(100.0, Math.round(raw * 10) / 10)) };
  });

  return {
    predicted_score: finalScore,
    grade: grade,
    tier: tier,
    status: status,
    feedback: feedback,
    action_items: actions,
    contributions: {
      baseline: c_base,
      study_hours: c_hours,
      attendance: c_att,
      previous_marks: c_prev
    },
    what_if: {
      delta_1h: Math.round((sim_1h - finalScore) * 10) / 10,
      delta_2h: Math.round((sim_2h - finalScore) * 10) / 10,
      delta_att: Math.round((sim_att - finalScore) * 10) / 10
    },
    goal_plans: goal_plans,
    sensitivity_curve: curve
  };
}

function renderResults(data) {
  // Score & Grade Pill
  predictedScoreEl.textContent = data.predicted_score.toFixed(1);
  gradeBadgeEl.textContent = `Grade ${data.grade}`;
  gradeBadgeEl.className = `grade-pill status-${data.status}`;
  tierLabelEl.textContent = data.tier;

  // Meter Fill
  meterFillEl.style.width = `${Math.min(100, Math.max(0, data.predicted_score))}%`;
  if (data.status === 'success') {
    meterFillEl.style.backgroundColor = 'var(--emerald-success)';
  } else if (data.status === 'info') {
    meterFillEl.style.backgroundColor = 'var(--primary-accent)';
  } else if (data.status === 'warning') {
    meterFillEl.style.backgroundColor = 'var(--amber-warning)';
  } else {
    meterFillEl.style.backgroundColor = 'var(--crimson-danger)';
  }

  // Factor Attribution
  const c = data.contributions;
  ptPreviousEl.textContent = `+${c.previous_marks.toFixed(1)} pts`;
  ptHoursEl.textContent = `+${c.study_hours.toFixed(1)} pts`;
  ptAttendanceEl.textContent = `+${c.attendance.toFixed(1)} pts`;
  ptBaselineEl.textContent = `+${c.baseline.toFixed(1)} pts`;

  barPreviousEl.style.width = `${Math.min(100, (c.previous_marks / 100) * 100)}%`;
  barHoursEl.style.width = `${Math.min(100, (c.study_hours / 35) * 100)}%`;
  barAttendanceEl.style.width = `${Math.min(100, (c.attendance / 25) * 100)}%`;

  // Quick Sim Values
  if (data.what_if) {
    valPlus1h.textContent = `+${data.what_if.delta_1h} marks`;
    valPlus2h.textContent = `+${data.what_if.delta_2h} marks`;
    valPlusAtt.textContent = `+${data.what_if.delta_att} marks`;
  }

  // Goal Planner
  renderGoalPlanner(data.goal_plans);

  // Advisory synthesis
  advisoryText.textContent = data.feedback;
  actionItemsList.innerHTML = '';
  (data.action_items || []).forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    actionItemsList.appendChild(li);
  });

  // Render SVG Sensitivity Chart
  if (data.sensitivity_curve) {
    renderSvgChart(data.sensitivity_curve, parseFloat(hoursInput.value), data.predicted_score);
  }
}

function renderGoalPlanner(plans) {
  if (!plans) return;
  plannerGrid.innerHTML = '';

  Object.entries(plans).forEach(([tier, info]) => {
    const card = document.createElement('div');
    card.className = 'target-card';

    if (info.achieved) {
      card.innerHTML = `
        <span class="target-title">Target: ${tier}</span>
        <span class="target-achieved">&#10003; Target Met / Exceeded</span>
      `;
    } else {
      let reqDetails = [];
      if (info.needed_hours_daily) {
        reqDetails.push(`Study: <strong>${info.needed_hours_daily} hrs/day</strong>`);
      }
      if (info.needed_attendance_pct) {
        reqDetails.push(`Attendance: <strong>${info.needed_attendance_pct}%</strong>`);
      }
      if (reqDetails.length === 0) {
        reqDetails.push("Requires prerequisite mark improvement.");
      }

      card.innerHTML = `
        <span class="target-title">Target: ${tier}</span>
        <span class="target-metric">${reqDetails.join('<br>or ')}</span>
      `;
    }

    plannerGrid.appendChild(card);
  });
}

function renderSvgChart(curve, currentHours, currentScore) {
  const width = 700;
  const height = 240;
  const padLeft = 50;
  const padRight = 30;
  const padTop = 20;
  const padBottom = 35;

  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const minH = 1.0, maxH = 10.0;
  const minS = 40.0, maxS = 100.0;

  const toX = (h) => padLeft + ((h - minH) / (maxH - minH)) * plotW;
  const toY = (s) => padTop + plotH - ((s - minS) / (maxS - minS)) * plotH;

  // Grid Lines & Labels
  let svgContent = `
    <defs>
      <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#2563eb" stop-opacity="0.18" />
        <stop offset="100%" stop-color="#2563eb" stop-opacity="0.01" />
      </linearGradient>
    </defs>
  `;

  // Horizontal Grid Lines (Score 40, 60, 80, 100)
  [50, 70, 85, 100].forEach(scoreVal => {
    const y = toY(scoreVal);
    svgContent += `
      <line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3" />
      <text x="${padLeft - 8}" y="${y + 4}" font-family="var(--font-mono)" font-size="10" fill="#94a3b8" text-anchor="end">${scoreVal}</text>
    `;
  });

  // Vertical Grid Lines (Hours 2, 4, 6, 8, 10)
  [2, 4, 6, 8, 10].forEach(hVal => {
    const x = toX(hVal);
    svgContent += `
      <line x1="${x}" y1="${padTop}" x2="${x}" y2="${padTop + plotH}" stroke="#f1f5f9" stroke-width="1" />
      <text x="${x}" y="${height - 10}" font-family="var(--font-mono)" font-size="10" fill="#94a3b8" text-anchor="middle">${hVal}h</text>
    `;
  });

  // Path Line & Area
  let pathD = '';
  let areaD = `M ${toX(curve[0].hours)} ${padTop + plotH}`;

  curve.forEach((pt, i) => {
    const x = toX(pt.hours);
    const y = toY(pt.predicted_score);
    if (i === 0) {
      pathD += `M ${x} ${y}`;
    } else {
      pathD += ` L ${x} ${y}`;
    }
    areaD += ` L ${x} ${y}`;
  });
  areaD += ` L ${toX(curve[curve.length - 1].hours)} ${padTop + plotH} Z`;

  svgContent += `<path d="${areaD}" fill="url(#areaGrad)" />`;
  svgContent += `<path d="${pathD}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" />`;

  // Draw Data Points
  curve.forEach(pt => {
    svgContent += `<circle cx="${toX(pt.hours)}" cy="${toY(pt.predicted_score)}" r="3" fill="#ffffff" stroke="#2563eb" stroke-width="2" />`;
  });

  // Highlight Current Student Position
  const curX = Math.min(width - padRight, Math.max(padLeft, toX(currentHours)));
  const curY = Math.min(padTop + plotH, Math.max(padTop, toY(currentScore)));

  svgContent += `
    <line x1="${curX}" y1="${padTop}" x2="${curX}" y2="${padTop + plotH}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3" />
    <circle cx="${curX}" cy="${curY}" r="6" fill="#dc2626" stroke="#ffffff" stroke-width="2" />
    <rect x="${Math.min(width - 100, curX + 8)}" y="${curY - 24}" width="75" height="20" rx="4" fill="#0f172a" />
    <text x="${Math.min(width - 100, curX + 8) + 37}" y="${curY - 10}" font-family="var(--font-mono)" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">${currentScore.toFixed(1)} / 100</text>
  `;

  sensitivityChart.innerHTML = svgContent;
}

async function fetchPresets() {
  try {
    const res = await fetch('/api/presets');
    if (!res.ok) return;
    const presets = await res.json();
    renderPresets(presets);
  } catch (e) {
    // Default presets
    const fallbackPresets = [
      { id: "honors", name: "Aarav Sharma", tag: "Honors Candidate", study_hours: 7.5, attendance: 94.0, previous_marks: 88.0, description: "Consistent diligence, solid grasp of curriculum, high attendance." },
      { id: "rebound", name: "Maya Lin", tag: "Midterm Rebound", study_hours: 7.0, attendance: 90.0, previous_marks: 64.0, description: "Overcoming lower midterms through rigorous daily practice and high attendance." },
      { id: "att_gap", name: "Liam Connor", tag: "Attendance Deficit", study_hours: 5.5, attendance: 62.0, previous_marks: 78.0, description: "Good subject aptitude but compromised by missed classroom lectures." },
      { id: "at_risk", name: "Dev Patel", tag: "At-Risk Support", study_hours: 2.0, attendance: 66.0, previous_marks: 48.0, description: "Struggling foundation and low engagement; requires structured tutoring." }
    ];
    renderPresets(fallbackPresets);
  }
}

function renderPresets(presets) {
  presetsGrid.innerHTML = '';
  presets.forEach((p, idx) => {
    const chip = document.createElement('div');
    chip.className = `preset-chip ${idx === 0 ? 'active' : ''}`;
    chip.dataset.id = p.id;
    chip.innerHTML = `
      <div class="preset-top">
        <span class="preset-name">${p.name}</span>
        <span class="preset-badge">${p.tag}</span>
      </div>
      <p class="preset-desc">${p.description}</p>
      <div class="preset-specs">${p.study_hours}h/day &bull; ${p.attendance}% Att &bull; ${p.previous_marks} Prev</div>
    `;

    chip.addEventListener('click', () => {
      document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      hoursSlider.value = p.study_hours;
      hoursInput.value = p.study_hours;
      attendanceSlider.value = p.attendance;
      attendanceInput.value = p.attendance;
      previousSlider.value = p.previous_marks;
      previousInput.value = p.previous_marks;

      runPrediction();
    });

    presetsGrid.appendChild(chip);
  });
}

function clearPresetSelection() {
  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
}
