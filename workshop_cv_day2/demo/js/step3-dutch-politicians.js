/**
 * step3-dutch-politicians.js — Dutch Politicians CV & NLP Bias Audit
 * CV: Loads ~143 politician photos, runs FairFace on each, aggregates by party.
 * NLP: Interactive dashboard with Chart.js charts for racism/sexism scores.
 */

const Step3 = {
  // ── CV Bias Audit state ──
  members: [],
  partyData: {},
  activeFilter: 'all',
  analyzing: false,

  ETHNICITY_COLORS: {
    'White':             '#4a9eff',
    'Black':             '#ff4a4a',
    'Latino_Hispanic':   '#ffd94a',
    'East Asian':        '#00ff66',
    'Southeast Asian':   '#ff8c4a',
    'Indian':            '#c84aff',
    'Middle Eastern':    '#4affec',
  },

  // ── NLP Dashboard state ──
  dashData: null,
  dashSelectedParties: new Set(),
  dashCharts: {},

  PARTY_COLORS: {
    "PVV":"#002F6C","VVD":"#FF6600","GL-PvdA":"#E12D1B","D66":"#01AF36",
    "CDA":"#007B5F","SP":"#EE2E22","PvdD":"#006C2E","CU":"#00A7EB",
    "FVD":"#841723","DENK":"#00B4D8","SGP":"#F67D00","BBB":"#94C11F",
    "JA21":"#1B2845","50PLUS":"#8B2FA0","Volt":"#562884",
    "Groep Markuszower":"#888888"
  },

  // ── Init ──
  async init() {
    // Accordion toggles
    this.initAccordions();

    // CV audit
    await this.loadData();
    this.renderGallery();
    document.getElementById('btn-analyze-all').addEventListener('click', () => this.analyzeAll());

    // NLP dashboard
    this.initDashboard();
  },

  // ══════════════════════════════════════════════════════════════════════
  // Accordion toggle
  // ══════════════════════════════════════════════════════════════════════

  initAccordions() {
    document.querySelectorAll('#step-3 .accordion-toggle').forEach(btn => {
      btn.addEventListener('click', () => {
        const accordion = btn.closest('.accordion');
        const isOpen = accordion.classList.contains('open');
        accordion.classList.toggle('open');
        btn.setAttribute('aria-expanded', !isOpen);

        // Lazy-init dashboard charts on first open
        if (!isOpen && accordion.id === 'accordion-dashboard' && this.dashData && !this.dashChartsDrawn) {
          // Small delay to let the DOM render the container
          setTimeout(() => {
            this.updateDashCharts();
            this.dashChartsDrawn = true;
          }, 50);
        }
      });
    });
  },

  // ══════════════════════════════════════════════════════════════════════
  // CV Bias Audit (existing functionality)
  // ══════════════════════════════════════════════════════════════════════

  async loadData() {
    const data = window.MEMBERS_DATA;
    if (!data) {
      console.error('MEMBERS_DATA not found — make sure data/members.js is loaded');
      return;
    }

    for (const [partyKey, party] of Object.entries(data)) {
      this.partyData[partyKey] = {
        name: party.party_name,
        count: party.number_of_members,
      };

      for (const member of party.members) {
        this.members.push({
          name: member.name,
          party: partyKey,
          partyName: party.party_name,
          photo: `data/${partyKey}/${member.profile_image}`,
          result: null,
        });
      }
    }
  },

  renderGallery() {
    this.renderPartyFilter();
    this.renderGrid();
  },

  renderPartyFilter() {
    const container = document.getElementById('pol-party-filter');
    const parties = Object.entries(this.partyData).sort((a, b) => b[1].count - a[1].count);

    const allBtn = `<button class="party-chip active" data-party="all">All (${this.members.length})</button>`;
    const partyBtns = parties.map(([key, p]) =>
      `<button class="party-chip" data-party="${key}">${key.toUpperCase()} (${p.count})</button>`
    ).join('');

    container.innerHTML = allBtn + partyBtns;

    container.addEventListener('click', (e) => {
      const btn = e.target.closest('.party-chip');
      if (!btn) return;
      container.querySelectorAll('.party-chip').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      this.activeFilter = btn.dataset.party;
      this.renderGrid();
    });
  },

  renderGrid() {
    const container = document.getElementById('pol-grid');
    const filtered = this.activeFilter === 'all'
      ? this.members
      : this.members.filter(m => m.party === this.activeFilter);

    container.innerHTML = filtered.map((m) => {
      const globalIdx = this.members.indexOf(m);
      const resultHtml = m.result ? this.renderCardResult(m.result) : '<div class="pol-card-pending">Not analyzed</div>';
      const borderColor = m.result ? (this.ETHNICITY_COLORS[m.result.dominant_race] || 'var(--accent)') : '#222';

      return `
        <div class="pol-card" data-idx="${globalIdx}" style="border-color:${borderColor}">
          <div class="pol-card-img">
            <img src="${m.photo}" alt="${m.name}" loading="lazy">
          </div>
          <div class="pol-card-info">
            <div class="pol-card-name">${m.name}</div>
            <div class="pol-card-party">${m.party.toUpperCase()}</div>
            ${resultHtml}
          </div>
        </div>
      `;
    }).join('');
  },

  renderCardResult(result) {
    const sorted = Object.entries(result.race).sort((a, b) => b[1] - a[1]);
    const top = sorted[0];
    const color = this.ETHNICITY_COLORS[top[0]] || 'var(--accent)';
    const topBars = sorted.slice(0, 3).map(([label, pct]) => `
      <div class="pol-mini-bar">
        <span class="pol-mini-label">${label}</span>
        <div class="pol-mini-track">
          <div class="pol-mini-fill" style="width:${pct.toFixed(0)}%;background:${this.ETHNICITY_COLORS[label] || 'var(--accent)'}"></div>
        </div>
        <span class="pol-mini-val">${pct.toFixed(0)}%</span>
      </div>
    `).join('');

    return `
      <div class="pol-card-result">
        <div class="pol-card-dominant" style="color:${color}">${result.dominant_race}</div>
        ${topBars}
      </div>
    `;
  },

  async analyzeAll() {
    if (this.analyzing) return;
    if (!Step2.modelsReady) {
      alert('Models are still loading. Please wait for both indicators to show "ready".');
      return;
    }

    this.analyzing = true;
    const btn = document.getElementById('btn-analyze-all');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    const progress = document.getElementById('pol-progress');
    const fill = document.getElementById('pol-progress-fill');
    const pct = document.getElementById('pol-progress-pct');
    const steps = document.getElementById('pol-progress-steps');
    progress.classList.add('visible');

    const total = this.members.length;

    for (let i = 0; i < total; i++) {
      const member = this.members[i];
      if (member.result) {
        const p = ((i + 1) / total * 100);
        fill.style.width = p + '%';
        pct.textContent = Math.round(p) + '%';
        steps.textContent = `${i + 1} / ${total}`;
        continue;
      }

      try {
        const img = await this.loadImageFromUrl(member.photo);
        const data = await Step2.analyzeImage(img);

        if (data.faces.length > 0) {
          member.result = data.faces[0];
        } else {
          member.result = await this.classifyWholeImage(img);
        }
      } catch (err) {
        console.warn(`Failed to analyze ${member.name}:`, err);
        member.result = null;
      }

      const p = ((i + 1) / total * 100);
      fill.style.width = p + '%';
      pct.textContent = Math.round(p) + '%';
      steps.textContent = `${i + 1} / ${total}`;

      this.renderGrid();
      await new Promise(r => setTimeout(r, 10));
    }

    progress.classList.remove('visible');
    btn.textContent = 'Analysis Complete';
    this.analyzing = false;

    this.renderChart();
    document.getElementById('pol-chart-section').classList.remove('hidden');

    // Auto-open the Discussion accordion after analysis
    const discussionAccordion = document.getElementById('accordion-discussion');
    if (discussionAccordion && !discussionAccordion.classList.contains('open')) {
      discussionAccordion.classList.add('open');
      discussionAccordion.querySelector('.accordion-toggle').setAttribute('aria-expanded', 'true');
    }
  },

  async classifyWholeImage(img) {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const tensor = Step2.preprocessFace(canvas, { x: 0, y: 0, width: canvas.width, height: canvas.height });
    const result = await Step2.classifyFace(tensor);

    return {
      region: { x: 0, y: 0, w: canvas.width, h: canvas.height },
      race: result.race,
      dominant_race: result.dominant_race,
    };
  },

  loadImageFromUrl(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load: ' + url));
      img.src = url;
    });
  },

  renderChart() {
    const container = document.getElementById('pol-chart');
    const partyResults = {};

    for (const member of this.members) {
      if (!member.result) continue;
      if (!partyResults[member.party]) {
        partyResults[member.party] = { name: member.partyName, counts: {}, total: 0 };
      }
      const pr = partyResults[member.party];
      pr.total++;
      const dom = member.result.dominant_race;
      pr.counts[dom] = (pr.counts[dom] || 0) + 1;
    }

    const sorted = Object.entries(partyResults).sort((a, b) => b[1].total - a[1].total);

    const chartHtml = sorted.map(([key, party]) => {
      const bars = Step2.RACE_LABELS.map(label => {
        const count = party.counts[label] || 0;
        const pct = party.total > 0 ? (count / party.total * 100) : 0;
        if (pct === 0) return '';
        return `
          <div class="chart-segment" style="width:${pct}%;background:${this.ETHNICITY_COLORS[label]}" title="${label}: ${count}/${party.total} (${pct.toFixed(0)}%)"></div>
        `;
      }).join('');

      return `
        <div class="chart-row">
          <div class="chart-label">${key.toUpperCase()}<span class="chart-count">${party.total}</span></div>
          <div class="chart-bar">${bars}</div>
        </div>
      `;
    }).join('');

    const legend = Step2.RACE_LABELS.map(label => `
      <span class="chart-legend-item">
        <span class="chart-legend-dot" style="background:${this.ETHNICITY_COLORS[label]}"></span>
        ${label}
      </span>
    `).join('');

    container.innerHTML = `
      <div class="chart-legend">${legend}</div>
      <div class="chart-rows">${chartHtml}</div>
    `;
  },

  // ══════════════════════════════════════════════════════════════════════
  // NLP Dashboard (Chart.js)
  // ══════════════════════════════════════════════════════════════════════

  dashChartsDrawn: false,

  initDashboard() {
    this.dashData = window.DASHBOARD_DATA;
    if (!this.dashData) {
      console.warn('DASHBOARD_DATA not found — NLP dashboard disabled');
      return;
    }

    // Sort select
    const sortSelect = document.getElementById('dashSortSelect');
    if (sortSelect) {
      sortSelect.addEventListener('change', () => this.updateDashCharts());
    }

    // Build party selector
    this.buildDashPartySelector();

    // Populate results tab
    this.populateDashResults();
  },

  getDashFilteredData() {
    let items = this.dashData.parties.filter(d => d.source === 'manifesto');
    if (this.dashSelectedParties.size > 0) {
      items = items.filter(d => this.dashSelectedParties.has(d.party));
    }
    return items;
  },

  sortDashData(items) {
    const sortEl = document.getElementById('dashSortSelect');
    const sort = sortEl ? sortEl.value : 'racism';
    const copy = [...items];
    if (sort === 'racism') copy.sort((a, b) => b.racismScore - a.racismScore);
    else if (sort === 'sexism') copy.sort((a, b) => b.sexismScore - a.sexismScore);
    else if (sort === 'combined') copy.sort((a, b) => (b.racismScore + b.sexismScore) - (a.racismScore + a.sexismScore));
    else copy.sort((a, b) => a.party.localeCompare(b.party));
    return copy;
  },

  buildDashPartySelector() {
    const container = document.getElementById('dashPartySelector');
    if (!container || !this.dashData) return;

    const allParties = [...new Set(this.dashData.parties.map(d => d.party))];

    const allBtn = document.createElement('button');
    allBtn.className = 'party-chip active';
    allBtn.textContent = 'All Parties';
    allBtn.addEventListener('click', () => {
      this.dashSelectedParties.clear();
      container.querySelectorAll('.party-chip').forEach(b => b.classList.remove('active'));
      allBtn.classList.add('active');
      document.getElementById('dashDetailPanel').classList.remove('visible');
      this.updateDashCharts();
    });
    container.appendChild(allBtn);

    allParties.forEach(party => {
      const btn = document.createElement('button');
      btn.className = 'party-chip';
      btn.textContent = party;
      btn.style.borderColor = this.PARTY_COLORS[party] || '#666';
      btn.addEventListener('click', () => {
        if (this.dashSelectedParties.has(party)) {
          this.dashSelectedParties.delete(party);
          btn.classList.remove('active');
        } else {
          this.dashSelectedParties.add(party);
          btn.classList.add('active');
        }
        if (this.dashSelectedParties.size > 0) allBtn.classList.remove('active');
        else allBtn.classList.add('active');

        if (this.dashSelectedParties.size === 1) this.showDashDetail([...this.dashSelectedParties][0]);
        else document.getElementById('dashDetailPanel').classList.remove('visible');

        this.updateDashCharts();
      });
      container.appendChild(btn);
    });
  },

  // ── Chart management ──

  destroyDashCharts() {
    Object.values(this.dashCharts).forEach(c => { if (c) c.destroy(); });
    this.dashCharts = {};
  },

  updateDashCharts() {
    this.destroyDashCharts();
    const items = this.sortDashData(this.getDashFilteredData());
    if (!items.length) return;

    this.drawCombinedBar(items);
    this.drawScatter(items);
    this.drawRadar(items);
    this.drawTopicChart(items);
    this.drawSentimentChart(items);
  },

  drawCombinedBar(items) {
    const ctx = document.getElementById('dashCombinedBarChart');
    if (!ctx) return;
    this.dashCharts.combinedBar = new Chart(ctx.getContext('2d'), {
      type: 'bar',
      data: {
        labels: items.map(d => d.party),
        datasets: [
          {
            label: 'Racism Score',
            data: items.map(d => d.racismScore),
            backgroundColor: '#D32F2FCC',
            borderColor: '#D32F2F',
            borderWidth: 1, borderRadius: 4, borderSkipped: false,
          },
          {
            label: 'Sexism Score',
            data: items.map(d => d.sexismScore),
            backgroundColor: '#7B1FA2CC',
            borderColor: '#7B1FA2',
            borderWidth: 1, borderRadius: 4, borderSkipped: false,
          }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: {
          legend: { labels: { color: '#e0e0e0' } },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)} / 100`
            }
          }
        },
        scales: {
          x: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } },
          y: { grid: { display: false }, ticks: { color: '#e0e0e0', font: { weight: '600' } } }
        },
        onClick: (_, elems) => {
          if (elems.length) this.showDashDetail(items[elems[0].index].party);
        }
      }
    });
  },

  drawScatter(items) {
    const ctx = document.getElementById('dashScatterChart');
    if (!ctx) return;
    this.dashCharts.scatter = new Chart(ctx.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: items.map(d => ({
          label: d.party,
          data: [{ x: d.racismScore, y: d.sexismScore }],
          backgroundColor: d.color || '#666',
          pointRadius: 8,
          pointHoverRadius: 12,
        }))
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const d = items[ctx.datasetIndex];
                return `${d.party}: Racism ${d.racismScore.toFixed(1)}, Sexism ${d.sexismScore.toFixed(1)}`;
              }
            }
          }
        },
        scales: {
          x: { min: 0, max: 100, title: { display: true, text: 'Racism Score', color: '#888' },
               grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } },
          y: { min: 0, max: 100, title: { display: true, text: 'Sexism Score', color: '#888' },
               grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } }
        },
        onClick: (_, elems) => {
          if (elems.length) this.showDashDetail(items[elems[0].datasetIndex].party);
        }
      }
    });
  },

  drawRadar(items) {
    const ctx = document.getElementById('dashRadarChart');
    if (!ctx) return;
    const radarLabels = ['Lex. Racism', 'Dog Whistles', 'Imm. Topic',
                         'Neg. Sentiment', 'ZS Racism', 'Lex. Sexism',
                         'Gen. Topic', 'ZS Sexism'];

    const datasets = items.slice(0, 8).map(d => {
      const c = d.components || {};
      return {
        label: d.party,
        data: [
          c.lex_racism || 0, c.lex_dogwhistle || 0,
          (c.immigration_topic || 0) * 100, c.sent_pct_negative || 0,
          c.zs_racism || 0, c.lex_sexism || 0,
          (c.gender_topic || 0) * 100, c.zs_sexism || 0,
        ],
        borderColor: d.color || '#666',
        backgroundColor: (d.color || '#666') + '22',
        pointBackgroundColor: d.color || '#666',
        borderWidth: 2,
      };
    });

    this.dashCharts.radar = new Chart(ctx.getContext('2d'), {
      type: 'radar',
      data: { labels: radarLabels, datasets },
      options: {
        responsive: true,
        plugins: { legend: { position: 'bottom', labels: { color: '#e0e0e0', font: { size: 10 } } } },
        scales: {
          r: {
            beginAtZero: true,
            grid: { color: 'rgba(255,255,255,0.08)' },
            angleLines: { color: 'rgba(255,255,255,0.08)' },
            ticks: { color: '#888', backdropColor: 'transparent' },
            pointLabels: { color: '#e0e0e0', font: { size: 10 } },
          }
        }
      }
    });
  },

  drawTopicChart(items) {
    const ctx = document.getElementById('dashTopicChart');
    if (!ctx) return;
    this.dashCharts.topic = new Chart(ctx.getContext('2d'), {
      type: 'bar',
      data: {
        labels: items.map(d => d.party),
        datasets: [
          {
            label: 'Immigration/Minority Topic',
            data: items.map(d => ((d.components || {}).immigration_topic || 0) * 100),
            backgroundColor: '#D32F2F', borderRadius: 3,
          },
          {
            label: 'Gender Topic',
            data: items.map(d => ((d.components || {}).gender_topic || 0) * 100),
            backgroundColor: '#7B1FA2', borderRadius: 3,
          }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#e0e0e0' } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#e0e0e0', maxRotation: 45 } },
          y: { title: { display: true, text: 'Topic Weight (%)', color: '#888' },
               grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } }
        }
      }
    });
  },

  drawSentimentChart(items) {
    const ctx = document.getElementById('dashSentimentChart');
    if (!ctx) return;
    this.dashCharts.sentiment = new Chart(ctx.getContext('2d'), {
      type: 'bar',
      data: {
        labels: items.map(d => d.party),
        datasets: [
          {
            label: '% Negative',
            data: items.map(d => (d.sentiment || {}).pctNegative || 0),
            backgroundColor: '#ff4a4a', borderRadius: 3,
          },
          {
            label: '% Positive',
            data: items.map(d => (d.sentiment || {}).pctPositive || 0),
            backgroundColor: '#00ff66', borderRadius: 3,
          }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#e0e0e0' } } },
        scales: {
          x: { stacked: false, grid: { display: false }, ticks: { color: '#e0e0e0', maxRotation: 45 } },
          y: { title: { display: true, text: 'Percentage', color: '#888' },
               grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } }
        }
      }
    });
  },

  // ── Detail panel ──

  showDashDetail(party) {
    const d = this.dashData.parties.find(p => p.party === party && p.source === 'manifesto');
    if (!d) return;

    const panel = document.getElementById('dashDetailPanel');
    panel.classList.add('visible');
    const nameEl = document.getElementById('dashDetailPartyName');
    nameEl.textContent = party;
    nameEl.style.color = d.color;

    const stats = document.getElementById('dashDetailStats');
    stats.innerHTML = `
      <div class="dash-stat-card">
        <div class="dash-stat-value" style="color:${d.racismScore > 50 ? 'var(--red)' : 'var(--green)'}">${d.racismScore.toFixed(1)}</div>
        <div class="dash-stat-label">Racism Score</div>
      </div>
      <div class="dash-stat-card">
        <div class="dash-stat-value" style="color:${d.sexismScore > 50 ? 'var(--red)' : 'var(--green)'}">${d.sexismScore.toFixed(1)}</div>
        <div class="dash-stat-label">Sexism Score</div>
      </div>
      <div class="dash-stat-card">
        <div class="dash-stat-value">${(d.sentiment || {}).numPassages || 'N/A'}</div>
        <div class="dash-stat-label">Passages Analysed</div>
      </div>
      <div class="dash-stat-card">
        <div class="dash-stat-value">${((d.sentiment || {}).meanScore || 0).toFixed(1)}</div>
        <div class="dash-stat-label">Mean Sentiment (1-5)</div>
      </div>
    `;

    document.getElementById('dashRacismTerms').innerHTML = (d.topRacismTerms || []).map(t =>
      `<li><span>${t.term}</span><span class="dash-term-count">${t.count}</span></li>`
    ).join('');
    document.getElementById('dashSexismTerms').innerHTML = (d.topSexismTerms || []).map(t =>
      `<li><span>${t.term}</span><span class="dash-term-count">${t.count}</span></li>`
    ).join('');

    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  // ── Results tab ──

  populateDashResults() {
    if (!this.dashData || !this.dashData.parties.length) return;

    const items = this.dashData.parties.filter(d => d.source === 'manifesto');
    if (!items.length) return;

    items.sort((a, b) => (b.racismScore + b.sexismScore) - (a.racismScore + a.sexismScore));

    // Results table
    let html = '<table class="dash-results-table"><tr><th>#</th><th>Party</th><th>Racism</th><th>Sexism</th><th>Combined</th></tr>';
    items.forEach((d, i) => {
      const combined = d.racismScore + d.sexismScore;
      const rClass = d.racismScore >= 70 ? 'score-high' : d.racismScore >= 40 ? 'score-med' : 'score-low';
      const sClass = d.sexismScore >= 70 ? 'score-high' : d.sexismScore >= 40 ? 'score-med' : 'score-low';
      html += `<tr>
        <td>${i + 1}</td>
        <td><strong style="color:${this.PARTY_COLORS[d.party] || '#666'}">${d.party}</strong></td>
        <td class="${rClass}">${d.racismScore.toFixed(1)}</td>
        <td class="${sClass}">${d.sexismScore.toFixed(1)}</td>
        <td>${combined.toFixed(1)}</td>
      </tr>`;
    });
    html += '</table>';
    document.getElementById('dashResultsContent').innerHTML = html;

    // Key observations
    const topRacism = [...items].sort((a, b) => b.racismScore - a.racismScore);
    const topSexism = [...items].sort((a, b) => b.sexismScore - a.sexismScore);
    const lowestRacism = topRacism[topRacism.length - 1];
    const lowestSexism = topSexism[topSexism.length - 1];

    let obs = '<ul class="dash-list">';
    obs += `<li><strong>${topRacism[0].party}</strong> scored highest on racism (${topRacism[0].racismScore.toFixed(1)}), `;
    obs += `followed by <strong>${topRacism[1].party}</strong> (${topRacism[1].racismScore.toFixed(1)}) `;
    obs += `and <strong>${topRacism[2].party}</strong> (${topRacism[2].racismScore.toFixed(1)}).</li>`;
    obs += `<li><strong>${topSexism[0].party}</strong> scored highest on sexism (${topSexism[0].sexismScore.toFixed(1)}), `;
    obs += `followed by <strong>${topSexism[1].party}</strong> (${topSexism[1].sexismScore.toFixed(1)}) `;
    obs += `and <strong>${topSexism[2].party}</strong> (${topSexism[2].sexismScore.toFixed(1)}).</li>`;
    obs += `<li><strong>${lowestRacism.party}</strong> had the lowest racism score (${lowestRacism.racismScore.toFixed(1)}); `;
    obs += `<strong>${lowestSexism.party}</strong> had the lowest sexism score (${lowestSexism.sexismScore.toFixed(1)}).</li>`;

    items.forEach(d => {
      const diff = Math.abs(d.racismScore - d.sexismScore);
      if (diff > 40) {
        const higher = d.racismScore > d.sexismScore ? 'racism' : 'sexism';
        obs += `<li><strong>${d.party}</strong> shows a notable divergence: much higher ${higher} (${Math.max(d.racismScore, d.sexismScore).toFixed(1)} vs ${Math.min(d.racismScore, d.sexismScore).toFixed(1)}).</li>`;
      }
    });
    obs += '</ul>';
    document.getElementById('dashKeyObservations').innerHTML = obs;

    // Component breakdown table
    let compHtml = '<table class="dash-results-table"><tr><th>Party</th><th>Lex. R</th><th>Dog W.</th>';
    compHtml += '<th>Neg. Sent.</th><th>ZS Racism</th><th>Lex. S</th><th>ZS Sexism</th></tr>';
    items.forEach(d => {
      const c = d.components || {};
      compHtml += `<tr>
        <td><strong style="color:${this.PARTY_COLORS[d.party] || '#666'}">${d.party}</strong></td>
        <td>${(c.lex_racism || 0).toFixed(1)}</td>
        <td>${(c.lex_dogwhistle || 0).toFixed(1)}</td>
        <td>${(c.sent_pct_negative || 0).toFixed(1)}%</td>
        <td>${(c.zs_racism || 0).toFixed(1)}%</td>
        <td>${(c.lex_sexism || 0).toFixed(1)}</td>
        <td>${(c.zs_sexism || 0).toFixed(1)}%</td>
      </tr>`;
    });
    compHtml += '</table>';
    document.getElementById('dashComponentTable').innerHTML = compHtml;
  },
};

document.addEventListener('DOMContentLoaded', () => Step3.init());
