import { AfterViewInit, Component, ElementRef, OnInit, QueryList, ViewChildren } from '@angular/core';
import { CommonModule } from '@angular/common';
import Chart from 'chart.js/auto';
import { ApiService } from '../../core/api.service';

type ResultRow = Record<string, string>;

interface ExperimentGroup {
  code: string;
  title: string;
  kind: 'sweep' | 'bar' | 'scalability';
  rows: ResultRow[];
}

const EXPERIMENT_TITLES: Record<string, string> = {
  A_MEM: 'MEM — Fidelity vs Noise (GHZ-4)',
  B_ZNE: 'ZNE — <ZZI> vs Noise (GHZ-3)',
  C_DDD: 'Dynamical Decoupling — <ZZI> vs Noise (GHZ-3)',
  D_CDR: 'CDR — <H_TFIM> vs Noise (VQE)',
  E_PEC: 'PEC — <ZZ> vs Noise (Bell)',
  F_VD: 'Virtual Distillation — <Z0> vs Noise (VQE)',
  G_QAOA_compare: 'Cross-technique comparison (QAOA MaxCut)',
  H_scalability: 'Scalability — runtime vs qubit count',
};

const CHART_COLORS = { raw: '#f2a65a', mitigated: '#8b93f8', ideal: '#4fd8c4' };
const LINE_COLORS = ['#8b93f8', '#4fd8c4', '#f2a65a'];

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './results.component.html',
})
export class ResultsComponent implements OnInit, AfterViewInit {
  @ViewChildren('chartCanvas') chartCanvases!: QueryList<ElementRef<HTMLCanvasElement>>;

  loading = true;
  error = '';
  allRows: ResultRow[] = [];
  groups: ExperimentGroup[] = [];

  tableColumns = ['experiment', 'technique', 'benchmark', 'noise_strength', 'ideal', 'raw', 'mitigated', 'error_reduction_pct', 'wall_time_s'];
  filterOptions: string[] = ['all'];
  activeFilter = 'all';
  visibleRows: ResultRow[] = [];

  figures: string[] = [];
  figuresError = '';

  private chartsBuilt = false;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getResults().subscribe({
      next: (res) => {
        this.allRows = res.rows;
        this.loading = false;
        if (!res.rows.length) {
          this.error = 'No precomputed results yet. Run experiments/run_benchmark_suite.py first.';
          return;
        }
        this.buildGroups();
        this.filterOptions = ['all', ...Array.from(new Set(this.allRows.map((r) => r['experiment'])))];
        this.visibleRows = this.allRows;
      },
      error: (e) => {
        this.loading = false;
        this.error = `Could not load results: ${e.message || e.statusText}`;
      },
    });

    this.api.getFigures().subscribe({
      next: (r) => (this.figures = r.figures),
      error: () => (this.figuresError = 'No figures found. Run the benchmark suite to generate them.'),
    });
  }

  ngAfterViewInit(): void {
    this.chartCanvases.changes.subscribe(() => this.tryBuildCharts());
    this.tryBuildCharts();
  }

  private tryBuildCharts(): void {
    if (this.chartsBuilt || !this.groups.length || !this.chartCanvases || this.chartCanvases.length !== this.groups.length) {
      return;
    }
    this.chartsBuilt = true;
    Chart.defaults.color = '#9aa3b5';
    Chart.defaults.font.family = "'IBM Plex Mono', monospace";
    Chart.defaults.font.size = 11;
    Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';

    this.chartCanvases.forEach((canvasRef, i) => {
      const group = this.groups[i];
      const ctx = canvasRef.nativeElement.getContext('2d');
      if (!ctx) return;
      if (group.kind === 'sweep') this.buildSweepChart(ctx, group.rows);
      else if (group.kind === 'bar') this.buildBarChart(ctx, group.rows);
      else this.buildScalabilityChart(ctx, group.rows);
    });
  }

  private buildGroups(): void {
    const byExperiment = new Map<string, ResultRow[]>();
    this.allRows.forEach((r) => {
      const exp = r['experiment'];
      if (!byExperiment.has(exp)) byExperiment.set(exp, []);
      byExperiment.get(exp)!.push(r);
    });

    this.groups = Array.from(byExperiment.entries()).map(([code, rows]) => ({
      code,
      title: EXPERIMENT_TITLES[code] || code,
      kind: code === 'H_scalability' ? 'scalability' : code === 'G_QAOA_compare' ? 'bar' : 'sweep',
      rows,
    }));
  }

  private buildSweepChart(ctx: CanvasRenderingContext2D, rows: ResultRow[]): void {
    const sorted = [...rows].sort((a, b) => parseFloat(a['noise_strength']) - parseFloat(b['noise_strength']));
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: sorted.map((r) => r['noise_strength']),
        datasets: [
          {
            label: 'Raw',
            data: sorted.map((r) => Math.abs(parseFloat(r['raw']) - parseFloat(r['ideal']))),
            borderColor: CHART_COLORS.raw,
            backgroundColor: CHART_COLORS.raw,
            tension: 0.3,
            pointRadius: 3,
          },
          {
            label: 'Mitigated',
            data: sorted.map((r) => Math.abs(parseFloat(r['mitigated']) - parseFloat(r['ideal']))),
            borderColor: CHART_COLORS.mitigated,
            backgroundColor: CHART_COLORS.mitigated,
            tension: 0.3,
            pointRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { boxWidth: 10 } } },
        scales: {
          x: { title: { display: true, text: 'noise strength' }, grid: { display: false } },
          y: { title: { display: true, text: 'absolute error' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        },
      },
    });
  }

  private buildBarChart(ctx: CanvasRenderingContext2D, rows: ResultRow[]): void {
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: rows.map((r) => r['technique']),
        datasets: [
          {
            label: 'error reduction (%)',
            data: rows.map((r) => parseFloat(r['error_reduction_pct'])),
            backgroundColor: '#8b93f8',
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { title: { display: true, text: 'error reduction (%)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        },
      },
    });
  }

  private buildScalabilityChart(ctx: CanvasRenderingContext2D, rows: ResultRow[]): void {
    const byTechnique = new Map<string, { n: number; t: number }[]>();
    rows.forEach((r) => {
      const match = (r['extra'] || '').match(/n_qubits=(\d+)/);
      if (!match) return;
      const n = parseInt(match[1], 10);
      const t = parseFloat(r['wall_time_s']);
      if (!byTechnique.has(r['technique'])) byTechnique.set(r['technique'], []);
      byTechnique.get(r['technique'])!.push({ n, t });
    });

    const entries = Array.from(byTechnique.entries());
    const datasets = entries.map(([technique, pts], i) => {
      pts.sort((a, b) => a.n - b.n);
      return {
        label: technique,
        data: pts.map((p) => p.t),
        borderColor: LINE_COLORS[i % LINE_COLORS.length],
        backgroundColor: LINE_COLORS[i % LINE_COLORS.length],
        tension: 0.3,
        pointRadius: 3,
      };
    });
    const labels = (entries[0]?.[1] ?? []).map((p) => p.n);

    new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        plugins: { legend: { labels: { boxWidth: 10 } } },
        scales: {
          x: { title: { display: true, text: 'qubit count' }, grid: { display: false } },
          y: { title: { display: true, text: 'wall-clock time (s)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        },
      },
    });
  }

  setFilter(exp: string): void {
    this.activeFilter = exp;
    this.visibleRows = exp === 'all' ? this.allRows : this.allRows.filter((r) => r['experiment'] === exp);
  }

  figureUrl(name: string): string {
    return this.api.figureUrl(name);
  }
}
