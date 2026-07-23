import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { ApiService, Benchmark, RunResult, Technique } from '../../core/api.service';

interface DetailRow {
  key: string;
  value: string;
}

@Component({
  selector: 'app-live',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './live.component.html',
})
export class LiveComponent implements OnInit {
  techniques: Record<string, Technique> = {};
  techniqueKeys: string[] = [];
  selectedTechnique = '';

  benchmarks: Record<string, Benchmark> = {};
  availableBenchmarkKeys: string[] = [];
  selectedBenchmark = '';
  qubitOptions: number[] = [];
  selectedQubits = 2;

  noiseStrength = 0.15;
  shots = 1500;

  loadingTechniques = true;
  loadError = '';

  running = false;
  runError = '';
  result: RunResult | null = null;
  detailRows: DetailRow[] = [];

  explaining = false;
  explanation = '';
  explanationUsedLlm = false;

  private presetTechnique: string | null = null;

  constructor(private api: ApiService, private route: ActivatedRoute) {}

  private benchmarksLoaded = false;
  private techniquesLoaded = false;

  ngOnInit(): void {
    const params = this.route.snapshot.queryParamMap;
    this.presetTechnique = params.get('technique');
    const presetNoise = params.get('noise');
    if (presetNoise) this.noiseStrength = parseFloat(presetNoise);

    this.api.getBenchmarks().subscribe({
      next: (b) => {
        this.benchmarks = b;
        this.benchmarksLoaded = true;
        this.tryInitializeSelection();
      },
    });

    this.api.getTechniques().subscribe({
      next: (t) => {
        this.techniques = t;
        this.techniqueKeys = Object.keys(t);
        this.selectedTechnique =
          this.presetTechnique && t[this.presetTechnique] ? this.presetTechnique : this.techniqueKeys[0] ?? '';
        this.loadingTechniques = false;
        this.techniquesLoaded = true;
        this.tryInitializeSelection();
      },
      error: () => {
        this.loadError = 'Could not reach the backend API. Is the server running?';
        this.loadingTechniques = false;
      },
    });
  }

  private tryInitializeSelection(): void {
    if (this.benchmarksLoaded && this.techniquesLoaded) {
      this.onTechniqueChange();
    }
  }

  get currentDescription(): string {
    return this.techniques[this.selectedTechnique]?.description ?? '';
  }

  get currentNoiseType(): string {
    return this.techniques[this.selectedTechnique]?.noise_type ?? '';
  }

  /** Benchmarks the currently selected technique is known to support. */
  private computeAvailableBenchmarks(): string[] {
    const t = this.techniques[this.selectedTechnique];
    if (!t) return [];
    const incompatible = new Set(t.incompatible_benchmarks || []);
    return Object.keys(this.benchmarks).filter((key) => !incompatible.has(key));
  }

  onTechniqueChange(): void {
    this.availableBenchmarkKeys = this.computeAvailableBenchmarks();
    const technique = this.techniques[this.selectedTechnique];
    const defaultBenchmark = technique?.benchmark ?? this.availableBenchmarkKeys[0];
    this.selectedBenchmark = this.availableBenchmarkKeys.includes(defaultBenchmark)
      ? defaultBenchmark
      : this.availableBenchmarkKeys[0];
    this.onBenchmarkChange();
  }

  onBenchmarkChange(): void {
    const b = this.benchmarks[this.selectedBenchmark];
    if (!b) return;
    this.qubitOptions = [];
    for (let n = b.min_qubits; n <= b.max_qubits; n++) this.qubitOptions.push(n);
    this.selectedQubits = b.n_qubits && this.qubitOptions.includes(b.n_qubits) ? b.n_qubits : this.qubitOptions[0];
  }

  verdict(): { color: string; label: string } {
    const pct = this.result?.error_reduction_pct ?? 0;
    if (pct > 5) return { color: '#4fd8c4', label: 'improved' };
    if (pct < -5) return { color: '#f76c6c', label: 'worsened' };
    return { color: '#f2a65a', label: 'roughly unchanged' };
  }

  barWidth(value: number): string {
    if (!this.result) return '0%';
    const maxAbs = Math.max(
      Math.abs(this.result.ideal),
      Math.abs(this.result.raw),
      Math.abs(this.result.mitigated),
      0.001
    );
    return `${Math.min(100, (Math.abs(value) / maxAbs) * 100)}%`;
  }

  runExperiment(): void {
    this.running = true;
    this.runError = '';
    this.explanation = '';
    this.api
      .runExperiment(this.selectedTechnique, this.noiseStrength, this.shots, this.selectedBenchmark, this.selectedQubits)
      .subscribe({
        next: (r) => {
          this.result = r;
          this.detailRows = Object.entries(r.detail || {})
            .filter(([k]) => !['ideal_distribution', 'raw_distribution', 'mitigated_distribution'].includes(k))
            .map(([k, v]) => ({
              key: k,
              value: Array.isArray(v)
                ? v.map((x) => Number(x).toFixed(3)).join(', ')
                : typeof v === 'number'
                ? v.toFixed(4)
                : String(v),
            }));
          this.running = false;
        },
        error: (e) => {
          this.runError = e?.error?.detail || 'Simulation failed.';
          this.running = false;
        },
      });
  }

  gateCountEntries(): { gate: string; count: number }[] {
    if (!this.result) return [];
    return Object.entries(this.result.gate_counts || {})
      .map(([gate, count]) => ({ gate, count }))
      .sort((a, b) => b.count - a.count);
  }

  explainResult(): void {
    if (!this.result) return;
    this.explaining = true;
    this.explanation = '';
    this.api.explainResult(this.result).subscribe({
      next: (r) => {
        this.explanation = r.explanation;
        this.explanationUsedLlm = r.used_llm;
        this.explaining = false;
      },
      error: () => {
        this.explanation = 'Could not generate an explanation right now.';
        this.explaining = false;
      },
    });
  }
}
