import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Technique {
  name: string;
  short: string;
  description: string;
  benchmark: string;
  metric: string;
  overhead: string;
  noise_type: string;
  incompatible_benchmarks: string[];
}

export interface Benchmark {
  name: string;
  n_qubits: number;
  min_qubits: number;
  max_qubits: number;
  description: string;
}

export interface RunResult {
  technique: string;
  technique_name: string;
  noise_strength: number;
  shots: number;
  wall_time_s: number;
  benchmark: string;
  benchmark_key: string;
  n_qubits: number;
  observable: string;
  ideal: number;
  raw: number;
  mitigated: number;
  error_reduction_pct: number;
  raw_error: number;
  mitigated_error: number;
  detail: Record<string, any>;
  notes: string[];
  gate_counts: Record<string, number>;
  circuit_depth: number;
  circuit_diagram: string | null;
}

export interface ResultsResponse {
  count: number;
  rows: Record<string, string>[];
}

export interface FiguresResponse {
  figures: string[];
}

export interface ChatMessage {
  role: string;
  content: string;
}

export interface ChatResponse {
  answer: string;
  used_llm: boolean;
  sources: { source: string; title: string }[];
}

export interface ExplainResponse {
  explanation: string;
  used_llm: boolean;
}

export interface TechniqueRanking {
  technique: string;
  benchmark: string;
  n_qubits: number;
  predicted_mitigated_error: number;
  rank: number;
}

export interface PredictResponse {
  noise_strength: number;
  ranking: TechniqueRanking[];
  recommended: string;
  model_metrics: string;
  caveat: string;
}

export interface HealthResponse {
  status: string;
  llm_available: boolean;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  // Same-origin: the Angular build is served by the FastAPI backend.
  private base = '';

  constructor(private http: HttpClient) {}

  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.base}/api/health`);
  }

  getTechniques(): Observable<Record<string, Technique>> {
    return this.http.get<Record<string, Technique>>(`${this.base}/api/techniques`);
  }

  getBenchmarks(): Observable<Record<string, Benchmark>> {
    return this.http.get<Record<string, Benchmark>>(`${this.base}/api/benchmarks`);
  }

  runExperiment(
    technique: string,
    noiseStrength: number,
    shots: number,
    benchmark?: string,
    nQubits?: number
  ): Observable<RunResult> {
    return this.http.post<RunResult>(`${this.base}/api/run`, {
      technique,
      noise_strength: noiseStrength,
      shots,
      benchmark: benchmark ?? null,
      n_qubits: nQubits ?? null,
    });
  }

  getResults(experiment?: string): Observable<ResultsResponse> {
    const url = experiment
      ? `${this.base}/api/results?experiment=${encodeURIComponent(experiment)}`
      : `${this.base}/api/results`;
    return this.http.get<ResultsResponse>(url);
  }

  getFigures(): Observable<FiguresResponse> {
    return this.http.get<FiguresResponse>(`${this.base}/api/figures`);
  }

  figureUrl(filename: string): string {
    return `${this.base}/api/figures/${filename}`;
  }

  chat(message: string, history: ChatMessage[]): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.base}/api/chat`, { message, history });
  }

  explainResult(result: RunResult): Observable<ExplainResponse> {
    return this.http.post<ExplainResponse>(`${this.base}/api/explain`, { result });
  }

  predict(noiseStrength: number, nQubits?: number): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(`${this.base}/api/predict`, {
      noise_strength: noiseStrength,
      n_qubits: nQubits ?? null,
    });
  }
}
