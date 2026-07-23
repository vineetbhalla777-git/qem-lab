import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService, PredictResponse } from '../../core/api.service';

@Component({
  selector: 'app-advisor',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './advisor.component.html',
})
export class AdvisorComponent {
  noiseStrength = 0.2;
  qubitOverride: number | null = null;

  loading = false;
  error = '';
  result: PredictResponse | null = null;

  constructor(private api: ApiService, private router: Router) {}

  predict(): void {
    this.loading = true;
    this.error = '';
    this.api.predict(this.noiseStrength, this.qubitOverride ?? undefined).subscribe({
      next: (r) => {
        this.result = r;
        this.loading = false;
      },
      error: (e) => {
        this.error = e?.error?.detail || 'Prediction failed.';
        this.loading = false;
      },
    });
  }

  maxError(): number {
    if (!this.result) return 1;
    return Math.max(...this.result.ranking.map((r) => r.predicted_mitigated_error), 0.001);
  }

  barWidth(value: number): string {
    return `${Math.min(100, (value / this.maxError()) * 100)}%`;
  }

  tryInLive(technique: string): void {
    this.router.navigate(['/live'], { queryParams: { technique, noise: this.noiseStrength } });
  }
}
