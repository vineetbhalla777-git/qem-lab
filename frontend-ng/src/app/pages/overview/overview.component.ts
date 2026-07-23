import { AfterViewInit, Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-overview',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './overview.component.html',
})
export class OverviewComponent implements AfterViewInit, OnDestroy {
  @ViewChild('heroCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  private rafId = 0;

  ngAfterViewInit(): void {
    this.initHeroAnimation();
  }

  ngOnDestroy(): void {
    cancelAnimationFrame(this.rafId);
  }

  private initHeroAnimation(): void {
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = 0;
    let height = 0;
    let dpr = window.devicePixelRatio || 1;

    const resize = () => {
      dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height || 280;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener('resize', resize);

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let t = 0;

    const drawGrid = () => {
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 1;
      const rows = 6;
      for (let i = 0; i <= rows; i++) {
        const y = (height / rows) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    };

    const traceY = (x: number, phase: number, noiseAmount: number): number => {
      const cx = (x / width) * Math.PI * 4 + phase;
      const base = Math.sin(cx) * 0.62 + Math.sin(cx * 2.3 + 1.2) * 0.18;
      let n = 0;
      if (noiseAmount > 0) {
        n = (Math.sin(x * 0.19 + phase * 3.1) * 0.5 + Math.sin(x * 0.53 + phase * 5.7) * 0.5) * noiseAmount;
      }
      const y = base + n;
      return height / 2 - y * (height * 0.34);
    };

    const drawTrace = (phase: number, noiseAmount: number, color: string, lineWidth: number, glow: boolean) => {
      ctx.beginPath();
      const step = 3;
      for (let x = 0; x <= width; x += step) {
        const y = traceY(x, phase, noiseAmount);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      if (glow) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
      } else {
        ctx.shadowBlur = 0;
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    const frame = () => {
      ctx.clearRect(0, 0, width, height);
      drawGrid();

      const cycle = (Math.sin(t * 0.35) + 1) / 2;
      const rawNoise = 0.42;
      const mitigatedNoise = 0.42 * (1 - cycle) * 0.28;
      const phase = t;

      drawTrace(phase, rawNoise, 'rgba(242,166,90,0.55)', 1.6, false);
      drawTrace(phase, mitigatedNoise, '#8b93f8', 2.2, true);
      drawTrace(phase, 0, '#4fd8c4', 1.4, false);

      t += reduceMotion ? 0 : 0.014;
      this.rafId = requestAnimationFrame(frame);
    };

    frame();
  }
}
