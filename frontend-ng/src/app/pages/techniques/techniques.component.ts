import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService, Technique } from '../../core/api.service';

@Component({
  selector: 'app-techniques',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './techniques.component.html',
})
export class TechniquesComponent implements OnInit {
  techniques: Technique[] = [];
  loading = true;
  error = '';

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getTechniques().subscribe({
      next: (t) => {
        this.techniques = Object.values(t);
        this.loading = false;
      },
      error: () => {
        this.error = 'Could not reach the backend API.';
        this.loading = false;
      },
    });
  }
}
