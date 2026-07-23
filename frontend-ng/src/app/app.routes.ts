import { Routes } from '@angular/router';
import { OverviewComponent } from './pages/overview/overview.component';
import { LiveComponent } from './pages/live/live.component';
import { AdvisorComponent } from './pages/advisor/advisor.component';
import { ResultsComponent } from './pages/results/results.component';
import { TechniquesComponent } from './pages/techniques/techniques.component';
import { AboutComponent } from './pages/about/about.component';

export const routes: Routes = [
  { path: '', component: OverviewComponent, title: 'QEM Lab — Overview' },
  { path: 'live', component: LiveComponent, title: 'QEM Lab — Live Experiment' },
  { path: 'advisor', component: AdvisorComponent, title: 'QEM Lab — Advisor' },
  { path: 'results', component: ResultsComponent, title: 'QEM Lab — Results' },
  { path: 'techniques', component: TechniquesComponent, title: 'QEM Lab — Techniques' },
  { path: 'about', component: AboutComponent, title: 'QEM Lab — About' },
  { path: '**', redirectTo: '' },
];
