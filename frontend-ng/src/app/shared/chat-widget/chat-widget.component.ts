import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { ChatStateService } from '../../core/chat-state.service';

@Component({
  selector: 'app-chat-widget',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat-widget.component.html',
})
export class ChatWidgetComponent {
  draft = '';
  sending = false;

  constructor(public chatState: ChatStateService, private api: ApiService) {}

  toggle(): void {
    this.chatState.toggle();
  }

  send(): void {
    const text = this.draft.trim();
    if (!text || this.sending) return;

    this.chatState.addMessage({ role: 'user', content: text });
    this.draft = '';
    this.sending = true;

    this.api.chat(text, this.chatState.historyForApi()).subscribe({
      next: (res) => {
        this.chatState.addMessage({
          role: 'assistant',
          content: res.answer,
          sources: res.sources,
          usedLlm: res.used_llm,
        });
        this.sending = false;
      },
      error: () => {
        this.chatState.addMessage({
          role: 'assistant',
          content: "Sorry, I couldn't reach the backend just now. Is the server running?",
        });
        this.sending = false;
      },
    });
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }
}
