import { Injectable } from '@angular/core';
import { ChatMessage } from './api.service';

export interface DisplayMessage extends ChatMessage {
  sources?: { source: string; title: string }[];
  usedLlm?: boolean;
}

/**
 * Holds chat history in a singleton service (rather than component state)
 * so the conversation survives Angular route navigation -- the chat widget
 * lives in the app shell, outside <router-outlet>, but this keeps the data
 * layer separate from where the widget happens to be rendered.
 */
@Injectable({ providedIn: 'root' })
export class ChatStateService {
  isOpen = false;
  messages: DisplayMessage[] = [
    {
      role: 'assistant',
      content:
        "Hi! I'm the QEM Lab assistant. Ask me about quantum error mitigation, NISQ noise, " +
        "any of the six techniques on this site, or how the ML/RAG parts of this project work.",
    },
  ];

  toggle(): void {
    this.isOpen = !this.isOpen;
  }

  addMessage(msg: DisplayMessage): void {
    this.messages.push(msg);
  }

  historyForApi(): ChatMessage[] {
    return this.messages.map((m) => ({ role: m.role, content: m.content }));
  }
}
