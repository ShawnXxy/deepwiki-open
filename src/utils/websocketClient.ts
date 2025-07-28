/**
 * WebSocket client for chat completions
 * This replaces the HTTP streaming endpoint with a WebSocket connection
 */

import { getServerBaseUrl, getWebSocketUrl, getTimeoutConfig } from './networkConfig';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatCompletionRequest {
  repo_url: string;
  messages: ChatMessage[];
  filePath?: string;
  token?: string;
  type?: string;
  provider?: string;
  model?: string;
  language?: string;
  excluded_dirs?: string;
  excluded_files?: string;
}

/**
 * Creates a WebSocket connection for chat completions
 * @param request The chat completion request
 * @param onMessage Callback for received messages
 * @param onError Callback for errors
 * @param onClose Callback for when the connection closes
 * @returns The WebSocket connection
 */
export const createChatWebSocket = (
  request: ChatCompletionRequest,
  onMessage: (message: string) => void,
  onError: (error: Event) => void,
  onClose: () => void
): WebSocket => {
  // Create WebSocket connection with improved error handling
  const wsUrl = getWebSocketUrl();
  const timeouts = getTimeoutConfig();
  
  console.log(`Attempting WebSocket connection to: ${wsUrl}`);
  console.log(`Using timeout config:`, timeouts);
  
  const ws = new WebSocket(wsUrl);
  
  // Set up event handlers with timeout protection
  let connectionTimeout: NodeJS.Timeout | undefined;
  
  ws.onopen = () => {
    console.log('WebSocket connection established');
    if (connectionTimeout) {
      clearTimeout(connectionTimeout);
    }
    // Send the request as JSON
    try {
      ws.send(JSON.stringify(request));
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      onError(error as Event);
    }
  };
  
  ws.onmessage = (event) => {
    // Call the message handler with the received text
    onMessage(event.data);
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    if (connectionTimeout) {
      clearTimeout(connectionTimeout);
    }
    onError(error);
  };
  
  ws.onclose = (event) => {
    console.log(`WebSocket connection closed: code=${event.code}, reason=${event.reason}`);
    if (connectionTimeout) {
      clearTimeout(connectionTimeout);
    }
    onClose();
  };
  
  // Set a connection timeout to detect network issues
  connectionTimeout = setTimeout(() => {
    if (ws.readyState === WebSocket.CONNECTING) {
      console.error('WebSocket connection timeout');
      ws.close();
      onError(new Event('timeout'));
    }
  }, timeouts.connectionTimeout);
  
  return ws;
};

/**
 * Closes a WebSocket connection
 * @param ws The WebSocket connection to close
 */
export const closeWebSocket = (ws: WebSocket | null): void => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
};
