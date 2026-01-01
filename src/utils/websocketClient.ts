/**
 * WebSocket client for chat completions
 * This replaces the HTTP streaming endpoint with a WebSocket connection
 * NOTE: WebSocket is only used in localhost environments where port 8001 is accessible.
 * In cloud deployments, the HTTP proxy (/api/chat/stream) is used instead.
 */

import { getWebSocketUrl, getTimeoutConfig, shouldUseWebSocket } from './networkConfig';

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
  branch?: string;
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
 * @returns The WebSocket connection or null if WebSocket should not be used
 */
export const createChatWebSocket = (
  request: ChatCompletionRequest,
  onMessage: (message: string) => void,
  onError: (error: Event) => void,
  onClose: () => void
): WebSocket | null => {
  // Check if WebSocket should be used (only in localhost environments)
  if (!shouldUseWebSocket()) {
    console.log('Cloud environment detected, WebSocket not available - use HTTP proxy instead');
    // Trigger error callback to signal fallback to HTTP
    setTimeout(() => onError(new Event('websocket-unavailable')), 0);
    return null;
  }
  
  // Create WebSocket connection with improved error handling
  const wsUrl = getWebSocketUrl();
  const timeouts = getTimeoutConfig();
  
  console.log(`Attempting WebSocket connection to: ${wsUrl}`);
  console.log(`Using timeout config:`, timeouts);
  
  const ws = new WebSocket(wsUrl);
  
  // Set up event handlers with timeout protection
  // eslint-disable-next-line prefer-const
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
    // Filter out keepalive messages (HTML comments used to keep connection alive during embedding)
    const data = event.data;
    if (data && !data.startsWith('<!-- keepalive')) {
      // Call the message handler with the received text
      onMessage(data);
    }
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
