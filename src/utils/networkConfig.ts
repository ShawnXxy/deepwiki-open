/**
 * Network configuration utilities for handling localhost vs network access
 */

/**
 * Get the appropriate server base URL for the current environment
 * This handles both localhost and network access scenarios
 */
export const getServerBaseUrl = (): string => {
  // For server-side or build time, use environment variable if available
  if (typeof process !== 'undefined' && process.env?.SERVER_BASE_URL) {
    return process.env.SERVER_BASE_URL;
  }
  
  // For client-side, derive from current window location
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const hostname = window.location.hostname;
    const port = '8001'; // Backend port
    
    // Special handling for localhost development
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return `${protocol}//${hostname}:${port}`;
    }
    
    // For network access, use the same hostname as the frontend
    return `${protocol}//${hostname}:${port}`;
  }
  
  // Fallback to localhost
  return 'http://localhost:8001';
};

/**
 * Get WebSocket URL from HTTP URL
 */
export const getWebSocketUrl = (httpUrl?: string): string => {
  const baseUrl = httpUrl || getServerBaseUrl();
  // Replace http:// with ws:// or https:// with wss://
  const wsBaseUrl = baseUrl.replace(/^http/, 'ws');
  return `${wsBaseUrl}/ws/chat`;
};

/**
 * Check if we're running in a network environment (not localhost)
 */
export const isNetworkEnvironment = (): boolean => {
  if (typeof window === 'undefined') return false;
  
  const hostname = window.location.hostname;
  return hostname !== 'localhost' && hostname !== '127.0.0.1';
};

/**
 * Get appropriate timeout values based on environment
 */
export const getTimeoutConfig = () => {
  const isNetwork = isNetworkEnvironment();
  
  return {
    // Connection timeout: longer for network access
    connectionTimeout: isNetwork ? 15000 : 10000,
    // Request timeout: much longer for network access due to potential latency
    requestTimeout: isNetwork ? 300000 : 180000, // 5 min vs 3 min
  };
};
