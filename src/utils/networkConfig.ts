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
 * Check if we're running in a cloud environment where only one port is exposed
 * In cloud deployments (Azure Container Apps, Azure App Service, Vercel, etc.),
 * only the frontend port is exposed externally, so backend port 8001 is not accessible.
 */
export const isCloudEnvironment = (): boolean => {
  if (typeof window === 'undefined') return false;
  
  const hostname = window.location.hostname;
  
  // List of cloud hosting patterns where WebSocket to port 8001 won't work
  const cloudPatterns = [
    '.azurecontainerapps.io',    // Azure Container Apps
    '.azurewebsites.net',         // Azure App Service
    '.azure-api.net',             // Azure API Management
    '.cloudapp.azure.com',        // Azure VMs
    '.vercel.app',                // Vercel
    '.netlify.app',               // Netlify
    '.herokuapp.com',             // Heroku
    '.railway.app',               // Railway
    '.render.com',                // Render
    '.fly.dev',                   // Fly.io
    '.amplifyapp.com',            // AWS Amplify
    '.cloudfront.net',            // AWS CloudFront
    '.pages.dev',                 // Cloudflare Pages
    '.workers.dev',               // Cloudflare Workers
  ];
  
  // Check if hostname matches any cloud pattern
  return cloudPatterns.some(pattern => hostname.endsWith(pattern));
};

/**
 * Check if we're running in a network environment (not localhost)
 * This is kept for backward compatibility but isCloudEnvironment is preferred
 */
export const isNetworkEnvironment = (): boolean => {
  if (typeof window === 'undefined') return false;
  
  const hostname = window.location.hostname;
  return hostname !== 'localhost' && hostname !== '127.0.0.1';
};

/**
 * Check if WebSocket should be used for backend communication
 * WebSocket is only reliable when backend port (8001) is directly accessible.
 * In cloud deployments, only the frontend port is exposed, so we must use HTTP proxy.
 * In local/Docker environments (even with non-localhost hostnames), WebSocket works.
 */
export const shouldUseWebSocket = (): boolean => {
  const isCloud = isCloudEnvironment();
  const hostname = typeof window !== 'undefined' ? window.location.hostname : 'unknown';
  
  // Log for debugging
  console.log(`[NetworkConfig] hostname: ${hostname}, isCloud: ${isCloud}, useWebSocket: ${!isCloud}`);
  
  // Use WebSocket unless we're in a cloud environment
  return !isCloud;
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
    requestTimeout: isNetwork ? 600000 : 300000, // 10 min vs 5 min for large repos
  };
};
