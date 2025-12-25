/**
 * Frontend Logger - Sends logs to backend for persistent storage
 * 
 * Logs are written to frontend-yymmdd.log on the server with daily rotation.
 * Falls back to console logging if backend is unavailable.
 */

import { getServerBaseUrl } from './networkConfig';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  context?: Record<string, unknown>;
}

// Buffer for batching logs
const logBuffer: LogEntry[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_INTERVAL = 2000; // 2 seconds
const MAX_BUFFER_SIZE = 50;

// Flag to track if backend logging is available
let backendLoggingAvailable = true;

/**
 * Get the API base URL for logging
 */
function getApiUrl(): string {
  return getServerBaseUrl();
}

/**
 * Flush buffered logs to the backend
 */
async function flushLogs(): Promise<void> {
  if (logBuffer.length === 0) return;
  
  const logsToSend = [...logBuffer];
  logBuffer.length = 0;
  
  if (!backendLoggingAvailable) {
    // Fall back to console
    logsToSend.forEach(entry => {
      const consoleMethod = entry.level === 'warn' ? 'warn' : 
                           entry.level === 'error' ? 'error' : 
                           entry.level === 'debug' ? 'debug' : 'log';
      console[consoleMethod](`[${entry.level.toUpperCase()}]`, entry.message, entry.context || '');
    });
    return;
  }
  
  try {
    const response = await fetch(`${getApiUrl()}/log/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ logs: logsToSend }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch {
    // Backend logging failed, fall back to console
    console.warn('[Logger] Backend logging unavailable, falling back to console');
    backendLoggingAvailable = false;
    
    logsToSend.forEach(entry => {
      const consoleMethod = entry.level === 'warn' ? 'warn' : 
                           entry.level === 'error' ? 'error' : 
                           entry.level === 'debug' ? 'debug' : 'log';
      console[consoleMethod](`[${entry.level.toUpperCase()}]`, entry.message, entry.context || '');
    });
  }
}

/**
 * Schedule a flush of the log buffer
 */
function scheduleFlush(): void {
  if (flushTimer === null) {
    flushTimer = setTimeout(() => {
      flushTimer = null;
      flushLogs();
    }, FLUSH_INTERVAL);
  }
}

/**
 * Add a log entry to the buffer
 */
function addLog(level: LogLevel, message: string, context?: Record<string, unknown>): void {
  logBuffer.push({ level, message, context });
  
  // Immediately flush if buffer is full
  if (logBuffer.length >= MAX_BUFFER_SIZE) {
    if (flushTimer) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    flushLogs();
  } else {
    scheduleFlush();
  }
}

/**
 * Logger object with level-specific methods
 */
const logger = {
  /**
   * Log a debug message
   */
  debug(message: string, context?: Record<string, unknown>): void {
    addLog('debug', message, context);
  },

  /**
   * Log an info message
   */
  info(message: string, context?: Record<string, unknown>): void {
    addLog('info', message, context);
  },

  /**
   * Log a warning message
   */
  warn(message: string, context?: Record<string, unknown>): void {
    addLog('warn', message, context);
  },

  /**
   * Log an error message
   */
  error(message: string, context?: Record<string, unknown>): void {
    addLog('error', message, context);
  },

  /**
   * Force flush all buffered logs immediately
   */
  async flush(): Promise<void> {
    if (flushTimer) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    await flushLogs();
  },

  /**
   * Check if backend logging is available
   */
  isBackendAvailable(): boolean {
    return backendLoggingAvailable;
  },

  /**
   * Reset backend availability (e.g., after reconnection)
   */
  resetBackendAvailability(): void {
    backendLoggingAvailable = true;
  }
};

// Flush logs before page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    // Synchronous flush attempt
    if (logBuffer.length > 0) {
      const logsToSend = [...logBuffer];
      // Use sendBeacon for reliable delivery on page unload
      try {
        navigator.sendBeacon(
          `${getApiUrl()}/log/batch`,
          JSON.stringify({ logs: logsToSend })
        );
      } catch {
        // Ignore errors during unload
      }
    }
  });
}

export default logger;
export { logger };
export type { LogLevel, LogEntry };
