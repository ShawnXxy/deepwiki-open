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

// Console method mapping
const CONSOLE_METHODS: Record<LogLevel, 'debug' | 'log' | 'warn' | 'error'> = {
  debug: 'debug',
  info: 'log',
  warn: 'warn',
  error: 'error'
};

// Buffer for batching logs
const logBuffer: LogEntry[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_INTERVAL = 2000; // 2 seconds
const MAX_BUFFER_SIZE = 50;

// Flag to track if backend logging is available
let backendLoggingAvailable = true;

/**
 * Log entry to browser console
 */
function logToConsole(entry: LogEntry): void {
  const method = CONSOLE_METHODS[entry.level];
  console[method](`[${entry.level.toUpperCase()}]`, entry.message, entry.context || '');
}

/**
 * Flush buffered logs to the backend
 */
async function flushLogs(): Promise<void> {
  if (logBuffer.length === 0) return;
  
  const logsToSend = [...logBuffer];
  logBuffer.length = 0;
  
  if (!backendLoggingAvailable) {
    return; // Already logged to console in addLog
  }
  
  try {
    const response = await fetch(`${getServerBaseUrl()}/log/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ logs: logsToSend }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch {
    console.warn('[Logger] Backend logging unavailable, falling back to console only');
    backendLoggingAvailable = false;
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
 * Add a log entry - outputs to console immediately and buffers for backend
 */
function addLog(level: LogLevel, message: string, context?: Record<string, unknown>): void {
  const entry: LogEntry = { level, message, context };
  
  // Always log to browser console for immediate visibility
  logToConsole(entry);
  
  // Buffer for backend logging
  logBuffer.push(entry);
  
  // Flush if buffer is full, otherwise schedule
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
  debug: (message: string, context?: Record<string, unknown>) => addLog('debug', message, context),
  info: (message: string, context?: Record<string, unknown>) => addLog('info', message, context),
  warn: (message: string, context?: Record<string, unknown>) => addLog('warn', message, context),
  error: (message: string, context?: Record<string, unknown>) => addLog('error', message, context),

  /** Force flush all buffered logs immediately */
  async flush(): Promise<void> {
    if (flushTimer) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    await flushLogs();
  },

  /** Check if backend logging is available */
  isBackendAvailable: () => backendLoggingAvailable,

  /** Reset backend availability (e.g., after reconnection) */
  resetBackendAvailability: () => { backendLoggingAvailable = true; }
};

// Flush logs before page unload using sendBeacon for reliable delivery
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    if (logBuffer.length > 0) {
      try {
        navigator.sendBeacon(
          `${getServerBaseUrl()}/log/batch`,
          JSON.stringify({ logs: [...logBuffer] })
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
