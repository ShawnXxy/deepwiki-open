/**
 * Frontend Logger - Sends logs to backend for persistent storage
 * 
 * Features:
 * - Level-aware deduplication: Different windows per level (aligned with backend)
 * - Batching: Logs are buffered and sent to backend in batches
 * - Auto-retry: Periodically retries backend if unavailable
 * - Fallback: Falls back to console logging if backend is unavailable
 * - All levels logged: No level filtering, all logs captured
 * 
 * Logs are written to frontend-yymmdd.log on the server with daily rotation.
 */

import { getServerBaseUrl } from './networkConfig';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  context?: Record<string, unknown>;
  timestamp?: string;
}

// Console method mapping
const CONSOLE_METHODS: Record<LogLevel, 'debug' | 'log' | 'warn' | 'error'> = {
  debug: 'debug',
  info: 'log',
  warn: 'warn',
  error: 'error'
};

// Console color styling for better visibility
const CONSOLE_STYLES: Record<LogLevel, string> = {
  debug: 'color: #888',
  info: 'color: #2196F3',
  warn: 'color: #FF9800; font-weight: bold',
  error: 'color: #F44336; font-weight: bold'
};

// Buffer for batching logs
const logBuffer: LogEntry[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_INTERVAL = 2000; // 2 seconds
const MAX_BUFFER_SIZE = 50;

// Level-aware deduplication windows (aligned with backend SmartLogFilter)
const DEDUP_WINDOWS: Record<LogLevel, number> = {
  error: 2000,    // 2s - errors always important
  warn: 10000,    // 10s - warnings semi-important
  info: 30000,    // 30s - info can be deduplicated more aggressively
  debug: 60000    // 60s - debug deduplicated most aggressively
};

// Deduplication tracking
interface DedupEntry {
  lastTime: number;
  count: number;
  level: LogLevel;
}
const dedupMap = new Map<string, DedupEntry>();

// Backend log shipping disabled — frontend and backend are isolated.
// All frontend logs go to browser console only.
const backendLoggingAvailable = false;

/**
 * Generate deduplication key from log entry
 * Uses truncated message for performance (no hashing)
 */
function getDedupKey(entry: LogEntry): string {
  const msg = entry.message.substring(0, 100); // Truncate for performance
  const contextStr = entry.context ? JSON.stringify(entry.context).substring(0, 50) : '';
  return `${entry.level}:${msg}:${contextStr}`;
}

/**
 * Check if message should be logged (level-aware deduplication)
 * Returns: { shouldLog: boolean, repeatCount?: number }
 */
function checkDedup(entry: LogEntry): { shouldLog: boolean; repeatCount?: number } {
  const key = getDedupKey(entry);
  const now = Date.now();
  const existing = dedupMap.get(key);
  const dedupWindow = DEDUP_WINDOWS[entry.level];
  
  if (existing && (now - existing.lastTime) < dedupWindow) {
    // Same message within window - increment count, don't log
    existing.count++;
    existing.lastTime = now;
    return { shouldLog: false };
  }
  
  // Either new message or window expired
  const repeatCount = existing?.count;
  dedupMap.set(key, { lastTime: now, count: 1, level: entry.level });
  
  // Clean up old entries periodically (every 100 entries)
  if (dedupMap.size > 500) {
    const cutoff = now - 120000; // 2 minutes
    const keysToDelete: string[] = [];
    dedupMap.forEach((v, k) => {
      if (v.lastTime < cutoff) {
        keysToDelete.push(k);
      }
    });
    keysToDelete.forEach(k => dedupMap.delete(k));
  }
  
  return { shouldLog: true, repeatCount: repeatCount && repeatCount > 1 ? repeatCount : undefined };
}

/**
 * Log entry to browser console with styled output
 */
function logToConsole(entry: LogEntry, repeatCount?: number): void {
  const method = CONSOLE_METHODS[entry.level];
  const style = CONSOLE_STYLES[entry.level];
  const prefix = repeatCount 
    ? `[${entry.level.toUpperCase()}] (×${repeatCount})` 
    : `[${entry.level.toUpperCase()}]`;
  
  // Use styled console output for better visibility
  if (entry.context && Object.keys(entry.context).length > 0) {
    console[method](`%c${prefix}`, style, entry.message, entry.context);
  } else {
    console[method](`%c${prefix}`, style, entry.message);
  }
}

/**
 * Check if backend should be retried.
 * Always returns false — frontend logs to console only (isolated from backend).
 */
function shouldRetryBackend(): boolean {
  return false;
}

/**
 * Flush buffered logs to the backend
 */
async function flushLogs(): Promise<void> {
  if (logBuffer.length === 0) return;
  
  const logsToSend = [...logBuffer];
  logBuffer.length = 0;
  
  // Check if we should try backend
  if (!shouldRetryBackend()) {
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
    
    // Success - backend shipping disabled (unused)
    if (!backendLoggingAvailable) {
      console.log('%c[Logger] Backend logging restored', 'color: #4CAF50');
    }
  } catch {
    if (backendLoggingAvailable) {
      console.warn('%c[Logger] Backend logging unavailable, using console only (will retry in 30s)', 'color: #FF9800');
    }
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
 * Uses level-aware deduplication to prevent identical log spam
 */
function addLog(level: LogLevel, message: string, context?: Record<string, unknown>): void {
  // Guard against SSR - only log in browser
  if (typeof window === 'undefined') return;
  
  const entry: LogEntry = { 
    level, 
    message, 
    context,
    timestamp: new Date().toISOString()
  };
  
  // Check deduplication
  const { shouldLog, repeatCount } = checkDedup(entry);
  
  if (!shouldLog) {
    // Skip this log (duplicate within window)
    return;
  }
  
  // If there was a repeat count, add it to the message
  const logEntry: LogEntry = repeatCount 
    ? { ...entry, message: `${message} (×${repeatCount} previous)` }
    : entry;
  
  // Always log to browser console for immediate visibility
  logToConsole(entry, repeatCount);
  
  // Buffer for backend logging
  logBuffer.push(logEntry);
  
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
 * 
 * Usage:
 *   import logger from '@/utils/logger';
 *   logger.info('User action', { userId: 123 });
 *   logger.error('Failed to load', { error: err.message });
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

  /** Reset backend availability (no-op — backend shipping disabled) */
  resetBackendAvailability: () => {
    // No-op: frontend and backend are isolated
  },
  
  /** Get deduplication stats (for debugging) */
  getStats: () => ({
    bufferSize: logBuffer.length,
    dedupEntries: dedupMap.size,
    backendAvailable: backendLoggingAvailable
  })
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
