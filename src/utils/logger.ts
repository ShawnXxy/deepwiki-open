/**
 * Frontend Logger - Sends logs to backend for persistent storage
 * 
 * Features:
 * - Deduplication: Identical messages within time window are counted, not repeated
 * - Batching: Logs are buffered and sent to backend in batches
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

// Deduplication tracking
interface DedupEntry {
  lastTime: number;
  count: number;
}
const dedupMap = new Map<string, DedupEntry>();
const DEDUP_WINDOW_MS = 5000; // 5 seconds deduplication window

// Flag to track if backend logging is available
let backendLoggingAvailable = true;

/**
 * Generate deduplication key from log entry
 */
function getDedupKey(entry: LogEntry): string {
  const contextStr = entry.context ? JSON.stringify(entry.context) : '';
  return `${entry.level}:${entry.message}:${contextStr}`;
}

/**
 * Check if message should be logged (deduplication)
 * Returns: { shouldLog: boolean, repeatCount?: number }
 */
function checkDedup(entry: LogEntry): { shouldLog: boolean; repeatCount?: number } {
  const key = getDedupKey(entry);
  const now = Date.now();
  const existing = dedupMap.get(key);
  
  if (existing && (now - existing.lastTime) < DEDUP_WINDOW_MS) {
    // Same message within window - increment count, don't log
    existing.count++;
    existing.lastTime = now;
    return { shouldLog: false };
  }
  
  // Either new message or window expired
  const repeatCount = existing?.count;
  dedupMap.set(key, { lastTime: now, count: 1 });
  
  // Clean up old entries periodically
  if (dedupMap.size > 1000) {
    const cutoff = now - DEDUP_WINDOW_MS * 2;
    for (const [k, v] of dedupMap.entries()) {
      if (v.lastTime < cutoff) {
        dedupMap.delete(k);
      }
    }
  }
  
  return { shouldLog: true, repeatCount: repeatCount && repeatCount > 1 ? repeatCount : undefined };
}

/**
 * Log entry to browser console
 */
function logToConsole(entry: LogEntry, repeatCount?: number): void {
  const method = CONSOLE_METHODS[entry.level];
  const prefix = repeatCount ? `[${entry.level.toUpperCase()}] (repeated ${repeatCount}x)` : `[${entry.level.toUpperCase()}]`;
  console[method](prefix, entry.message, entry.context || '');
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
 * Uses deduplication to prevent identical log spam
 */
function addLog(level: LogLevel, message: string, context?: Record<string, unknown>): void {
  const entry: LogEntry = { level, message, context };
  
  // Check deduplication
  const { shouldLog, repeatCount } = checkDedup(entry);
  
  if (!shouldLog) {
    // Skip this log (duplicate within window)
    return;
  }
  
  // If there was a repeat count, add it to the message
  const logEntry: LogEntry = repeatCount 
    ? { level, message: `${message} (previous message repeated ${repeatCount}x)`, context }
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
