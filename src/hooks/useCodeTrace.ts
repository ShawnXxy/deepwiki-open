import { useState, useCallback } from 'react';
import type { CodeTraceResult } from '@/types/codetrace';

interface UseCodeTraceResult {
  data: CodeTraceResult | null;
  loading: boolean;
  error: string | null;
  generate: (question: string, repoUrl: string, repoType: string, branch?: string | null, token?: string | null) => Promise<void>;
}

export function useCodeTrace(): UseCodeTraceResult {
  const [data, setData] = useState<CodeTraceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = useCallback(async (
    question: string,
    repoUrl: string,
    repoType: string,
    branch?: string | null,
    token?: string | null,
  ) => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const body: Record<string, string> = {
        question,
        repo_url: repoUrl,
        type: repoType,
      };
      if (branch) body.branch = branch;
      if (token) body.token = token;

      const resp = await fetch('/api/codetrace', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.error || `Request failed (${resp.status})`);
      }

      const result = await resp.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate code trace');
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, generate };
}
