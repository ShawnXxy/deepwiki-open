import { useState, useEffect, useCallback } from 'react';
import type { CodeMapData } from '@/types/codemap';

interface UseCodeMapResult {
  data: CodeMapData | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useCodeMap(
  owner: string | undefined,
  repo: string | undefined,
  repoType: string,
  branch: string | null,
): UseCodeMapResult {
  const [data, setData] = useState<CodeMapData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCodeMap = useCallback(async () => {
    if (!owner || !repo) return;

    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        owner,
        repo,
        repo_type: repoType,
      });
      if (branch) params.set('branch', branch);

      const resp = await fetch(`/api/codemap_cache?${params}`);
      if (resp.ok) {
        const json = await resp.json();
        setData(json);
      } else if (resp.status === 404) {
        setData(null);
        setError('Codemap not yet generated. Process the repository to create it.');
      } else {
        setError(`Failed to load codemap (${resp.status})`);
      }
    } catch {
      setError('Failed to fetch codemap data');
    } finally {
      setLoading(false);
    }
  }, [owner, repo, repoType, branch]);

  useEffect(() => {
    fetchCodeMap();
  }, [fetchCodeMap]);

  return { data, loading, error, refetch: fetchCodeMap };
}
