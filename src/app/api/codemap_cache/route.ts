import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';

/**
 * GET /api/codemap_cache — Read codemap graph data.
 *
 * In cloud (FASTAPI_PORT set): proxies to backend FastAPI which reads from blob.
 * In local: reads JSON files from ~/.adalflow/codemap/.
 *
 * Query params: owner, repo, repo_type, branch
 */

const BACKEND_PORT = process.env.FASTAPI_PORT;

function getCacheDir(): string {
  return path.join(os.homedir(), '.adalflow', 'codemap');
}

function getCacheFilename(
  owner: string,
  repo: string,
  repoType: string,
  branch?: string | null,
): string {
  const branchSuffix = branch || 'default';
  return `codemap_${repoType}_${owner}_${repo}_${branchSuffix}.json`;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
function normalizeKeys(data: any): any {
  /**
   * Convert snake_case keys from the Python backend to camelCase
   * expected by the frontend TypeScript interfaces.
   */
  if (Array.isArray(data)) return data.map(normalizeKeys);
  if (data && typeof data === 'object') {
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(data)) {
      const camel = k.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
      out[camel] = normalizeKeys(v);
    }
    return out;
  }
  return data;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const rawOwner = searchParams.get('owner');
  const rawRepo = searchParams.get('repo');
  const owner = rawOwner ? decodeURIComponent(rawOwner) : null;
  const repo = rawRepo ? decodeURIComponent(rawRepo) : null;
  const repoType = searchParams.get('repo_type') || 'azuredevops';
  const branch = searchParams.get('branch') || null;

  if (!owner || !repo) {
    return NextResponse.json(
      { error: 'owner and repo are required' },
      { status: 400 },
    );
  }

  // Try backend API first (handles blob storage in cloud mode)
  if (BACKEND_PORT) {
    try {
      const params = new URLSearchParams({
        owner, repo, repo_type: repoType,
      });
      if (branch) params.set('branch', branch);

      const backendUrl = `http://127.0.0.1:${BACKEND_PORT}/api/codemap?${params}`;
      const resp = await fetch(backendUrl, { cache: 'no-store' });
      if (resp.ok) {
        const data = await resp.json();
        return NextResponse.json(normalizeKeys(data));
      }
      if (resp.status !== 404) {
        console.error(`[codemap_cache] Backend error: ${resp.status}`);
      }
    } catch {
      console.warn('[codemap_cache] Backend unavailable, falling back to local disk');
    }
  }

  // Fallback: read from local disk
  const cacheDir = getCacheDir();
  const filename = getCacheFilename(owner, repo, repoType, branch);
  const filePath = path.join(cacheDir, filename);

  if (fs.existsSync(filePath)) {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const data = JSON.parse(content);
      return NextResponse.json(normalizeKeys(data));
    } catch (err) {
      console.error(`Error reading codemap cache ${filePath}:`, err);
      return NextResponse.json(
        { error: 'Failed to parse codemap cache file' },
        { status: 500 },
      );
    }
  }

  return NextResponse.json(
    { error: 'Codemap not found' },
    { status: 404 },
  );
}
