import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';

/**
 * GET /api/wiki_cache — Read wiki cache.
 *
 * In cloud (FASTAPI_PORT set): proxies to backend FastAPI which reads from blob.
 * In local: reads JSON files from ~/.adalflow/wikicache/.
 *
 * Query params: owner, repo, repo_type, language, comprehensive, branch
 */

const BACKEND_PORT = process.env.FASTAPI_PORT || process.env.PORT;

function getCacheDir(): string {
  return path.join(os.homedir(), '.adalflow', 'wikicache');
}

function getCacheFilename(
  owner: string,
  repo: string,
  repoType: string,
  language: string,
  comprehensive: boolean,
  branch?: string | null,
): string {
  const mode = comprehensive ? 'comprehensive' : 'concise';
  const branchSuffix = branch || 'default';
  return `deepwiki_cache_${repoType}_${owner}_${repo}_${language}_${mode}_${branchSuffix}.json`;
}

function getLegacyFilename(
  owner: string,
  repo: string,
  repoType: string,
  language: string,
  comprehensive: boolean,
): string {
  const mode = comprehensive ? 'comprehensive' : 'concise';
  return `deepwiki_cache_${repoType}_${owner}_${repo}_${language}_${mode}.json`;
}

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const rawOwner = searchParams.get('owner');
  const rawRepo = searchParams.get('repo');
  const owner = rawOwner ? decodeURIComponent(rawOwner) : null;
  const repo = rawRepo ? decodeURIComponent(rawRepo) : null;
  const repoType = searchParams.get('repo_type') || 'azuredevops';
  const language = searchParams.get('language') || 'en';
  const comprehensive = searchParams.get('comprehensive') !== 'false';
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
        owner, repo, repo_type: repoType, language,
        comprehensive: String(comprehensive),
      });
      if (branch) params.set('branch', branch);

      const backendUrl = `http://127.0.0.1:${BACKEND_PORT}/api/wiki_cache?${params}`;
      const resp = await fetch(backendUrl, { cache: 'no-store' });
      if (resp.ok) {
        const data = await resp.json();
        return NextResponse.json(data);
      }
      // If backend returns 404, fall through to local disk
      if (resp.status !== 404) {
        console.error(`[wiki_cache] Backend error: ${resp.status}`);
      }
    } catch (err) {
      console.warn('[wiki_cache] Backend unavailable, falling back to local disk');
    }
  }

  // Fallback: read from local disk
  const cacheDir = getCacheDir();

  // Try new format (with branch suffix) first, then legacy
  const candidates = [
    getCacheFilename(owner, repo, repoType, language, comprehensive, branch),
    getLegacyFilename(owner, repo, repoType, language, comprehensive),
  ];

  console.log(`[wiki_cache] Looking for: ${candidates.join(' | ')} in ${cacheDir}`);

  for (const filename of candidates) {
    const filePath = path.join(cacheDir, filename);
    if (fs.existsSync(filePath)) {
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        const data = JSON.parse(content);
        return NextResponse.json(data);
      } catch (err) {
        console.error(`Error reading wiki cache ${filePath}:`, err);
        return NextResponse.json(
          { error: 'Failed to parse wiki cache file' },
          { status: 500 },
        );
      }
    }
  }

  return NextResponse.json(
    { error: 'Wiki cache not found' },
    { status: 404 },
  );
}
