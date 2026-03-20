import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';

/**
 * GET /api/wiki_cache — Read wiki cache directly from local disk.
 *
 * This replaces the FastAPI backend proxy. The frontend reads JSON cache
 * files from ~/.adalflow/wikicache/ without needing `backend.main` running.
 *
 * Query params: owner, repo, repo_type, language, comprehensive, branch
 */

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

  // Decode URL-encoded params (handles double-encoding like %2520 → %20 → space)
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
