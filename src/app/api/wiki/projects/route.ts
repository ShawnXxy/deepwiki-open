import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';

/**
 * GET /api/wiki/projects — List processed projects.
 * DELETE /api/wiki/projects — Delete a cached wiki.
 *
 * In cloud (FASTAPI_PORT set): proxies to backend FastAPI which reads from blob.
 * In local: scans ~/.adalflow/wikicache/.
 */

const BACKEND_PORT = process.env.FASTAPI_PORT || process.env.PORT;

interface ProcessedProject {
  id: string;
  owner: string;
  repo: string;
  name: string;
  repo_type: string;
  submittedAt: number;
  language: string;
  comprehensive: boolean;
  branch?: string;
}

// Cache filename pattern: deepwiki_cache_{type}_{owner}_{repo}_{lang}_{mode}[_{branch}].json
// Owner may contain spaces; lang may contain hyphens (e.g. zh-tw) but NOT underscores; repo may contain hyphens.
const CACHE_PATTERN = /^deepwiki_cache_(\w+)_(.+?)_([^_]+)_([a-z]+(?:-[a-z]+)*)_(comprehensive|concise)(?:_(.+))?\.json$/;

function getCacheDir(): string {
  return path.join(os.homedir(), '.adalflow', 'wikicache');
}

export async function GET() {
  // Try backend API first (handles blob storage in cloud mode)
  if (BACKEND_PORT) {
    try {
      const backendUrl = `http://127.0.0.1:${BACKEND_PORT}/api/processed_projects`;
      const resp = await fetch(backendUrl, { cache: 'no-store' });
      if (resp.ok) {
        const data = await resp.json();
        return NextResponse.json(data);
      }
    } catch (err) {
      console.warn('[projects] Backend unavailable, falling back to local disk');
    }
  }

  // Fallback: scan local disk
  const cacheDir = getCacheDir();

  if (!fs.existsSync(cacheDir)) {
    return NextResponse.json([]);
  }

  try {
    const files = fs.readdirSync(cacheDir).filter(f => f.startsWith('deepwiki_cache_') && f.endsWith('.json'));
    const projects: ProcessedProject[] = [];

    for (const filename of files) {
      const match = filename.match(CACHE_PATTERN);
      if (!match) continue;

      const [, repoType, owner, repo, language, mode, branch] = match;

      // Get file modification time for sorting
      let mtime = 0;
      try {
        const stat = fs.statSync(path.join(cacheDir, filename));
        mtime = stat.mtimeMs;
      } catch { /* ignore */ }

      projects.push({
        id: filename.replace('.json', ''),
        owner,
        repo,
        name: `${owner}/${repo}`,
        repo_type: repoType,
        submittedAt: mtime,
        language,
        comprehensive: mode === 'comprehensive',
        branch: branch || undefined,
      });
    }

    // Sort by most recently modified
    projects.sort((a, b) => b.submittedAt - a.submittedAt);

    return NextResponse.json(projects);
  } catch (err) {
    console.error('Error scanning wiki cache directory:', err);
    return NextResponse.json(
      { error: 'Failed to list projects' },
      { status: 500 },
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const body = await request.json();
    const { owner, repo, repo_type, language, comprehensive, branch } = body;

    if (!owner || !repo || !repo_type || !language || comprehensive === undefined) {
      return NextResponse.json(
        { error: 'owner, repo, repo_type, language, and comprehensive are required' },
        { status: 400 },
      );
    }

    const cacheDir = getCacheDir();
    const mode = comprehensive ? 'comprehensive' : 'concise';
    const branchSuffix = branch || 'default';
    const filename = `deepwiki_cache_${repo_type}_${owner}_${repo}_${language}_${mode}_${branchSuffix}.json`;
    const filePath = path.join(cacheDir, filename);

    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      return NextResponse.json({ message: 'Project deleted successfully' });
    }

    // Try legacy filename (no branch)
    const legacyFilename = `deepwiki_cache_${repo_type}_${owner}_${repo}_${language}_${mode}.json`;
    const legacyPath = path.join(cacheDir, legacyFilename);
    if (fs.existsSync(legacyPath)) {
      fs.unlinkSync(legacyPath);
      return NextResponse.json({ message: 'Project deleted successfully' });
    }

    return NextResponse.json({ error: 'Cache file not found' }, { status: 404 });
  } catch (err) {
    console.error('Error deleting wiki cache:', err);
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: `Failed to delete: ${message}` }, { status: 500 });
  }
}