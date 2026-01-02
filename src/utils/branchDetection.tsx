/**
 * Git Branch Detection Utility for DeepWiki
 * 
 * This utility provides various methods to detect the current Git branch
 * when the branch is not explicitly provided in URL parameters.
 */

import { RepoInfo } from '@/types/repoinfo';

// Set to true to enable debug logging for branch detection
const DEBUG_BRANCH_DETECTION = false;

/**
 * Attempts to detect the current Git branch from various sources
 * 
 * @param repoInfo - Repository information
 * @param fallbackBranch - Fallback branch if detection fails (default: null for legacy/default)
 * @returns The detected branch name or null if not detected
 */
export function detectCurrentBranch(repoInfo: RepoInfo, fallbackBranch: string | null = null): string | null {
  if (DEBUG_BRANCH_DETECTION) {
    console.log('🌿 Branch Detection - Starting:', { owner: repoInfo.owner, repo: repoInfo.repo, branch: repoInfo.branch });
  }

  // 1. If branch is explicitly set in repoInfo, use it
  if (repoInfo.branch) {
    return repoInfo.branch;
  }

  // 2. Try to detect from current URL path (for repository browsing)
  if (typeof window !== 'undefined') {
    const currentPath = window.location.pathname;
    
    // For URLs like /owner/repo/tree/branch-name or /owner/repo/blob/branch-name
    const branchFromPath = extractBranchFromPath(currentPath);
    if (branchFromPath && branchFromPath !== 'main') {
      if (DEBUG_BRANCH_DETECTION) console.log('✅ Branch from URL path:', branchFromPath);
      return branchFromPath;
    }
    
    // Also check URL search parameters for branch
    const urlParams = new URLSearchParams(window.location.search);
    const branchFromParams = urlParams.get('branch');
    if (branchFromParams) {
      if (DEBUG_BRANCH_DETECTION) console.log('✅ Branch from URL params:', branchFromParams);
      return branchFromParams;
    }
  }

  // 3. Try to detect from repository URL patterns
  const branchFromRepoUrl = extractBranchFromRepoUrl(repoInfo.repoUrl);
  if (branchFromRepoUrl) {
    if (DEBUG_BRANCH_DETECTION) console.log('✅ Branch from repo URL:', branchFromRepoUrl);
    return branchFromRepoUrl;
  }

  // 4. For local development, try to use known branch context
  const knownBranch = getKnownCurrentBranch(repoInfo);
  if (knownBranch) {
    if (DEBUG_BRANCH_DETECTION) console.log('✅ Known branch context:', knownBranch);
    return knownBranch;
  }

  // 5. Return fallback (null by default, meaning use "default" tag for legacy wikis)
  if (DEBUG_BRANCH_DETECTION && fallbackBranch) {
    console.log('⚠️ Branch Detection - Using fallback:', fallbackBranch);
  }
  return fallbackBranch;
}

/**
 * Extracts branch name from URL path patterns
 * 
 * @param path - Current URL path
 * @returns Detected branch name or null
 */
function extractBranchFromPath(path: string): string | null {
  // Match patterns like /owner/repo/tree/branch-name or /owner/repo/blob/branch-name
  const pathPatterns = [
    /\/[^\/]+\/[^\/]+\/(?:tree|blob)\/([^\/]+)/,  // GitHub/GitLab style
    /\/[^\/]+\/[^\/]+\/src\/([^\/]+)/,            // Bitbucket style
    /\/[^\/]+\/[^\/]+\/_git\/[^\/]+\/[^\/]*\?.*version=GB([^&]+)/  // Azure DevOps style
  ];

  for (const pattern of pathPatterns) {
    const match = path.match(pattern);
    if (match && match[1]) {
      return decodeURIComponent(match[1]);
    }
  }

  return null;
}

/**
 * Extracts branch information from repository URL
 * 
 * @param repoUrl - Repository URL
 * @returns Detected branch name or null
 */
function extractBranchFromRepoUrl(repoUrl: string | null): string | null {
  if (!repoUrl) return null;

  try {
    const url = new URL(repoUrl);
    
    // For Azure DevOps URLs with version parameter
    if (url.hostname === 'dev.azure.com' || url.hostname.includes('visualstudio.com')) {
      const versionParam = url.searchParams.get('version');
      if (versionParam && versionParam.startsWith('GB')) {
        return versionParam.substring(2); // Remove 'GB' prefix
      }
    }

    // For other repository URLs with branch in path
    const pathParts = url.pathname.split('/');
    const branchIndex = pathParts.findIndex(part => ['tree', 'blob', 'src'].includes(part));
    if (branchIndex !== -1 && pathParts[branchIndex + 1]) {
      return decodeURIComponent(pathParts[branchIndex + 1]);
    }
  } catch {
    // Silently ignore URL parsing errors - branch detection is best-effort
  }

  return null;
}

/**
 * Gets the known current branch for the repository
 * This is where we can implement repository-specific branch detection
 * 
 * @param repoInfo - Repository information
 * @returns Known branch name or null
 */
function getKnownCurrentBranch(repoInfo: RepoInfo): string | null {
  // Check multiple possible owner formats and variations
  const ownerLower = repoInfo.owner?.toLowerCase() || '';
  const repoLower = repoInfo.repo?.toLowerCase() || '';
  
  // Check for deepwiki-open repository with various owner formats
  if ((ownerLower === 'shawnxxy' || ownerLower === 'shawnx') && 
      (repoLower === 'deepwiki-open' || repoLower === 'deepwiki')) {
    return 'orcas';
  }
  
  // Also check if the repo URL contains deepwiki-open
  if (repoInfo.repoUrl && repoInfo.repoUrl.toLowerCase().includes('deepwiki-open')) {
    return 'orcas';
  }
  
  // Additional check: if we're in the deepwiki-open environment, assume orcas branch
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    const pathname = window.location.pathname;
    
    // Check if we're running in a deepwiki-open context
    if (pathname.includes('deepwiki') || hostname.includes('deepwiki')) {
      return 'orcas';
    }
  }

  // No known branch for this repository - will use "default" tag for legacy wikis
  return null;
}

/**
 * Enhanced version of the original citation processor that includes branch detection
 * 
 * @param content - Content to process
 * @param repoInfo - Repository information
 * @param explicitBranch - Explicitly provided branch (optional)
 * @returns Processed content with proper branch URLs
 */
export function processCitations(content: string, repoInfo: RepoInfo, explicitBranch?: string): string {
  // Use explicit branch if provided, otherwise detect current branch
  // detectCurrentBranch returns null for legacy wikis (will use "default" tag)
  const branchToUse = explicitBranch || detectCurrentBranch(repoInfo);
  
  if (DEBUG_BRANCH_DETECTION && branchToUse) {
    console.log('📝 Citation Processing - Using branch:', branchToUse);
  }
  
  // Import the original processCitations function
  // Note: This would need to be properly imported in the actual implementation
  // For now, this is just the structure
  
  return content; // Placeholder - would call actual processCitations with detected branch
}
