/**
 * Citation Processing Utility for DeepWiki
 * 
 * This utility handles the processing of AI-generated citations that come with empty URLs
 * and fills them with proper repository URLs. It supports all major repository types
 * including Azure DevOps, GitHub, GitLab, and Bitbucket.
 * 
 * The utility generates branch-agnostic URLs that automatically resolve to the default
 * branch of each repository, eliminating the need for explicit branch detection.
 * 
 * The utility is designed to work with citations in the format:
 * Sources: [filename.ext:line-range]() or [filename.ext:line]()
 */

import { RepoInfo } from '@/types/repoinfo';
import logger from '@/utils/logger';

/**
 * Generates a branch-agnostic repository file URL based on the repository type and file path
 * 
 * This function creates URLs that automatically resolve to the default branch of the repository:
 * - GitHub/GitLab/Bitbucket: Uses 'HEAD' to reference the default branch
 * - Azure DevOps: Omits the version parameter to use the default branch
 * 
 * @param filePath - The path to the file within the repository
 * @param repoInfo - Repository information including type, URL, owner, and repo name
 * @param defaultBranch - The default branch name (kept for compatibility but not used in branch-agnostic mode)
 * @returns The complete branch-agnostic URL to the file in the repository
 */
export function generateFileUrl(filePath: string, repoInfo: RepoInfo, defaultBranch: string = 'main'): string {
  // Debug logging (deduplicated by logger)
  logger.debug('Citation: generateFileUrl', {
    filePath,
    repoType: repoInfo.type,
    hasRepoUrl: !!repoInfo.repoUrl
  });
  
  if (repoInfo.type === 'local') {
    // For local repositories, we can't generate web URLs
    return filePath;
  }

  const repoUrl = repoInfo.repoUrl;
  if (!repoUrl) {
    return filePath;
  }

  try {
    const url = new URL(repoUrl);
    const hostname = url.hostname;
    
    if (hostname === 'github.com' || hostname.includes('github')) {
      // GitHub: pin to commit hash or branch, fallback to HEAD
      const ref = defaultBranch || 'HEAD';
      return `${repoUrl}/blob/${ref}/${filePath}`;
    } else if (hostname === 'gitlab.com' || hostname.includes('gitlab')) {
      const ref = defaultBranch || 'HEAD';
      return `${repoUrl}/-/blob/${ref}/${filePath}`;
    } else if (hostname === 'bitbucket.org' || hostname.includes('bitbucket')) {
      const ref = defaultBranch || 'HEAD';
      return `${repoUrl}/src/${ref}/${filePath}`;
    } else if (hostname === 'dev.azure.com' || hostname.includes('visualstudio.com')) {
      // Azure DevOps: pin to commit hash via version=GC{hash} or branch via version=GB{branch}
      const encodedPath = encodeURIComponent(filePath.startsWith('/') ? filePath : `/${filePath}`);
      if (defaultBranch && /^[0-9a-f]{7,40}$/.test(defaultBranch)) {
        // Looks like a commit hash — use GC (git commit) prefix
        return `${repoUrl}?path=${encodedPath}&version=GC${defaultBranch}`;
      } else if (defaultBranch) {
        // Branch name — use GB (git branch) prefix
        return `${repoUrl}?path=${encodedPath}&version=GB${defaultBranch}`;
      }
      return `${repoUrl}?path=${encodedPath}`;
    }
  } catch (error) {
    logger.warn('Citation: Error generating file URL', { error: String(error), filePath });
  }

  // Fallback to just the file path
  return filePath;
}

/**
 * Processes markdown content to replace empty citation URLs with proper branch-agnostic repository URLs
 * 
 * This function looks for citation patterns in the format:
 * Sources: [filename.ext:line-range]() or [filename.ext:line]()
 * 
 * And replaces the empty parentheses with branch-agnostic repository URLs that automatically
 * resolve to the default branch of each repository.
 * 
 * @param content - The markdown content containing citations
 * @param repoInfo - Repository information for URL generation
 * @param defaultBranch - The default branch name (kept for compatibility but not used in branch-agnostic mode)
 * @returns The processed content with proper branch-agnostic citation URLs
 */
export function processCitations(content: string, repoInfo: RepoInfo, defaultBranch: string = 'main'): string {
  logger.debug('Citation: processCitations', {
    contentLength: content.length,
    repoType: repoInfo.type
  });
  
  if (!content) {
    return content;
  }

  // Pattern to match citation links with empty URLs: Sources: [filename.ext:line-range]()
  // Skip citations that already have URLs (non-empty parens)
  const citationPattern = /Sources:\s*(?:\[([^\]]+)\]\(\)(?:,\s*)?)+/g;
  const singleCitationPattern = /\[([^\]]+)\]\(\)/g;

  return content.replace(citationPattern, (fullMatch) => {
    // Extract all individual citations from the Sources line
    const citations: string[] = [];
    let citationMatch;
    
    // Reset the regex lastIndex to ensure we start from the beginning of the match
    singleCitationPattern.lastIndex = 0;
    
    while ((citationMatch = singleCitationPattern.exec(fullMatch)) !== null) {
      const citationContent = citationMatch[1];
      
      // Extract just the filename (before colon or space+L for line refs)
      // Handles: "file.py", "file.py:45", "file.py L45-L120"
      let filename = citationContent;
      const colonIdx = citationContent.indexOf(':');
      const lineRefIdx = citationContent.indexOf(' L');
      if (colonIdx > 0) {
        filename = citationContent.substring(0, colonIdx);
      } else if (lineRefIdx > 0) {
        filename = citationContent.substring(0, lineRefIdx);
      }
      
      // Generate the proper URL for this file
      const fileUrl = generateFileUrl(filename, repoInfo, defaultBranch);
      
      // Create the citation link with the proper URL
      citations.push(`[${citationContent}](${fileUrl})`);
    }
    
    // Return the processed Sources line with all proper URLs
    return `Sources: ${citations.join(', ')}`;
  });
}

/**
 * Processes Mermaid diagram content to replace citation comments with proper URLs
 * 
 * This is specifically for Mermaid diagrams which convert citations to comments
 * for syntax compatibility.
 * 
 * @param content - The Mermaid diagram content
 * @param repoInfo - Repository information for URL generation
 * @param defaultBranch - The default branch name (default: 'main')
 * @returns The processed Mermaid content with proper citation comments
 */
export function processMermaidCitations(content: string, repoInfo: RepoInfo, defaultBranch: string = 'main'): string {
  if (!content) return content;

  // Pattern to match Mermaid comment citations: %% Source: filename
  const mermaidCitationPattern = /%% Source: ([^\n\r]+)/g;

  return content.replace(mermaidCitationPattern, (match, filename) => {
    // Generate the proper URL for this file
    const fileUrl = generateFileUrl(filename.trim(), repoInfo, defaultBranch);
    
    // Return the comment with the proper URL
    return `%% Source: ${filename.trim()} - ${fileUrl}`;
  });
}
