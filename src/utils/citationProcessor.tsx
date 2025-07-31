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
  // Debug logging to trace branch parameter flow
  console.log('🔍 Citation Processor - generateFileUrl called (branch-agnostic mode):', {
    filePath,
    repoType: repoInfo.type,
    repoOwner: repoInfo.owner,
    repoName: repoInfo.repo,
    repoBranch: repoInfo.branch,
    defaultBranchParam: defaultBranch,
    repoUrl: repoInfo.repoUrl
  });
  
  if (repoInfo.type === 'local') {
    console.log('📁 Citation Processor - Local repo, returning file path as-is');
    // For local repositories, we can't generate web URLs
    return filePath;
  }

  const repoUrl = repoInfo.repoUrl;
  if (!repoUrl) {
    console.log('⚠️ Citation Processor - No repo URL, returning file path as-is');
    return filePath;
  }

  try {
    const url = new URL(repoUrl);
    const hostname = url.hostname;
    
    console.log('🌐 Citation Processor - Processing URL for hostname (branch-agnostic):', hostname);
    
    if (hostname === 'github.com' || hostname.includes('github')) {
      // GitHub URL format (branch-agnostic): https://github.com/owner/repo/blob/HEAD/path
      // Using HEAD to refer to the default branch without specifying it explicitly
      const generatedUrl = `${repoUrl}/blob/HEAD/${filePath}`;
      console.log('📍 Citation Processor - Generated branch-agnostic GitHub URL:', generatedUrl);
      return generatedUrl;
    } else if (hostname === 'gitlab.com' || hostname.includes('gitlab')) {
      // GitLab URL format (branch-agnostic): https://gitlab.com/owner/repo/-/blob/HEAD/path
      // Using HEAD to refer to the default branch
      const generatedUrl = `${repoUrl}/-/blob/HEAD/${filePath}`;
      console.log('📍 Citation Processor - Generated branch-agnostic GitLab URL:', generatedUrl);
      return generatedUrl;
    } else if (hostname === 'bitbucket.org' || hostname.includes('bitbucket')) {
      // Bitbucket URL format (branch-agnostic): https://bitbucket.org/owner/repo/src/HEAD/path
      // Using HEAD to refer to the default branch
      const generatedUrl = `${repoUrl}/src/HEAD/${filePath}`;
      console.log('📍 Citation Processor - Generated branch-agnostic Bitbucket URL:', generatedUrl);
      return generatedUrl;
    } else if (hostname === 'dev.azure.com' || hostname.includes('visualstudio.com')) {
      // Azure DevOps URL format (branch-agnostic):
      // https://dev.azure.com/{organization}/{project}/_git/{repo}?path=/path-to-file
      // Removing the &version=GB{branch} parameter entirely to make it branch-agnostic
      
      // Ensure filePath starts with '/' for proper URL construction
      const encodedPath = encodeURIComponent(filePath.startsWith('/') ? filePath : `/${filePath}`);
      
      const azureUrl = `${repoUrl}?path=${encodedPath}`;
      console.log('📍 Citation Processor - Generated branch-agnostic Azure DevOps URL:', azureUrl);
      
      return azureUrl;
    }
  } catch (error) {
    console.warn('Error generating file URL:', error);
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
  console.log('📝 Citation Processor - processCitations called (branch-agnostic mode):', {
    contentLength: content.length,
    repoType: repoInfo.type,
    repoOwner: repoInfo.owner,
    repoName: repoInfo.repo,
    repoBranch: repoInfo.branch,
    defaultBranchParam: defaultBranch
  });
  
  if (!content) {
    console.log('⚠️ Citation Processor - No content to process');
    return content;
  }

  // Pattern to match citation links in the format: Sources: [filename.ext:line-range]() or [filename.ext:line]()
  // This handles both single files and multiple citations
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
      
      // Extract just the filename (before the colon if present)
      const filename = citationContent.includes(':') 
        ? citationContent.split(':')[0] 
        : citationContent;
      
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
