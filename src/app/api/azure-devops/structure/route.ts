import { NextRequest, NextResponse } from 'next/server';

interface AzureDevOpsStructureRequest {
  repo_url: string;
  token?: string;
}

interface AzureDevOpsTreeItem {
  path: string;
  isFolder: boolean;
  size?: number;
}

export async function POST(request: NextRequest) {
  try {
    const body: AzureDevOpsStructureRequest = await request.json();
    const { repo_url, token } = body;

    // Debug: Log what we received in the request body
    console.log('[AzureDevOps API] Request body received:', {
      repo_url,
      hasToken: !!token,
      tokenLength: token?.length || 0,
      tokenPreview: token ? `${token.substring(0, 5)}...` : 'none'
    });

    if (!repo_url) {
      return NextResponse.json(
        { error: 'Repository URL is required' },
        { status: 400 }
      );
    }

    // Parse Azure DevOps URL to extract organization and project
    let organization = '';
    let project = '';
    let repository = '';
    let apiBaseUrl = 'https://dev.azure.com';

    try {
      const url = new URL(repo_url);
      const pathParts = url.pathname.split('/').filter(Boolean).map(part => decodeURIComponent(part));
      const gitIndex = pathParts.indexOf('_git');
      
      console.log('[AzureDevOps API] Parsing URL:', { repo_url, hostname: url.hostname, pathParts, gitIndex });
      
      if (url.hostname === 'dev.azure.com') {
        // Format variations:
        // 1. dev.azure.com/{organization}/{project}/_git/{repository} (gitIndex = 2)
        // 2. dev.azure.com/{organization}/_git/{repository} (gitIndex = 1)
        apiBaseUrl = 'https://dev.azure.com';
        if (gitIndex >= 1 && pathParts.length > gitIndex + 1) {
          organization = pathParts[0];
          repository = pathParts[gitIndex + 1];
          // project is right before _git, or same as org if gitIndex is 1
          project = gitIndex >= 2 ? pathParts[gitIndex - 1] : pathParts[0];
          console.log('[AzureDevOps API] dev.azure.com parsed:', { organization, project, repository });
        }
      } else if (url.hostname.includes('visualstudio.com')) {
        // Format: {organization}.visualstudio.com/{project}/_git/{repository}
        // API base URL uses dev.azure.com but with the organization from the hostname
        organization = url.hostname.split('.')[0];
        apiBaseUrl = 'https://dev.azure.com';  // Modern API endpoint
        if (gitIndex >= 1 && pathParts.length > gitIndex + 1) {
          repository = pathParts[gitIndex + 1];
          project = pathParts[gitIndex - 1];
          console.log('[AzureDevOps API] visualstudio.com parsed:', { organization, project, repository, apiBaseUrl });
        }
      } else {
        throw new Error('Invalid Azure DevOps URL format');
      }
    } catch (parseError) {
      console.error('[AzureDevOps API] URL parsing error:', parseError);
      return NextResponse.json(
        { error: 'Invalid Azure DevOps URL format' },
        { status: 400 }
      );
    }

    if (!organization || !project || !repository) {
      return NextResponse.json(
        { error: 'Could not parse Azure DevOps URL components' },
        { status: 400 }
      );
    }

    // Prepare headers for Azure DevOps API
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (token) {
      // Azure DevOps uses Basic auth with PAT
      const auth = Buffer.from(`:${token}`).toString('base64');
      headers['Authorization'] = `Basic ${auth}`;
      console.log('[AzureDevOps API] Authorization header set (token provided)');
    } else {
      console.log('[AzureDevOps API] WARNING: No token provided - will likely fail for private repos');
    }

    let defaultBranch = 'main';
    let fileTreeData = '';
    let readmeContent = '';

    // URL-encode components for API calls (spaces, special chars)
    const encodedOrg = encodeURIComponent(organization);
    const encodedProject = encodeURIComponent(project);
    const encodedRepo = encodeURIComponent(repository);

    try {
      // Step 1: Get repository info to determine default branch
      const repoInfoUrl = `${apiBaseUrl}/${encodedOrg}/${encodedProject}/_apis/git/repositories/${encodedRepo}?api-version=6.0`;
      console.log('[AzureDevOps API] Fetching repo info from:', repoInfoUrl);
      
      const repoInfoResponse = await fetch(repoInfoUrl, { headers });
      
      if (repoInfoResponse.ok) {
        const contentType = repoInfoResponse.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          const repoInfo = await repoInfoResponse.json();
          defaultBranch = repoInfo.defaultBranch?.replace('refs/heads/', '') || 'main';
        } else {
          console.warn('Repository info response is not JSON, using default branch "main"');
        }
      } else {
        // Check if this is an authentication issue
        if (repoInfoResponse.status === 401 || repoInfoResponse.status === 203) {
          throw new Error('Authentication required. Please provide a valid Azure DevOps Personal Access Token (PAT).');
        }
        console.warn(`Could not fetch repository info (${repoInfoResponse.status}), using default branch "main"`);
      }

      // Step 2: Get the repository tree
      const treeUrl = `${apiBaseUrl}/${encodedOrg}/${encodedProject}/_apis/git/repositories/${encodedRepo}/items?recursionLevel=Full&api-version=6.0`;
      console.log('[AzureDevOps API] Fetching tree from:', treeUrl);
      
      const treeResponse = await fetch(treeUrl, { headers });

      if (!treeResponse.ok) {
        if (treeResponse.status === 401 || treeResponse.status === 203) {
          throw new Error('Authentication required. Please provide a valid Azure DevOps Personal Access Token (PAT).');
        } else if (treeResponse.status === 404) {
          throw new Error('Repository not found. Please check the repository URL and your access permissions.');
        } else {
          const errorText = await treeResponse.text().catch(() => 'Unknown error');
          throw new Error(`Azure DevOps API error (${treeResponse.status}): ${errorText.substring(0, 200)}`);
        }
      }

      // Verify response is JSON before parsing
      const treeContentType = treeResponse.headers.get('content-type') || '';
      if (!treeContentType.includes('application/json')) {
        const responseText = await treeResponse.text();
        // Check if it's a login page (HTML response)
        if (responseText.includes('<!DOCTYPE') || responseText.includes('<html')) {
          throw new Error('Authentication required. Azure DevOps returned a login page. Please provide a valid Personal Access Token (PAT).');
        }
        throw new Error(`Azure DevOps API returned unexpected content type: ${treeContentType}`);
      }

      const treeData = await treeResponse.json();
      
      if (treeData.value && Array.isArray(treeData.value)) {
        // Filter for files only (not folders) and create file tree string
        const files = treeData.value
          .filter((item: AzureDevOpsTreeItem) => !item.isFolder && item.path)
          .map((item: AzureDevOpsTreeItem) => item.path)
          .sort();
        
        fileTreeData = files.join('\n');
      }

      // Step 3: Try to fetch README.md content
      try {
        const readmeUrl = `${apiBaseUrl}/${encodedOrg}/${encodedProject}/_apis/git/repositories/${encodedRepo}/items?path=/README.md&api-version=6.0`;
        
        const readmeResponse = await fetch(readmeUrl, { headers });
        
        if (readmeResponse.ok) {
          readmeContent = await readmeResponse.text();
        } else {
          console.warn(`Could not fetch README.md, status: ${readmeResponse.status}`);
        }
      } catch (err) {
        console.warn('Could not fetch README.md, continuing with empty README', err);
      }

    } catch (error) {
      console.error('Error fetching Azure DevOps repository data:', error);
      throw error;
    }

    if (!fileTreeData) {
      return NextResponse.json(
        { error: 'No files found in repository. Repository might be empty or inaccessible.' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      file_tree: fileTreeData,
      readme: readmeContent,
      default_branch: defaultBranch,
      organization,
      project,
      repository
    });

  } catch (error) {
    console.error('Error in Azure DevOps structure endpoint:', error);
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
}
