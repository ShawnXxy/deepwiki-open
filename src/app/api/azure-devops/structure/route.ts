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

    try {
      const url = new URL(repo_url);
      
      if (url.hostname === 'dev.azure.com') {
        // Format: dev.azure.com/{organization}/{project}/_git/{repository}
        const pathParts = url.pathname.split('/').filter(Boolean);
        if (pathParts.length >= 4 && pathParts[2] === '_git') {
          organization = pathParts[0];
          project = pathParts[1];
          repository = pathParts[3];
        }
      } else if (url.hostname.includes('visualstudio.com')) {
        // Format: {organization}.visualstudio.com/{project}/_git/{repository}
        const pathParts = url.pathname.split('/').filter(Boolean);
        if (pathParts.length >= 3 && pathParts[1] === '_git') {
          organization = url.hostname.split('.')[0];
          project = pathParts[0];
          repository = pathParts[2];
        }
      } else {
        throw new Error('Invalid Azure DevOps URL format');
      }
    } catch {
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
    }

    let defaultBranch = 'main';
    let fileTreeData = '';
    let readmeContent = '';

    try {
      // Step 1: Get repository info to determine default branch
      const repoInfoUrl = `https://dev.azure.com/${organization}/${project}/_apis/git/repositories/${repository}?api-version=6.0`;
      
      const repoInfoResponse = await fetch(repoInfoUrl, { headers });
      
      if (repoInfoResponse.ok) {
        const repoInfo = await repoInfoResponse.json();
        defaultBranch = repoInfo.defaultBranch?.replace('refs/heads/', '') || 'main';
      } else {
        console.warn('Could not fetch repository info, using default branch "main"');
      }

      // Step 2: Get the repository tree
      const treeUrl = `https://dev.azure.com/${organization}/${project}/_apis/git/repositories/${repository}/items?recursionLevel=Full&api-version=6.0`;
      
      const treeResponse = await fetch(treeUrl, { headers });

      if (!treeResponse.ok) {
        if (treeResponse.status === 401) {
          throw new Error('Unauthorized access to Azure DevOps. Please check your Personal Access Token (PAT).');
        } else if (treeResponse.status === 404) {
          throw new Error('Repository not found. Please check the repository URL and your access permissions.');
        } else {
          const errorText = await treeResponse.text().catch(() => 'Unknown error');
          throw new Error(`Azure DevOps API error (${treeResponse.status}): ${errorText}`);
        }
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
        const readmeUrl = `https://dev.azure.com/${organization}/${project}/_apis/git/repositories/${repository}/items?path=/README.md&api-version=6.0`;
        
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
