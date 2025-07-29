export interface RepoInfo {
    owner: string;
    repo: string;
    type: string;
    token: string | null;
    branch: string | null;
    localPath: string | null;
    repoUrl: string | null;
}

export default RepoInfo;