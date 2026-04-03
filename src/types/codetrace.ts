export interface CodeReference {
  refId: string;
  filePath: string;
  startLine: number;
  endLine: number;
  snippet: string;
  annotation: string;
}

export interface CodeTraceSection {
  id: string;
  title: string;
  motivation: string;
  details: string;
  codeRefs: CodeReference[];
  connections: string[];
}

export interface CodeTraceResult {
  query: string;
  title: string;
  sections: CodeTraceSection[];
  sourceFiles: string[];
  generatedAt?: string;
}
