'use client';

import React, {useState, useRef, useEffect} from 'react';
import {FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import Markdown from './Markdown';
import { useLanguage } from '@/contexts/LanguageContext';
import RepoInfo from '@/types/repoinfo';
import getRepoUrl from '@/utils/getRepoUrl';
import ModelSelectionModal from './ModelSelectionModal';
import { createChatWebSocket, closeWebSocket, ChatCompletionRequest } from '@/utils/websocketClient';
import { processCitations } from '@/utils/citationProcessor';
import { detectCurrentBranch } from '@/utils/branchDetection';

interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  models: Model[];
  supportsCustomModel?: boolean;
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  isDeepResearch?: boolean;
  researchData?: {
    iterations: ResearchIteration[];
    finalConclusion?: string;
    isThinkingExpanded?: boolean;
  };
}

interface ResearchIteration {
  id: string;
  iteration: number;
  title: string;
  content: string;
  type: 'plan' | 'update' | 'conclusion';
  isComplete: boolean;
}

interface ResearchStage {
  title: string;
  content: string;
  iteration: number;
  type: 'plan' | 'update' | 'conclusion';
}

interface AskProps {
  repoInfo: RepoInfo;
  provider?: string;
  model?: string;
  isCustomModel?: boolean;
  customModel?: string;
  language?: string;
  isVisible?: boolean;
  onRef?: (ref: { clearConversation: () => void }) => void;
}

const Ask: React.FC<AskProps> = ({
  repoInfo,
  provider = '',
  model = '',
  isCustomModel = false,
  customModel = '',
  language = 'en',
  isVisible = true,
  onRef
}) => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  const [currentIterationIndex, setCurrentIterationIndex] = useState<Record<string, number>>({});

  // Model selection state
  const [selectedProvider, setSelectedProvider] = useState(provider);
  const [selectedModel, setSelectedModel] = useState(model);
  const [isCustomSelectedModel, setIsCustomSelectedModel] = useState(isCustomModel);
  const [customSelectedModel, setCustomSelectedModel] = useState(customModel);
  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);
  const [isComprehensiveView, setIsComprehensiveView] = useState(true);

  // Get language context for translations
  const { messages } = useLanguage();

  // Research navigation state
  const [researchStages, setResearchStages] = useState<ResearchStage[]>([]);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [researchIteration, setResearchIteration] = useState(0);
  const [researchComplete, setResearchComplete] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const responseRef = useRef<HTMLDivElement>(null);
  const providerRef = useRef(provider);
  const modelRef = useRef(model);
  
  // Track the current assistant message ID for deep research updates
  const currentAssistantMessageIdRef = useRef<string | null>(null);
  // Accumulate all iteration content across continueResearch calls
  const allIterationsContentRef = useRef<string[]>([]);

  // OpenAI connection status
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'checking' | 'connected' | 'error'>('idle');
  const [connectionError, setConnectionError] = useState('');
  const [connectionModel, setConnectionModel] = useState('');

  // Check OpenAI connectivity when chat panel becomes visible
  useEffect(() => {
    if (!isVisible) return;
    let cancelled = false;

    const checkConnection = async () => {
      setConnectionStatus('checking');
      setConnectionError('');
      try {
        const res = await fetch('/api/health/openai', { signal: AbortSignal.timeout(15000) });
        if (cancelled) return;
        const data = await res.json();
        if (data.status === 'connected') {
          setConnectionStatus('connected');
          setConnectionModel(data.model || '');
        } else {
          setConnectionStatus('error');
          setConnectionError(data.message || 'Connection failed');
        }
      } catch (err) {
        if (cancelled) return;
        setConnectionStatus('error');
        setConnectionError(err instanceof Error ? err.message : 'Backend unreachable');
      }
    };

    checkConnection();
    return () => { cancelled = true; };
  }, [isVisible]);

  // Focus input on component mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // Expose clearConversation method to parent component
  useEffect(() => {
    if (onRef) {
      onRef({ clearConversation });
    }
  }, [onRef]);

  // Scroll to bottom of response when it changes
  useEffect(() => {
    if (responseRef.current) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [response]);

  // Close WebSocket when component unmounts
  useEffect(() => {
    return () => {
      closeWebSocket(webSocketRef.current);
    };
  }, []);

  useEffect(() => {
    providerRef.current = provider;
    modelRef.current = model;
  }, [provider, model]);

  useEffect(() => {
    const fetchModel = async () => {
      try {
        setIsLoading(true);

        const response = await fetch('/api/models/config');
        if (!response.ok) {
          throw new Error(`Error fetching model configurations: ${response.status}`);
        }

        const data = await response.json();

        // Get the provider to use (current or default)
        const activeProvider = providerRef.current || data.defaultProvider;
        const selectedProviderConfig = data.providers.find((p: Provider) => p.id === activeProvider);
        
        // Set provider
        if (providerRef.current === '') {
          setSelectedProvider(data.defaultProvider);
        } else {
          setSelectedProvider(providerRef.current);
        }

        // Validate and set model - ensure it matches available models from API
        // This handles stale localStorage values (e.g., old "gpt-4.1" cache)
        if (selectedProviderConfig && selectedProviderConfig.models.length > 0) {
          const validModelIds = selectedProviderConfig.models.map((m: Model) => m.id);
          if (modelRef.current === '' || !validModelIds.includes(modelRef.current)) {
            // Model is empty or invalid - set to first available model
            setSelectedModel(selectedProviderConfig.models[0].id);
          } else {
            setSelectedModel(modelRef.current);
          }
        }
      } catch (err) {
        console.error('Failed to fetch model configurations:', err);
      } finally {
        setIsLoading(false);
      }
    };
    if(provider == '' || model == '') {
      fetchModel()
    }
  }, [provider, model]);

  const clearConversation = () => {
    setQuestion('');
    setResponse('');
    setConversationHistory([]);
    setChatMessages([]);
    setExpandedThinking({});
    setCurrentIterationIndex({});
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setCurrentStageIndex(0);
    currentAssistantMessageIdRef.current = null;
    allIterationsContentRef.current = [];
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  // Toggle thinking section expansion for a message
  const toggleThinking = (messageId: string) => {
    setExpandedThinking(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  };

  // Navigate to a specific iteration in a message's thinking section
  const navigateIteration = (messageId: string, direction: 'prev' | 'next', totalIterations: number) => {
    setCurrentIterationIndex(prev => {
      const current = prev[messageId] || 0;
      if (direction === 'prev' && current > 0) {
        return { ...prev, [messageId]: current - 1 };
      }
      if (direction === 'next' && current < totalIterations - 1) {
        return { ...prev, [messageId]: current + 1 };
      }
      return prev;
    });
  };

  // Extract final conclusion from content
  const extractFinalConclusion = (content: string): string | null => {
    const conclusionMatch = content.match(/## Final Conclusion([\s\S]*?)$/);
    if (conclusionMatch) {
      return '## Final Conclusion' + conclusionMatch[1];
    }
    // Also check for regular conclusion
    const regularConclusionMatch = content.match(/## Conclusion([\s\S]*?)$/);
    if (regularConclusionMatch && !content.includes('Next Steps')) {
      return '## Conclusion' + regularConclusionMatch[1];
    }
    return null;
  };

  // Parse research content into iterations from accumulated content array
  const parseResearchIterationsFromArray = (iterationContents: string[]): ResearchIteration[] => {
    const iterations: ResearchIteration[] = [];
    
    iterationContents.forEach((content, idx) => {
      let type: 'plan' | 'update' | 'conclusion' = 'update';
      let title = `Research Update ${idx + 1}`;
      
      if (content.includes('## Research Plan')) {
        type = 'plan';
        title = 'Research Plan';
      } else if (content.includes('## Final Conclusion')) {
        type = 'conclusion';
        title = 'Final Conclusion';
      } else {
        // Extract iteration number from "## Research Update X"
        const updateMatch = content.match(/## Research Update (\d+)/);
        if (updateMatch) {
          title = `Research Update ${updateMatch[1]}`;
        }
      }
      
      iterations.push({
        id: `iteration-${idx}`,
        iteration: idx,
        title,
        content,
        type,
        isComplete: true
      });
    });
    
    return iterations;
  };

  // Legacy parse function for backward compatibility
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _parseResearchIterations = (content: string, iteration: number): ResearchIteration[] => {
    const iterations: ResearchIteration[] = [];
    
    // Check for research plan
    if (content.includes('## Research Plan')) {
      iterations.push({
        id: `plan-${Date.now()}`,
        iteration: 0,
        title: 'Research Plan',
        content: content,
        type: 'plan',
        isComplete: true
      });
    }
    
    // Check for research updates
    for (let i = 1; i <= iteration; i++) {
      if (content.includes(`## Research Update ${i}`) || (i === 1 && content.includes('## Research Update'))) {
        iterations.push({
          id: `update-${i}-${Date.now()}`,
          iteration: i,
          title: `Research Update ${i}`,
          content: content,
          type: 'update',
          isComplete: true
        });
      }
    }
    
    return iterations;
  };
  const downloadresponse = () =>{
  const blob = new Blob([response], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `response-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

  // Function to check if research is complete based on response content
  const checkIfResearchComplete = (content: string): boolean => {
    // Check for explicit final conclusion markers
    if (content.includes('## Final Conclusion')) {
      return true;
    }

    // Check for conclusion sections that don't indicate further research
    if ((content.includes('## Conclusion') || content.includes('## Summary')) &&
      !content.includes('I will now proceed to') &&
      !content.includes('Next Steps') &&
      !content.includes('next iteration')) {
      return true;
    }

    // Check for phrases that explicitly indicate completion
    if (content.includes('This concludes our research') ||
      content.includes('This completes our investigation') ||
      content.includes('This concludes the deep research process') ||
      content.includes('Key Findings and Implementation Details') ||
      content.includes('In conclusion,') ||
      (content.includes('Final') && content.includes('Conclusion'))) {
      return true;
    }

    // Check for topic-specific completion indicators
    if (content.includes('Dockerfile') &&
      (content.includes('This Dockerfile') || content.includes('The Dockerfile')) &&
      !content.includes('Next Steps') &&
      !content.includes('In the next iteration')) {
      return true;
    }

    return false;
  };

  // Function to extract research stages from the response
  const extractResearchStage = (content: string, iteration: number): ResearchStage | null => {
    // Check for research plan (first iteration)
    if (iteration === 1 && content.includes('## Research Plan')) {
      const planMatch = content.match(/## Research Plan([\s\S]*?)(?:## Next Steps|$)/);
      if (planMatch) {
        return {
          title: 'Research Plan',
          content: content,
          iteration: 1,
          type: 'plan'
        };
      }
    }

    // Check for research updates (iterations 1-4)
    if (iteration >= 1 && iteration <= 4) {
      const updateMatch = content.match(new RegExp(`## Research Update ${iteration}([\\s\\S]*?)(?:## Next Steps|$)`));
      if (updateMatch) {
        return {
          title: `Research Update ${iteration}`,
          content: content,
          iteration: iteration,
          type: 'update'
        };
      }
    }

    // Check for final conclusion
    if (content.includes('## Final Conclusion')) {
      const conclusionMatch = content.match(/## Final Conclusion([\s\S]*?)$/);
      if (conclusionMatch) {
        return {
          title: 'Final Conclusion',
          content: content,
          iteration: iteration,
          type: 'conclusion'
        };
      }
    }

    return null;
  };

  // Function to navigate to a specific research stage
  const navigateToStage = (index: number) => {
    if (index >= 0 && index < researchStages.length) {
      setCurrentStageIndex(index);
      setResponse(researchStages[index].content);
    }
  };

  // Function to navigate to the next research stage
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _navigateToNextStage = () => {
    if (currentStageIndex < researchStages.length - 1) {
      navigateToStage(currentStageIndex + 1);
    }
  };

  // Function to navigate to the previous research stage
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _navigateToPreviousStage = () => {
    if (currentStageIndex > 0) {
      navigateToStage(currentStageIndex - 1);
    }
  };

  // WebSocket reference
  const webSocketRef = useRef<WebSocket | null>(null);

  // Function to continue research automatically
  const continueResearch = async () => {
    if (!deepResearch || researchComplete || !response || isLoading) return;

    // Add a small delay to allow the user to read the current response
    await new Promise(resolve => setTimeout(resolve, 2000));

    setIsLoading(true);
    
    // Get the current assistant message ID
    const assistantMessageId = currentAssistantMessageIdRef.current;

    try {
      // Store the current response for use in the history
      const currentResponse = response;

      // Create a new message from the AI's previous response
      const newHistory: Message[] = [
        ...conversationHistory,
        {
          role: 'assistant',
          content: currentResponse
        },
        {
          role: 'user',
          content: '[DEEP RESEARCH] Continue the research'
        }
      ];

      // Update conversation history
      setConversationHistory(newHistory);

      // Increment research iteration
      const newIteration = researchIteration + 1;
      setResearchIteration(newIteration);

      // Clear previous response
      setResponse('');

      // Prepare the request body
      const requestBody: ChatCompletionRequest = {
        repo_url: getRepoUrl(repoInfo),
        type: repoInfo.type,
        messages: newHistory.map(msg => ({ role: msg.role as 'user' | 'assistant', content: msg.content })),
        provider: selectedProvider,
        model: isCustomSelectedModel ? customSelectedModel : selectedModel,
        language: language
      };

      // Add tokens if available
      if (repoInfo?.token) {
        requestBody.token = repoInfo.token;
      }

      // Add branch if available
      if (repoInfo?.branch) {
        requestBody.branch = repoInfo.branch;
      }

      // Close any existing WebSocket connection
      closeWebSocket(webSocketRef.current);

      let fullResponse = '';

      // Create a new WebSocket connection (returns null in cloud environments)
      const ws = createChatWebSocket(
        requestBody,
        // Message handler
        (message: string) => {
          fullResponse += message;
          setResponse(fullResponse);

          // Update chat message with accumulated iterations
          if (deepResearch && assistantMessageId) {
            // Build iterations from all accumulated content plus current streaming
            const iterationsContent = [...allIterationsContentRef.current, fullResponse];
            const iterations = parseResearchIterationsFromArray(iterationsContent);
            const conclusion = extractFinalConclusion(fullResponse);
            
            console.log('[Deep Research] continueResearch streaming - accumulated:', allIterationsContentRef.current.length, 'current iteration:', newIteration, 'parsed iterations:', iterations.length);
            
            setChatMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { 
                    ...msg, 
                    content: fullResponse,
                    researchData: {
                      iterations,
                      finalConclusion: conclusion || undefined,
                      isThinkingExpanded: true
                    }
                  }
                : msg
            ));
          }

          // Extract research stage if this is a deep research response
          if (deepResearch) {
            const stage = extractResearchStage(fullResponse, newIteration);
            if (stage) {
              // Add the stage to the research stages if it's not already there
              setResearchStages(prev => {
                // Check if we already have this stage
                const existingStageIndex = prev.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
                if (existingStageIndex >= 0) {
                  // Update existing stage
                  const newStages = [...prev];
                  newStages[existingStageIndex] = stage;
                  return newStages;
                } else {
                  // Add new stage
                  return [...prev, stage];
                }
              });

              // Update current stage index to the latest stage
              setCurrentStageIndex(researchStages.length);
            }
          }
        },
        // Error handler
        (error: Event) => {
          console.error('WebSocket error:', error);
          // Fallback to HTTP if WebSocket fails or is unavailable
          fallbackToHttp(requestBody, assistantMessageId || undefined);
        },
        // Close handler
        () => {
          // Store this iteration's content
          allIterationsContentRef.current = [...allIterationsContentRef.current, fullResponse];
          
          // Check if research is complete when the WebSocket closes
          const isComplete = checkIfResearchComplete(fullResponse);

          // Force completion after a maximum number of iterations (5)
          const forceComplete = newIteration >= 5;

          if (forceComplete && !isComplete) {
            // If we're forcing completion, append a comprehensive conclusion to the response
            const completionNote = "\n\n## Final Conclusion\nAfter multiple iterations of deep research, we've gathered significant insights about this topic. This concludes our investigation process, having reached the maximum number of research iterations. The findings presented across all iterations collectively form our comprehensive answer to the original question.";
            fullResponse += completionNote;
            setResponse(fullResponse);
            
            // Update the stored content with the completion note
            allIterationsContentRef.current[allIterationsContentRef.current.length - 1] = fullResponse;
            setResearchComplete(true);
          } else {
            setResearchComplete(isComplete);
          }
          
          // Final update to chat message
          if (assistantMessageId) {
            const iterationsContent = allIterationsContentRef.current;
            const iterations = parseResearchIterationsFromArray(iterationsContent);
            const conclusion = extractFinalConclusion(fullResponse);
            
            setChatMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { 
                    ...msg, 
                    isStreaming: !isComplete && !forceComplete,
                    content: fullResponse,
                    researchData: {
                      iterations,
                      finalConclusion: conclusion || undefined,
                      isThinkingExpanded: !conclusion // Collapse if we have conclusion
                    }
                  }
                : msg
            ));
            
            // Collapse thinking when research is complete
            if (isComplete || forceComplete) {
              setExpandedThinking(prev => ({ ...prev, [assistantMessageId]: false }));
            }
          }

          setIsLoading(false);
        }
      );
      
      // Store reference (may be null in cloud environments, error handler will trigger HTTP fallback)
      webSocketRef.current = ws;
    } catch (error) {
      console.error('Error during API call:', error);
      setResponse(prev => prev + '\n\nError: Failed to continue research. Please try again.');
      setResearchComplete(true);
      setIsLoading(false);
    }
  };

  // Fallback to HTTP if WebSocket fails
  const fallbackToHttp = async (requestBody: ChatCompletionRequest, assistantMessageId?: string) => {
    try {
      // Make the API call using HTTP
      const apiResponse = await fetch(`/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!apiResponse.ok) {
        throw new Error(`API error: ${apiResponse.status}`);
      }

      // Process the streaming response
      const reader = apiResponse.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Failed to get response reader');
      }

      // Read the stream
      let fullResponse = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        fullResponse += chunk;
        setResponse(fullResponse);
        
        // Update chat message if we have an ID
        if (assistantMessageId) {
          setChatMessages(prev => prev.map(msg => 
            msg.id === assistantMessageId 
              ? { ...msg, content: fullResponse }
              : msg
          ));
        }

        // Extract research stage if this is a deep research response
        if (deepResearch) {
          const stage = extractResearchStage(fullResponse, researchIteration);
          if (stage) {
            // Add the stage to the research stages
            setResearchStages(prev => {
              const existingStageIndex = prev.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
              if (existingStageIndex >= 0) {
                const newStages = [...prev];
                newStages[existingStageIndex] = stage;
                return newStages;
              } else {
                return [...prev, stage];
              }
            });
          }
        }
      }

      // Mark message as done streaming
      if (assistantMessageId) {
        setChatMessages(prev => prev.map(msg => 
          msg.id === assistantMessageId 
            ? { ...msg, isStreaming: false }
            : msg
        ));
      }

      // Check if research is complete
      const isComplete = checkIfResearchComplete(fullResponse);

      // Force completion after a maximum number of iterations (5)
      const forceComplete = researchIteration >= 5;

      if (forceComplete && !isComplete) {
        // If we're forcing completion, append a comprehensive conclusion to the response
        const completionNote = "\n\n## Final Conclusion\nAfter multiple iterations of deep research, we've gathered significant insights about this topic. This concludes our investigation process, having reached the maximum number of research iterations. The findings presented across all iterations collectively form our comprehensive answer to the original question.";
        fullResponse += completionNote;
        setResponse(fullResponse);
        if (assistantMessageId) {
          setChatMessages(prev => prev.map(msg => 
            msg.id === assistantMessageId 
              ? { ...msg, content: fullResponse }
              : msg
          ));
        }
        setResearchComplete(true);
      } else {
        setResearchComplete(isComplete);
      }
    } catch (error) {
      console.error('Error during HTTP fallback:', error);
      setResponse(prev => prev + '\n\nError: Failed to get a response. Please try again.');
      if (assistantMessageId) {
        setChatMessages(prev => prev.map(msg => 
          msg.id === assistantMessageId 
            ? { ...msg, content: msg.content + '\n\nError: Failed to get a response. Please try again.', isStreaming: false }
            : msg
        ));
      }
      setResearchComplete(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Effect to continue research when response is updated
  useEffect(() => {
    if (deepResearch && response && !isLoading && !researchComplete) {
      const isComplete = checkIfResearchComplete(response);
      if (isComplete) {
        setResearchComplete(true);
      } else if (researchIteration > 0 && researchIteration < 5) {
        // Only auto-continue if we're already in a research process and haven't reached max iterations
        // Use setTimeout to avoid potential infinite loops
        const timer = setTimeout(() => {
          continueResearch();
        }, 1000);
        return () => clearTimeout(timer);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [response, isLoading, deepResearch, researchComplete, researchIteration]);

  // Effect to update research stages when the response changes
  useEffect(() => {
    if (deepResearch && response && !isLoading) {
      // Try to extract a research stage from the response
      const stage = extractResearchStage(response, researchIteration);
      if (stage) {
        // Add or update the stage in the research stages
        setResearchStages(prev => {
          // Check if we already have this stage
          const existingStageIndex = prev.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
          if (existingStageIndex >= 0) {
            // Update existing stage
            const newStages = [...prev];
            newStages[existingStageIndex] = stage;
            return newStages;
          } else {
            // Add new stage
            return [...prev, stage];
          }
        });

        // Update current stage index to point to this stage
        setCurrentStageIndex(prev => {
          const newIndex = researchStages.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
          return newIndex >= 0 ? newIndex : prev;
        });
      }
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [response, isLoading, deepResearch, researchIteration]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!question.trim() || isLoading) return;

    handleConfirmAsk();
  };

  // Handle confirm and send request
  const handleConfirmAsk = async () => {
    const currentQuestion = question;
    setQuestion(''); // Clear input immediately
    setIsLoading(true);
    setResponse('');
    setResearchIteration(0);
    setResearchComplete(false);
    
    // Reset iteration tracking for new deep research
    allIterationsContentRef.current = [];

    // Add user message to chat
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: deepResearch ? `[DEEP RESEARCH] ${currentQuestion}` : currentQuestion,
      timestamp: new Date(),
      isDeepResearch: deepResearch
    };
    
    // Add assistant placeholder message for streaming
    const assistantMessageId = `assistant-${Date.now()}`;
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      isDeepResearch: deepResearch,
      researchData: deepResearch ? {
        iterations: [],
        finalConclusion: undefined,
        isThinkingExpanded: true
      } : undefined
    };
    
    setChatMessages(prev => [...prev, userMessage, assistantMessage]);
    
    // Store the assistant message ID for deep research updates
    currentAssistantMessageIdRef.current = assistantMessageId;
    
    // Initialize thinking as expanded for new deep research
    if (deepResearch) {
      setExpandedThinking(prev => ({ ...prev, [assistantMessageId]: true }));
    }

    try {
      // Create initial message
      const initialMessage: Message = {
        role: 'user',
        content: deepResearch ? `[DEEP RESEARCH] ${currentQuestion}` : currentQuestion
      };

      // Set initial conversation history
      const newHistory: Message[] = [initialMessage];
      setConversationHistory(newHistory);

      // Prepare request body
      const requestBody: ChatCompletionRequest = {
        repo_url: getRepoUrl(repoInfo),
        type: repoInfo.type,
        messages: newHistory.map(msg => ({ role: msg.role as 'user' | 'assistant', content: msg.content })),
        provider: selectedProvider,
        model: isCustomSelectedModel ? customSelectedModel : selectedModel,
        language: language
      };

      // Add tokens if available
      if (repoInfo?.token) {
        requestBody.token = repoInfo.token;
      }

      // Add branch if available
      if (repoInfo?.branch) {
        requestBody.branch = repoInfo.branch;
      }

      // Close any existing WebSocket connection
      closeWebSocket(webSocketRef.current);

      let fullResponse = '';
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const _accumulatedIterations: ResearchIteration[] = [];

      // Create a new WebSocket connection (returns null in cloud environments)
      const ws = createChatWebSocket(
        requestBody,
        // Message handler
        (message: string) => {
          fullResponse += message;
          setResponse(fullResponse);
          
          // For deep research, track iterations and final conclusion
          if (deepResearch) {
            // Build iterations from current streaming content
            const iterationsContent = [fullResponse];
            const iterations = parseResearchIterationsFromArray(iterationsContent);
            const conclusion = extractFinalConclusion(fullResponse);
            // accumulatedIterations tracking removed (handled by setChatMessages)
            
            setChatMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { 
                    ...msg, 
                    content: fullResponse,
                    researchData: {
                      iterations: iterations,
                      finalConclusion: conclusion || undefined,
                      isThinkingExpanded: true
                    }
                  }
                : msg
            ));
          } else {
            // Update the assistant message with streaming content
            setChatMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { ...msg, content: fullResponse }
                : msg
            ));
          }

          // Extract research stage if this is a deep research response
          if (deepResearch) {
            const stage = extractResearchStage(fullResponse, 1); // First iteration
            if (stage) {
              // Add the stage to the research stages
              setResearchStages([stage]);
              setCurrentStageIndex(0);
            }
          }
        },
        // Error handler
        (error: Event) => {
          console.error('WebSocket error:', error);
          // Fallback to HTTP if WebSocket fails or is unavailable
          fallbackToHttp(requestBody, assistantMessageId);
        },
        // Close handler
        () => {
          // For deep research, store this iteration's content and finalize the message
          if (deepResearch) {
            // Store first iteration content
            allIterationsContentRef.current = [fullResponse];
            
            const iterations = parseResearchIterationsFromArray([fullResponse]);
            const conclusion = extractFinalConclusion(fullResponse);
            const isComplete = checkIfResearchComplete(fullResponse);
            
            console.log('[Deep Research] handleConfirmAsk close - first iteration stored, isComplete:', isComplete, 'iterations:', iterations.length);
            
            setChatMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { 
                    ...msg, 
                    isStreaming: !isComplete,
                    content: fullResponse,
                    researchData: {
                      iterations,
                      finalConclusion: conclusion || undefined,
                      isThinkingExpanded: !conclusion // Collapse if we have conclusion
                    }
                  }
                : msg
            ));
            
            // Collapse thinking when research is complete
            if (conclusion) {
              setExpandedThinking(prev => ({ ...prev, [assistantMessageId]: false }));
            }
            
            setResearchComplete(isComplete);

            // If not complete, start the research process
            if (!isComplete) {
              setResearchIteration(1);
              // The continueResearch function will be triggered by the useEffect
            }
          } else {
            // Mark message as done streaming
            setChatMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { ...msg, isStreaming: false }
                : msg
            ));
          }

          setIsLoading(false);
        }
      );
      
      // Store reference (may be null in cloud environments, error handler will trigger HTTP fallback)
      webSocketRef.current = ws;
    } catch (error) {
      console.error('Error during API call:', error);
      setResponse(prev => prev + '\n\nError: Failed to get a response. Please try again.');
      setChatMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId 
          ? { ...msg, content: msg.content + '\n\nError: Failed to get a response. Please try again.', isStreaming: false }
          : msg
      ));
      setResearchComplete(true);
      setIsLoading(false);
    }
  };

  const [buttonWidth, setButtonWidth] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Measure button width and update state
  useEffect(() => {
    if (buttonRef.current) {
      const width = buttonRef.current.offsetWidth;
      setButtonWidth(width);
    }
  }, [messages.ask?.askButton, isLoading]);

  return (
    <div className="flex flex-col h-full">
      <div ref={responseRef} className="flex-1 overflow-y-auto p-4">
        {/* Chat messages */}
        {chatMessages.length > 0 ? (
          <div className="space-y-4">
            {chatMessages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-[var(--accent-primary)] text-white rounded-br-md'
                      : 'bg-[var(--card-bg)] border border-[var(--border-color)] rounded-bl-md'
                  }`}
                >
                  {msg.role === 'user' ? (
                    <div className="text-sm whitespace-pre-wrap">
                      {msg.isDeepResearch && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-700 text-white mr-2 mb-1">
                          🔬 Deep Research
                        </span>
                      )}
                      {msg.content.replace('[DEEP RESEARCH] ', '')}
                    </div>
                  ) : (
                    <div className="prose prose-sm dark:prose-invert max-w-none">
                      {/* Deep Research UI */}
                      {msg.isDeepResearch && msg.researchData ? (
                        <div className="space-y-3">
                          {/* Thinking Process - Collapsible Section */}
                          {(msg.researchData.iterations.length > 0 || msg.isStreaming) && (
                            <div className="border border-purple-200 dark:border-purple-800 rounded-lg overflow-hidden">
                              {/* Thinking Header - Click to expand/collapse */}
                              <button
                                onClick={() => toggleThinking(msg.id)}
                                className="w-full flex items-center justify-between px-3 py-2 bg-purple-50 dark:bg-purple-900/30 hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
                              >
                                <div className="flex items-center gap-2">
                                  <div className={`w-2 h-2 rounded-full ${msg.isStreaming ? 'bg-purple-500 animate-pulse' : 'bg-green-500'}`}></div>
                                  <span className="text-xs font-medium text-purple-700 dark:text-purple-300">
                                    {msg.isStreaming ? 'Thinking...' : 'Research Process'}
                                  </span>
                                  {msg.researchData.iterations.length > 0 && (
                                    <span className="text-xs text-purple-500 dark:text-purple-400">
                                      ({msg.researchData.iterations.length} {msg.researchData.iterations.length === 1 ? 'step' : 'steps'})
                                    </span>
                                  )}
                                </div>
                                <svg 
                                  className={`w-4 h-4 text-purple-500 transition-transform ${expandedThinking[msg.id] ? 'rotate-180' : ''}`} 
                                  fill="none" 
                                  viewBox="0 0 24 24" 
                                  stroke="currentColor"
                                >
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                </svg>
                              </button>
                              
                              {/* Thinking Content - Expandable */}
                              {expandedThinking[msg.id] && (
                                <div className="max-h-64 overflow-y-auto">
                                  {msg.researchData.iterations.length > 0 ? (
                                    <div className="relative">
                                      {/* Iteration Navigation */}
                                      {msg.researchData.iterations.length > 1 && (
                                        <div className="sticky top-0 z-10 flex items-center justify-between px-3 py-2 bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm border-b border-purple-100 dark:border-purple-800">
                                          <button
                                            onClick={() => navigateIteration(msg.id, 'prev', msg.researchData!.iterations.length)}
                                            disabled={(currentIterationIndex[msg.id] || 0) === 0}
                                            className={`p-1 rounded ${(currentIterationIndex[msg.id] || 0) === 0 ? 'text-gray-300 dark:text-gray-600' : 'text-purple-600 hover:bg-purple-100 dark:hover:bg-purple-900'}`}
                                            title="Previous iteration"
                                            aria-label="Previous iteration"
                                          >
                                            <FaChevronLeft size={12} />
                                          </button>
                                          <div className="flex items-center gap-2">
                                            {msg.researchData.iterations.map((_, idx) => (
                                              <div
                                                key={idx}
                                                className={`w-2 h-2 rounded-full transition-colors ${
                                                  idx === (currentIterationIndex[msg.id] || 0)
                                                    ? 'bg-purple-600'
                                                    : 'bg-purple-200 dark:bg-purple-700'
                                                }`}
                                              />
                                            ))}
                                          </div>
                                          <button
                                            onClick={() => navigateIteration(msg.id, 'next', msg.researchData!.iterations.length)}
                                            disabled={(currentIterationIndex[msg.id] || 0) === msg.researchData!.iterations.length - 1}
                                            className={`p-1 rounded ${(currentIterationIndex[msg.id] || 0) === msg.researchData!.iterations.length - 1 ? 'text-gray-300 dark:text-gray-600' : 'text-purple-600 hover:bg-purple-100 dark:hover:bg-purple-900'}`}
                                            title="Next iteration"
                                            aria-label="Next iteration"
                                          >
                                            <FaChevronRight size={12} />
                                          </button>
                                        </div>
                                      )}
                                      
                                      {/* Current Iteration Content */}
                                      <div className="p-3">
                                        <div className="text-xs font-medium text-purple-600 dark:text-purple-400 mb-2">
                                          {msg.researchData.iterations[currentIterationIndex[msg.id] || 0]?.title || 'Research Plan'}
                                        </div>
                                        <div className="text-xs text-gray-600 dark:text-gray-300 prose prose-xs max-w-none">
                                          <Markdown 
                                            content={processCitations(
                                              msg.researchData.iterations[currentIterationIndex[msg.id] || 0]?.content || '', 
                                              repoInfo, 
                                              detectCurrentBranch(repoInfo, 'main') ?? 'main'
                                            )} 
                                          />
                                        </div>
                                      </div>
                                    </div>
                                  ) : msg.isStreaming ? (
                                    <div className="p-3">
                                      <div className="flex items-center gap-2 text-xs text-purple-600 dark:text-purple-400">
                                        <div className="animate-spin w-3 h-3 border-2 border-purple-500 border-t-transparent rounded-full"></div>
                                        <span>Analyzing codebase and planning research...</span>
                                      </div>
                                      {msg.content && (
                                        <div className="mt-2 text-xs text-gray-600 dark:text-gray-300 prose prose-xs max-w-none">
                                          <Markdown content={processCitations(msg.content, repoInfo, detectCurrentBranch(repoInfo, 'main') ?? 'main')} />
                                        </div>
                                      )}
                                    </div>
                                  ) : null}
                                </div>
                              )}
                            </div>
                          )}
                          
                          {/* Final Conclusion - Main Response */}
                          {msg.researchData.finalConclusion ? (
                            <div className="pt-2">
                              <Markdown content={processCitations(msg.researchData.finalConclusion, repoInfo, detectCurrentBranch(repoInfo, 'main') ?? 'main')} />
                            </div>
                          ) : !msg.isStreaming && msg.researchData.iterations.length > 0 ? (
                            // Show the last iteration's content as the conclusion if no explicit conclusion found
                            <div className="pt-2">
                              <Markdown content={processCitations(
                                msg.researchData.iterations[msg.researchData.iterations.length - 1]?.content || msg.content, 
                                repoInfo, 
                                detectCurrentBranch(repoInfo, 'main') ?? 'main'
                              )} />
                            </div>
                          ) : !msg.isStreaming && msg.content && !msg.researchData.iterations.length ? (
                            // Fallback: show full content if no iterations parsed
                            <Markdown content={processCitations(msg.content, repoInfo, detectCurrentBranch(repoInfo, 'main') ?? 'main')} />
                          ) : msg.isStreaming && !expandedThinking[msg.id] ? (
                            // Show streaming indicator when thinking is collapsed
                            <div className="flex items-center gap-2 text-xs text-gray-500">
                              <div className="animate-pulse flex space-x-1">
                                <div className="h-1.5 w-1.5 bg-purple-600 rounded-full"></div>
                                <div className="h-1.5 w-1.5 bg-purple-600 rounded-full"></div>
                                <div className="h-1.5 w-1.5 bg-purple-600 rounded-full"></div>
                              </div>
                              <span>Researching...</span>
                            </div>
                          ) : null}
                        </div>
                      ) : msg.content ? (
                        /* Regular (non-deep-research) response */
                        <Markdown content={processCitations(msg.content, repoInfo, detectCurrentBranch(repoInfo, 'main') ?? 'main')} />
                      ) : msg.isStreaming ? (
                        <div className="flex items-center space-x-2">
                          <div className="animate-pulse flex space-x-1">
                            <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                            <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                            <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                          </div>
                          <span className="text-xs text-gray-500 dark:text-gray-400">Thinking...</span>
                        </div>
                      ) : null}
                    </div>
                  )}
                  {/* Timestamp */}
                  <div className={`text-xs mt-1 ${msg.role === 'user' ? 'text-white/70' : 'text-gray-400'}`}>
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    {msg.isStreaming && <span className="ml-2 italic">typing...</span>}
                  </div>
                </div>
              </div>
            ))}

            {/* Action buttons when there are messages */}
            {chatMessages.some(m => m.role === 'assistant' && m.content) && (
              <div className="flex justify-center pt-2">
                <div className="flex items-center space-x-2">
                  {/* Download button */}
                  <button
                    onClick={downloadresponse}
                    className="text-xs text-gray-500 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 px-3 py-1.5 rounded-full bg-[var(--card-bg)] border border-[var(--border-color)] hover:border-green-500 flex items-center gap-1 transition-colors"
                    title="Download response as markdown file"
                  >
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Download
                  </button>

                  {/* Clear button */}
                  <button
                    id="ask-clear-conversation"
                    onClick={clearConversation}
                    className="text-xs text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 px-3 py-1.5 rounded-full bg-[var(--card-bg)] border border-[var(--border-color)] hover:border-red-500 transition-colors"
                  >
                    Clear chat
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Empty state when no messages */
          <div className="flex flex-col items-center justify-center h-full text-[var(--muted)]">
            <svg className="w-16 h-16 mb-4 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <p className="text-base font-medium">Ask a question about this codebase</p>
            <p className="text-sm opacity-70 mt-1">Your conversation will appear here</p>
            {/* Connection status in empty state */}
            {connectionStatus === 'error' && (
              <div className="mt-4 px-3 py-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 max-w-xs">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
                  <span className="text-xs font-medium text-red-700 dark:text-red-400">Chatbot unavailable</span>
                </div>
                <p className="text-xs text-red-600 dark:text-red-400 mt-1 break-words">{connectionError}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Connection status bar */}
      {connectionStatus !== 'idle' && (
        <div className="flex-shrink-0 px-4 py-1.5 border-t border-[var(--border-color)] flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
            connectionStatus === 'checking' ? 'bg-yellow-400 animate-pulse' :
            connectionStatus === 'connected' ? 'bg-green-500' :
            'bg-red-500'
          }`} />
          <span className={`text-xs truncate ${
            connectionStatus === 'checking' ? 'text-yellow-600 dark:text-yellow-400' :
            connectionStatus === 'connected' ? 'text-green-600 dark:text-green-400' :
            'text-red-600 dark:text-red-400'
          }`}>
            {connectionStatus === 'checking' ? 'Connecting to chatbot...' :
             connectionStatus === 'connected' ? 'Connected' :
             connectionError || 'Connection failed'}
          </span>
        </div>
      )}

      {/* Question input - fixed at bottom */}
      <div className="flex-shrink-0 p-4 border-t border-[var(--border-color)] bg-[var(--card-bg)]">
        <form onSubmit={handleSubmit}>
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={messages.ask?.placeholder || 'What would you like to know about this codebase?'}
              className="block w-full rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] text-[var(--foreground)] px-5 py-3.5 text-base shadow-sm focus:border-[var(--accent-primary)] focus:ring-2 focus:ring-[var(--accent-primary)]/30 focus:outline-none transition-all"
              style={{ paddingRight: `${buttonWidth + 24}px` }}
              disabled={isLoading}
            />
            <button
              ref={buttonRef}
              type="submit"
              disabled={isLoading || !question.trim()}
              className={`absolute right-3 top-1/2 transform -translate-y-1/2 px-4 py-2 rounded-md font-medium text-sm ${
                isLoading || !question.trim()
                  ? 'bg-[var(--button-disabled-bg)] text-[var(--button-disabled-text)] cursor-not-allowed'
                  : 'bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary)]/90 shadow-sm'
              } transition-all duration-200 flex items-center gap-1.5`}
            >
              {isLoading ? (
                <div className="w-4 h-4 rounded-full border-2 border-t-transparent border-white animate-spin" />
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                  <span>{messages.ask?.askButton || 'Ask'}</span>
                </>
              )}
            </button>
          </div>

          {/* Deep Research toggle */}
          <div className="flex items-center mt-2 justify-between">
            <div className="group relative">
              <label className="flex items-center cursor-pointer">
                <span className="text-xs text-gray-600 dark:text-gray-400 mr-2">Deep Research</span>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={deepResearch}
                    onChange={() => setDeepResearch(!deepResearch)}
                    className="sr-only"
                  />
                  <div className={`w-10 h-5 rounded-full transition-colors ${deepResearch ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}`}></div>
                  <div className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform transform ${deepResearch ? 'translate-x-5' : ''}`}></div>
                </div>
              </label>
              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs rounded p-2 w-72 z-10">
                <div className="relative">
                  <div className="absolute -bottom-2 left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
                  <p className="mb-1">Deep Research conducts a multi-turn investigation process:</p>
                  <ul className="list-disc pl-4 text-xs">
                    <li><strong>Initial Research:</strong> Creates a research plan and initial findings</li>
                    <li><strong>Iteration 1:</strong> Explores specific aspects in depth</li>
                    <li><strong>Iteration 2:</strong> Investigates remaining questions</li>
                    <li><strong>Iterations 3-4:</strong> Dives deeper into complex areas</li>
                    <li><strong>Final Conclusion:</strong> Comprehensive answer based on all iterations</li>
                  </ul>
                  <p className="mt-1 text-xs italic">The AI automatically continues research until complete (up to 5 iterations)</p>
                </div>
              </div>
            </div>
            {deepResearch && (
              <div className="text-xs text-purple-600 dark:text-purple-400">
                Multi-turn research process enabled
                {researchIteration > 0 && !researchComplete && ` (iteration ${researchIteration})`}
                {researchComplete && ` (complete)`}
              </div>
            )}
          </div>
        </form>
      </div>

      {/* Model Selection Modal */}
      <ModelSelectionModal
        isOpen={isModelSelectionModalOpen}
        onClose={() => setIsModelSelectionModalOpen(false)}
        provider={selectedProvider}
        setProvider={setSelectedProvider}
        model={selectedModel}
        setModel={setSelectedModel}
        isCustomModel={isCustomSelectedModel}
        setIsCustomModel={setIsCustomSelectedModel}
        customModel={customSelectedModel}
        setCustomModel={setCustomSelectedModel}
        isComprehensiveView={isComprehensiveView}
        setIsComprehensiveView={setIsComprehensiveView}
        showFileFilters={false}
        onApply={() => {
          console.log('Model selection applied:', selectedProvider, selectedModel);
        }}
        showWikiType={false}
        authRequired={false}
        isAuthLoading={false}
      />
    </div>
  );
};

export default Ask;
