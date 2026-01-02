import { NextRequest, NextResponse } from 'next/server';

// Get the server base URL for backend API calls
// In containerized deployments, frontend and backend run in the same container
// Frontend on port 3000, Backend on port 8001, both accessible via localhost internally
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const getTargetServerUrl = (_req: NextRequest) => {
  // Use environment variable if explicitly set
  if (process.env.SERVER_BASE_URL) {
    return process.env.SERVER_BASE_URL;
  }
  
  // In server-side API routes, always use localhost:8001 to reach the backend
  // This works in:
  // - Local development (both services on localhost)
  // - Docker containers (both services in same container)
  // - Azure Container Apps (both services in same container)
  // The external hostname (e.g., *.azurecontainerapps.io) only exposes port 3000/443
  return 'http://localhost:8001';
};

// This is a fallback HTTP implementation that will be used if WebSockets are not available
// or if there's an error with the WebSocket connection
export async function POST(req: NextRequest) {
  try {
    const requestBody = await req.json(); // Assuming the frontend sends JSON

    // Note: This endpoint now uses the HTTP fallback instead of WebSockets
    // The WebSocket implementation is in src/utils/websocketClient.ts
    // This HTTP endpoint is kept for backward compatibility
    console.log('Using HTTP fallback for chat completion instead of WebSockets');

    const targetServerUrl = getTargetServerUrl(req);
    const targetUrl = `${targetServerUrl}/chat/completions/stream`;
    
    console.log(`Proxying request to: ${targetUrl}`);

    // Make the actual request to the backend service with increased timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minute timeout for large repos

    try {
      const backendResponse = await fetch(targetUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream', // Indicate that we expect a stream
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // If the backend service returned an error, forward that error to the client
      if (!backendResponse.ok) {
        const errorBody = await backendResponse.text();
        const errorHeaders = new Headers();
        backendResponse.headers.forEach((value, key) => {
          errorHeaders.set(key, value);
        });
        return new NextResponse(errorBody, {
          status: backendResponse.status,
          statusText: backendResponse.statusText,
          headers: errorHeaders,
        });
      }

      // Ensure the backend response has a body to stream
      if (!backendResponse.body) {
        return new NextResponse('Stream body from backend is null', { status: 500 });
      }

      // Create a new ReadableStream to pipe the data from the backend to the client
      const stream = new ReadableStream({
        async start(controller) {
          const reader = backendResponse.body!.getReader();
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) {
                break;
              }
              controller.enqueue(value);
            }
          } catch (error) {
            console.error('Error reading from backend stream in proxy:', error);
            controller.error(error);
          } finally {
            controller.close();
            reader.releaseLock(); // Important to release the lock on the reader
          }
        },
        cancel(reason) {
          console.log('Client cancelled stream request:', reason);
        }
      });

      // Set up headers for the response to the client
      const responseHeaders = new Headers();
      // Copy the Content-Type from the backend response (e.g., 'text/event-stream')
      const contentType = backendResponse.headers.get('Content-Type');
      if (contentType) {
        responseHeaders.set('Content-Type', contentType);
      }
      // It's good practice for streams not to be cached or transformed by intermediaries.
      responseHeaders.set('Cache-Control', 'no-cache, no-transform');

      return new NextResponse(stream, {
        status: backendResponse.status, // Should be 200 for a successful stream start
        headers: responseHeaders,
      });

    } catch (fetchError) {
      clearTimeout(timeoutId);
      console.error('Error fetching from backend:', fetchError);
      
      if (fetchError instanceof Error) {
        if (fetchError.name === 'AbortError') {
          return new NextResponse(JSON.stringify({ 
            error: 'Request timeout - backend took too long to respond',
            retryable: true,
            suggestion: 'The backend may be processing another request. Please wait and try again.'
          }), {
            status: 504,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        
        // Check for headers timeout (backend busy processing)
        const cause = (fetchError as Error & { cause?: { code?: string } }).cause;
        if (cause?.code === 'UND_ERR_HEADERS_TIMEOUT') {
          console.error(`Backend busy (headers timeout): ${targetUrl}`);
          return new NextResponse(JSON.stringify({ 
            error: 'Backend is busy processing another request. The request will be retried automatically.',
            retryable: true,
            errorType: 'BackendBusy',
            suggestion: 'Large repositories take several minutes per page. Please wait for current processing to complete.'
          }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        
        // Handle connection refused/network errors more explicitly
        const errorMsg = fetchError.message || 'Unknown error';
        if (errorMsg.includes('fetch failed') || errorMsg.includes('ECONNREFUSED') || errorMsg.includes('ENOTFOUND')) {
          console.error(`Backend connection failed: ${targetUrl} - ${errorMsg}`);
          return new NextResponse(JSON.stringify({ 
            error: `Backend connection failed: ${errorMsg}. Please ensure the backend service is running.`,
            retryable: true,
            details: { targetUrl, errorType: fetchError.name }
          }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' },
          });
        }
      }
      
      throw fetchError; // Re-throw to be caught by outer catch
    }

  } catch (error) {
    console.error('Error in API proxy route (/api/chat/stream):', error);
    let errorMessage = 'Internal Server Error in proxy';
    let statusCode = 500;
    
    if (error instanceof Error) {
      errorMessage = error.message;
      // Handle specific error types
      if (error.message.includes('fetch failed') || error.message.includes('ECONNREFUSED')) {
        errorMessage = `Unable to connect to backend service: ${error.message}`;
        statusCode = 503; // Service Unavailable
      }
    }
    
    return new NextResponse(JSON.stringify({ error: errorMessage }), {
      status: statusCode,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// Optional: Handle OPTIONS requests for CORS if you ever call this from a different origin
// or use custom headers that trigger preflight requests. For same-origin, it's less critical.
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204, // No Content
    headers: {
      'Access-Control-Allow-Origin': '*', // Be more specific in production if needed
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization', // Adjust as per client's request headers
    },
  });
}