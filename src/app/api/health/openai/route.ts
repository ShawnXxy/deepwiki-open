import { NextResponse } from 'next/server';

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

export async function GET() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    const backendResponse = await fetch(`${TARGET_SERVER_BASE_URL}/health/openai`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Backend unreachable';
    return NextResponse.json(
      { status: 'error', message: `Backend connection failed: ${message}` },
      { status: 502 }
    );
  }
}
