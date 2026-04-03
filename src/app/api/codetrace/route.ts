import { NextRequest, NextResponse } from 'next/server';

/**
 * POST /api/codetrace — Generate a code trace.
 * Proxies to backend FastAPI POST /api/codetrace.
 */

const BACKEND_PORT = process.env.FASTAPI_PORT || '8001';

/* eslint-disable @typescript-eslint/no-explicit-any */
function normalizeKeys(data: any): any {
  if (Array.isArray(data)) return data.map(normalizeKeys);
  if (data && typeof data === 'object') {
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(data)) {
      const camel = k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
      out[camel] = normalizeKeys(v);
    }
    return out;
  }
  return data;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const backendUrl = `http://127.0.0.1:${BACKEND_PORT}/api/codetrace`;
    const resp = await fetch(backendUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.text();
      console.error(`[codetrace] Backend error ${resp.status}: ${err}`);
      return NextResponse.json(
        { error: `Backend error: ${resp.status}` },
        { status: resp.status },
      );
    }

    const data = await resp.json();
    return NextResponse.json(normalizeKeys(data));
  } catch (err) {
    console.error('[codetrace] Error:', err);
    return NextResponse.json(
      { error: 'Failed to generate code trace' },
      { status: 500 },
    );
  }
}
