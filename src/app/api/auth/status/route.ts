import { NextResponse } from "next/server";

export async function GET() {
  // Read-only viewer: auth not required for browsing cached wikis.
  // The code_processor handles authentication at generation time.
  return NextResponse.json({ auth_required: false, mode: "none" });
}
