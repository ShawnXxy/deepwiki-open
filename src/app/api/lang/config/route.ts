import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

/**
 * GET /api/lang/config — Return supported languages.
 * Reads from backend/config/lang.json directly (no backend needed).
 */
export async function GET() {
  // Try reading from backend/config/lang.json (works in dev and Docker)
  const candidates = [
    path.join(process.cwd(), 'backend', 'config', 'lang.json'),
    path.join(process.cwd(), '..', 'backend', 'config', 'lang.json'),
  ];

  for (const filePath of candidates) {
    if (fs.existsSync(filePath)) {
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        return NextResponse.json(JSON.parse(content));
      } catch (err) {
        console.error('Error reading lang.json:', err);
      }
    }
  }

  // Fallback: hardcoded defaults
  return NextResponse.json({
    supported_languages: {
      en: 'English',
      ja: 'Japanese (日本語)',
      zh: 'Mandarin Chinese (中文)',
      'zh-tw': 'Traditional Chinese (繁體中文)',
      es: 'Spanish (Español)',
      kr: 'Korean (한국어)',
      vi: 'Vietnamese (Tiếng Việt)',
      'pt-br': 'Brazilian Portuguese (Português Brasileiro)',
      fr: 'Français (French)',
      ru: 'Русский (Russian)',
    },
    default: 'en',
  });
}
