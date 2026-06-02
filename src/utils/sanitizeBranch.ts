/**
 * Branch-name sanitisation for cache filenames and storage paths.
 *
 * This MUST stay in exact parity with the Python source of truth
 * `sanitize_branch_for_path()` in `backend/utils/filter.py`, because the
 * backend (Python) and the Next.js API routes (TypeScript) both construct
 * wiki/codemap cache filenames for the same artifact. If the two diverge,
 * the same branch produces two different filenames (e.g. `8.0` vs `8-0`),
 * creating duplicate caches that never reconcile.
 *
 * Canonical form is DASH: every character outside `[a-zA-Z0-9_-]` (including
 * dots and slashes) is replaced with `-`, runs of `-` are collapsed, and
 * leading/trailing `-` are trimmed.
 */
export function sanitizeBranchForPath(
  branch?: string | null,
  fallback = 'main',
): string {
  if (!branch || !branch.trim()) {
    return fallback;
  }
  let sanitized = branch.trim().replace(/[^a-zA-Z0-9_-]/g, '-');
  sanitized = sanitized.replace(/-+/g, '-').replace(/^-+|-+$/g, '');
  return sanitized || fallback;
}
