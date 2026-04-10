'use client';

import React from 'react';

interface AzureIconProps {
  className?: string;
  size?: number;
}

/**
 * Orcas Logo — stylized killer whale icon for app branding.
 * A sleek, modern orca silhouette leaping out of water.
 */
export const OrcasLogo: React.FC<{ className?: string; size?: number }> = ({ className = '', size = 28 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 64 64"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="Orcas"
  >
    {/* Orca body — leaping arc */}
    <path
      d="M14 44c0 0 4-6 10-14c4-5.5 8-10 14-14c4-2.5 8-3.5 12-2c3 1.2 5 4 5.5 7.5c0.5 4-1 8-4 11c-4 4-9 6-14 7c-4 0.8-8 0.5-11-1c-2.5-1.2-4-3-5-5.5"
      fill="currentColor"
      strokeLinejoin="round"
    />
    {/* Orca dorsal fin */}
    <path
      d="M36 16c-1-6-0.5-10 2-13c0.5 3 2 6 4 9c1.5 2.5 2.5 4 2 5.5c-1 2-4 2-6 0.5c-0.8-0.6-1.5-1.2-2-2z"
      fill="currentColor"
    />
    {/* Orca tail fluke */}
    <path
      d="M10 46c-3 2-6 2.5-8 1c1.5-1 3-3 4.5-5c1-1.2 2-2 3-1.5c1.5 0.8 2 3 0.5 5.5z"
      fill="currentColor"
    />
    {/* Eye patch — the iconic orca white spot */}
    <ellipse
      cx="44"
      cy="23"
      rx="3"
      ry="2"
      transform="rotate(-25 44 23)"
      fill="var(--card-bg, #ffffff)"
      opacity="0.95"
    />
    {/* Belly — lighter underbody */}
    <path
      d="M20 38c3-4 7-8 12-11c3.5-2 7-3.5 10-3c-2 2-5 5-8 8c-3 3-6 5.5-9 6.5c-2.5 0.8-4 0.5-5-0.5z"
      fill="var(--card-bg, #ffffff)"
      opacity="0.25"
    />
    {/* Water splash — stylized wave at base */}
    <path
      d="M6 52c2-1.5 5-2 8-1c2 0.7 4 0.5 6-0.5c2.5-1.2 5-1 7 0.5c2 1.2 4 1.5 6 0.5c2-1 4-1.2 6-0.5c2 0.7 4 0.5 5.5-0.5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      fill="none"
      opacity="0.35"
    />
  </svg>
);

/**
 * Microsoft Azure DevOps-style icon (legacy, kept for compatibility).
 */
export const AzureIcon: React.FC<AzureIconProps> = ({ className = '', size = 24 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 18 18"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="Azure"
  >
    <path
      d="M6.5 1.5L1 16h4.5l1.2-3h5.6l1.2 3H18L12.5 1.5H6.5z"
      fill="currentColor"
      opacity="0.9"
    />
    <path
      d="M9.5 5.5L7.2 11.5h4.6L9.5 5.5z"
      fill="var(--card-bg, #ffffff)"
    />
  </svg>
);

/**
 * Microsoft logo wordmark for footer/branding areas.
 */
export const MicrosoftLogo: React.FC<{ className?: string }> = ({ className = '' }) => (
  <svg
    width="108"
    height="24"
    viewBox="0 0 108 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="Microsoft"
  >
    {/* Microsoft four-square logo */}
    <rect x="0" y="0" width="10" height="10" fill="#F25022" />
    <rect x="12" y="0" width="10" height="10" fill="#7FBA00" />
    <rect x="0" y="12" width="10" height="10" fill="#00A4EF" />
    <rect x="12" y="12" width="10" height="10" fill="#FFB900" />
  </svg>
);

/**
 * @deprecated Use OrcasLogo instead. Kept for backward compatibility.
 */
export const AzureDevOpsIcon = OrcasLogo;

export default AzureIcon;
