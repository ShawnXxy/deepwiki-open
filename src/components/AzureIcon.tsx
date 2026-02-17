'use client';

import React from 'react';

interface AzureIconProps {
  className?: string;
  size?: number;
}

/**
 * Microsoft Azure DevOps-style icon for branding.
 * Uses the Azure triangle/swoosh motif.
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
    {/* Azure triangle motif */}
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
 * Azure DevOps icon for the app header.
 */
export const AzureDevOpsIcon: React.FC<{ className?: string; size?: number }> = ({ className = '', size = 28 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="Azure DevOps"
  >
    <path
      d="M22 4v16l-6 2V6L2 16v-4l14-10h6z"
      fill="currentColor"
    />
    <path
      d="M2 8l6-6v4.5L2 12V8z"
      fill="currentColor"
      opacity="0.7"
    />
  </svg>
);

export default AzureIcon;
