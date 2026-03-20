import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone',
  // Security: Disable verbose logging to prevent token exposure in URLs
  logging: {
    fetches: {
      fullUrl: false, // Don't log full URLs with query params
    },
  },
  // Suppress verbose HTTP request logging (tokens in query params)
  // Note: NODE_ENV=production already reduces logging, but this adds explicit control
  devIndicators: false,
  // Optimize build for Docker
  experimental: {
    optimizePackageImports: ['@mermaid-js/mermaid', 'react-syntax-highlighter'],
  },
  // Reduce memory usage during build
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }
    // Optimize bundle size
    config.optimization = {
      ...config.optimization,
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
          },
        },
      },
    };
    return config;
  },
  // No rewrites needed — wiki_cache, projects, lang/config, and export
  // are all handled by Next.js API routes (no backend proxy required).
  // The Ask/Chat feature connects directly via WebSocket when the backend
  // is available, and fails gracefully when it's not.
};

export default nextConfig;
