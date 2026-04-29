/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverComponentsExternalPackages: ["apify-client", "proxy-agent"],
  },
};

module.exports = nextConfig;
