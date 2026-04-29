/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  serverExternalPackages: ["apify-client", "proxy-agent"],
};

module.exports = nextConfig;
