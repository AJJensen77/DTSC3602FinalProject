import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Old ReelPicks page names now live under the list-style pages.
  async redirects() {
    return [
      { source: "/rate", destination: "/browse", permanent: false },
      { source: "/my-ratings", destination: "/list", permanent: false },
    ];
  },
};

export default nextConfig;
