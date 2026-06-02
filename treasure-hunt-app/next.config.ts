import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["@prisma/client", "prisma"],
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
