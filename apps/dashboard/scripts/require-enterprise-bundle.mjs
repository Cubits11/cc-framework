import { existsSync } from "node:fs";

const bundlePath = process.env.ENTERPRISE_BUNDLE_PATH;

if (!bundlePath) {
  console.error(
    [
      "ENTERPRISE_BUNDLE_PATH is required for dashboard smoke validation.",
      "Run `make enterprise-smoke` from the repository root to generate a moto-backed bundle and run this through the e2e pipeline.",
      "For direct dashboard work, set ENTERPRISE_BUNDLE_PATH to a generated enterprise dashboard bundle JSON file.",
    ].join("\n"),
  );
  process.exit(1);
}

if (!existsSync(bundlePath)) {
  console.error(`ENTERPRISE_BUNDLE_PATH does not exist: ${bundlePath}`);
  process.exit(1);
}
