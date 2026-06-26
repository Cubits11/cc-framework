import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";
import { DashboardShell } from "../components/DashboardShell";
import { verifyEnterpriseBundle } from "../lib/merkle";
import type { EnterpriseBundle } from "../lib/types";

describe("enterprise dashboard smoke", () => {
  test("uploads a bundle, renders all three views, and verifies Merkle proofs client-side", async () => {
    const bundlePath = process.env.ENTERPRISE_BUNDLE_PATH;
    expect(bundlePath).toBeTruthy();
    const raw = readFileSync(String(bundlePath), "utf-8");
    const bundle = JSON.parse(raw) as EnterpriseBundle;
    const user = userEvent.setup();

    render(<DashboardShell />);

    await user.click(screen.getByRole("tab", { name: /verify/i }));
    const file = new File([raw], "enterprise_bundle.json", { type: "application/json" });
    await user.upload(screen.getByLabelText(/evidence bundle json/i), file);

    await waitFor(() => expect(screen.getByText(/passed/i)).toBeInTheDocument());
    const proofResult = await verifyEnterpriseBundle(bundle);
    expect(proofResult.ok).toBe(true);
    expect(proofResult.inclusionOk).toBe(true);
    expect(proofResult.consistencyOk).toBe(true);

    await user.click(screen.getByRole("tab", { name: /composition risk/i }));
    expect(screen.getByRole("img", { name: /frechet-hoeffding envelope/i })).toBeInTheDocument();
    expect(screen.getAllByText(/empirical estimate/i).length).toBeGreaterThan(0);

    await user.click(screen.getByRole("tab", { name: /assurance case/i }));
    expect(screen.getByRole("tree")).toBeInTheDocument();
    expect(screen.getAllByText(/defeater/i).length).toBeGreaterThan(0);
  });
});
