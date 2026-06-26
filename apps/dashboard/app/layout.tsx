import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "CC Framework Evidence Dashboard",
  description: "Explanatory evidence dashboard for CC Framework enterprise smoke bundles.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
