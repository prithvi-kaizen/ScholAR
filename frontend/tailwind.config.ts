import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#050505",
        panel: "#101113",
        panelSoft: "#17191d",
        line: "#2a2d33",
        muted: "#9ca3af",
        acid: "#b7f96d"
      },
      boxShadow: {
        glow: "0 18px 60px rgba(0, 0, 0, 0.55)"
      }
    }
  },
  plugins: []
};

export default config;
