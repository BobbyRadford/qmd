/**
 * auth.ts - Token management for QMD remote inference
 *
 * Stores auth tokens in ~/.config/qmd/auth.json.
 * Both server and client read from this file automatically.
 * Environment variable QMD_AUTH_TOKEN overrides the stored token.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import { randomBytes } from "crypto";

const CONFIG_DIR = join(homedir(), ".config", "qmd");
const AUTH_FILE = join(CONFIG_DIR, "auth.json");

type AuthConfig = {
  token: string;
  createdAt: string;
};

/**
 * Get the auth config file path.
 */
export function getAuthPath(): string {
  return AUTH_FILE;
}

/**
 * Read the stored auth token, or null if none exists.
 */
function readStoredToken(): string | null {
  try {
    if (!existsSync(AUTH_FILE)) return null;
    const data = JSON.parse(readFileSync(AUTH_FILE, "utf-8")) as AuthConfig;
    return data.token || null;
  } catch {
    return null;
  }
}

/**
 * Get the active auth token.
 * Priority: QMD_AUTH_TOKEN env var > ~/.config/qmd/auth.json
 */
export function getAuthToken(): string | null {
  return process.env.QMD_AUTH_TOKEN || readStoredToken();
}

/**
 * Generate a new random token and save it.
 * Returns the new token.
 */
export function generateToken(): string {
  const token = randomBytes(32).toString("base64url");
  mkdirSync(CONFIG_DIR, { recursive: true });

  const config: AuthConfig = {
    token,
    createdAt: new Date().toISOString(),
  };

  writeFileSync(AUTH_FILE, JSON.stringify(config, null, 2) + "\n", { mode: 0o600 });
  return token;
}

/**
 * Revoke (delete) the stored token.
 */
export function revokeToken(): boolean {
  try {
    if (existsSync(AUTH_FILE)) {
      const { unlinkSync } = require("fs");
      unlinkSync(AUTH_FILE);
      return true;
    }
  } catch { /* */ }
  return false;
}
