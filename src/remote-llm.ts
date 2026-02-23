/**
 * remote-llm.ts - Remote LLM client for QMD
 *
 * Implements the same LlamaCpp interface but proxies all inference to a remote
 * QMD server (started with `qmd serve`). This allows machines without a GPU
 * to use QMD's full hybrid search pipeline.
 *
 * Configure via environment:
 *   QMD_REMOTE_URL=https://mbp.example.com:8282
 *   QMD_AUTH_TOKEN=optional-secret
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankDocument,
  RerankOptions,
  RerankResult,
  Queryable,
  TokenLogProb,
} from "./llm.js";

// Re-export types the caller may need
export type { EmbeddingResult, GenerateResult, RerankResult, Queryable };

export type RemoteLLMConfig = {
  /** Base URL of the remote QMD server (e.g., https://mbp.example.com:8282) */
  url: string;
  /** Optional bearer token for auth */
  authToken?: string;
  /** Request timeout in ms (default: 120s — reranking can be slow) */
  timeoutMs?: number;
};

/**
 * Remote LLM implementation that proxies to a QMD inference server.
 *
 * This class is a drop-in replacement for LlamaCpp. It implements the same
 * LLM interface plus the extra methods (tokenize, countTokens, embedBatch,
 * getDeviceInfo) that the codebase calls directly on the LlamaCpp instance.
 */
export class RemoteLLM implements LLM {
  private baseUrl: string;
  private authToken?: string;
  private timeoutMs: number;

  constructor(config: RemoteLLMConfig) {
    // Strip trailing slash
    this.baseUrl = config.url.replace(/\/+$/, "");
    this.authToken = config.authToken;
    this.timeoutMs = config.timeoutMs ?? 120_000;
  }

  // =========================================================================
  // HTTP helper
  // =========================================================================

  private async post<T>(path: string, body: unknown): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.authToken) {
      headers["Authorization"] = `Bearer ${this.authToken}`;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const resp = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`QMD remote ${path} failed (${resp.status}): ${text}`);
      }

      return (await resp.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  private async get<T>(path: string): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {};
    if (this.authToken) {
      headers["Authorization"] = `Bearer ${this.authToken}`;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const resp = await fetch(url, {
        method: "GET",
        headers,
        signal: controller.signal,
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`QMD remote ${path} failed (${resp.status}): ${text}`);
      }

      return (await resp.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  // =========================================================================
  // LLM interface
  // =========================================================================

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.post<EmbeddingResult | null>("/embed", { text, options });
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this.post<GenerateResult | null>("/generate", { prompt, options });
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return this.post<ModelInfo>("/model-exists", { model });
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    return this.post<Queryable[]>("/expand-query", { query, options });
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    return this.post<RerankResult>("/rerank", { query, documents, options });
  }

  async dispose(): Promise<void> {
    // Nothing to dispose on the client side
  }

  // =========================================================================
  // Extra methods called directly on LlamaCpp by the codebase
  // =========================================================================

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.post<(EmbeddingResult | null)[]>("/embed-batch", { texts });
  }

  async tokenize(text: string): Promise<readonly number[]> {
    const result = await this.post<{ tokens: number[] }>("/tokenize", { text });
    return result.tokens;
  }

  async countTokens(text: string): Promise<number> {
    const result = await this.post<{ count: number }>("/count-tokens", { text });
    return result.count;
  }

  async detokenize(tokens: readonly number[]): Promise<string> {
    const result = await this.post<{ text: string }>("/detokenize", { tokens: Array.from(tokens) });
    return result.text;
  }

  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    return this.get("/device");
  }

  // =========================================================================
  // Inactivity / lifecycle stubs (no-ops for remote)
  // =========================================================================

  async unloadIdleResources(): Promise<void> {
    // No-op: resources live on the server
  }

  touchActivity(): void {
    // No-op
  }
}
