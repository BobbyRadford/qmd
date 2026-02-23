/**
 * remote-server.ts - HTTP server for remote QMD inference
 *
 * Run this on a machine with a GPU (e.g., a Mac with Apple Silicon).
 * It loads GGUF models via node-llama-cpp and exposes them over HTTP.
 *
 * Usage:
 *   qmd serve [--port 8282] [--host 0.0.0.0]
 *
 * Then point the client machine at it:
 *   QMD_REMOTE_URL=https://mbp.example.com qmd query "my search"
 */

import { createServer, type IncomingMessage, type ServerResponse } from "http";
import { getAuthToken } from "./auth.js";
import { LlamaCpp } from "./llm.js";
import type {
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankDocument,
  RerankOptions,
  RerankResult,
  Queryable,
} from "./llm.js";

// =============================================================================
// Request/Response Types
// =============================================================================

type EmbedRequest = { text: string; options?: EmbedOptions };
type EmbedBatchRequest = { texts: string[] };
type GenerateRequest = { prompt: string; options?: GenerateOptions };
type ExpandQueryRequest = { query: string; options?: { context?: string; includeLexical?: boolean } };
type RerankRequest = { query: string; documents: RerankDocument[]; options?: RerankOptions };
type TokenizeRequest = { text: string };
type CountTokensRequest = { text: string };
type DetokenizeRequest = { tokens: number[] };
type ModelExistsRequest = { model: string };

// =============================================================================
// Server
// =============================================================================

export async function startRemoteServer(options: {
  port?: number;
  host?: string;
  authToken?: string;
} = {}): Promise<void> {
  const port = options.port ?? 8282;
  const host = options.host ?? "0.0.0.0";
  const authToken = options.authToken ?? getAuthToken();

  const llm = new LlamaCpp();

  // Pre-warm: trigger model loading so first request isn't slow
  console.log("Loading models...");
  const warmStart = Date.now();
  try {
    // Trigger embed model load
    await llm.embed("warmup");
    console.log(`  Embed model loaded (${((Date.now() - warmStart) / 1000).toFixed(1)}s)`);

    // Trigger generate model load
    const genStart = Date.now();
    await llm.generate("warmup", { maxTokens: 1 });
    console.log(`  Generate model loaded (${((Date.now() - genStart) / 1000).toFixed(1)}s)`);

    // Trigger rerank model load
    const rerankStart = Date.now();
    await llm.rerank("warmup", [{ file: "test", text: "warmup text" }]);
    console.log(`  Rerank model loaded (${((Date.now() - rerankStart) / 1000).toFixed(1)}s)`);
  } catch (err) {
    console.error("Warning: model pre-warming failed:", err);
  }
  console.log(`Models ready in ${((Date.now() - warmStart) / 1000).toFixed(1)}s`);

  // Helper to read JSON body
  function readBody(req: IncomingMessage): Promise<string> {
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
      req.on("error", reject);
    });
  }

  function sendJSON(res: ServerResponse, status: number, data: unknown): void {
    const body = JSON.stringify(data);
    res.writeHead(status, {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(body),
    });
    res.end(body);
  }

  function sendError(res: ServerResponse, status: number, message: string): void {
    sendJSON(res, status, { error: message });
  }

  const server = createServer(async (req, res) => {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    const path = url.pathname;
    const method = req.method?.toUpperCase();

    // Auth check
    if (authToken) {
      const provided = req.headers.authorization?.replace(/^Bearer\s+/i, "");
      if (provided !== authToken) {
        sendError(res, 401, "Unauthorized");
        return;
      }
    }

    // Health check
    if (path === "/health" && method === "GET") {
      sendJSON(res, 200, { status: "ok", uptime: process.uptime() });
      return;
    }

    // Device info
    if (path === "/device" && method === "GET") {
      try {
        const info = await llm.getDeviceInfo();
        sendJSON(res, 200, info);
      } catch (err: any) {
        sendError(res, 500, err.message);
      }
      return;
    }

    // All other routes are POST with JSON body
    if (method !== "POST") {
      sendError(res, 405, "Method not allowed");
      return;
    }

    let body: any;
    try {
      const raw = await readBody(req);
      body = JSON.parse(raw);
    } catch {
      sendError(res, 400, "Invalid JSON body");
      return;
    }

    try {
      switch (path) {
        case "/embed": {
          const { text, options } = body as EmbedRequest;
          const result = await llm.embed(text, options);
          sendJSON(res, 200, result);
          break;
        }

        case "/embed-batch": {
          const { texts } = body as EmbedBatchRequest;
          const result = await llm.embedBatch(texts);
          sendJSON(res, 200, result);
          break;
        }

        case "/generate": {
          const { prompt, options } = body as GenerateRequest;
          const result = await llm.generate(prompt, options);
          sendJSON(res, 200, result);
          break;
        }

        case "/expand-query": {
          const { query, options } = body as ExpandQueryRequest;
          const result = await llm.expandQuery(query, options);
          sendJSON(res, 200, result);
          break;
        }

        case "/rerank": {
          const { query, documents, options } = body as RerankRequest;
          const result = await llm.rerank(query, documents, options);
          sendJSON(res, 200, result);
          break;
        }

        case "/tokenize": {
          const { text } = body as TokenizeRequest;
          const tokens = await llm.tokenize(text);
          // Convert opaque Token objects to numbers for wire transfer
          sendJSON(res, 200, { tokens: Array.from(tokens) });
          break;
        }

        case "/count-tokens": {
          const { text } = body as CountTokensRequest;
          const count = await llm.countTokens(text);
          sendJSON(res, 200, { count });
          break;
        }

        case "/detokenize": {
          const { tokens } = body as DetokenizeRequest;
          const text = await llm.detokenize(tokens as any);
          sendJSON(res, 200, { text });
          break;
        }

        case "/model-exists": {
          const { model } = body as ModelExistsRequest;
          const result = await llm.modelExists(model);
          sendJSON(res, 200, result);
          break;
        }

        default:
          sendError(res, 404, `Unknown route: ${path}`);
      }
    } catch (err: any) {
      console.error(`Error handling ${path}:`, err);
      sendError(res, 500, err.message || "Internal server error");
    }
  });

  server.listen(port, host, () => {
    console.log(`\nQMD inference server listening on http://${host}:${port}`);
    console.log(`\nSet on client: QMD_REMOTE_URL=http://<this-machine>:${port}`);
    if (authToken) {
      console.log(`Auth enabled: set QMD_AUTH_TOKEN on client`);
    }
  });

  // Graceful shutdown
  const shutdown = async () => {
    console.log("\nShutting down...");
    server.close();
    await llm.dispose();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}
