import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

import { afterEach, describe, expect, it } from "@effect/vitest";
import { Effect } from "effect";
import type { Tool } from "@executor-js/sdk";

import { fuseRankings, keywordWeight } from "./embedding-search";
import { searchTools } from "./tool-invoker";

// Each vocabulary slot is a dimension; synonyms share one, so "colleague"
// matches "person" by embedding while sharing no token with it.
const VOCAB = [
  ["person", "colleague", "people"],
  ["mail", "email", "inbox"],
  ["build", "deploy"],
];
const fakeEmbedding = (text: string) =>
  VOCAB.map((words) => words.filter((w) => text.toLowerCase().includes(w)).length + 0.01);

type Server = { url: string; inputs: string[][]; close: () => Promise<void> };
const startServer = (opts: { status?: number; delayMs?: number } = {}) =>
  new Promise<Server>((resolve) => {
    const inputs: string[][] = [];
    const server = createServer((req, res) => {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        // oxlint-disable-next-line executor/no-json-parse -- boundary: raw HTTP body inside a node:http test fake
        const { input } = JSON.parse(body) as { input: string[] };
        inputs.push(input);
        setTimeout(() => {
          if (opts.status) {
            res.writeHead(opts.status).end();
            return;
          }
          res.writeHead(200, { "content-type": "application/json" });
          res.end(
            JSON.stringify({
              data: input.map((t, index) => ({ index, embedding: fakeEmbedding(t) })),
            }),
          );
        }, opts.delayMs ?? 0);
      });
    });
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        url: `http://127.0.0.1:${port}/v1/embeddings`,
        inputs,
        close: () =>
          new Promise((r) => {
            server.closeAllConnections();
            server.close(() => r());
          }),
      });
    });
  });

const tool = (address: string, name: string, integration: string, description: string) =>
  // oxlint-disable-next-line executor/no-double-cast -- boundary: search reads only these fields; a full Tool fixture adds nothing
  ({ address, name, integration, description }) as unknown as Tool;

const TOOLS = [
  tool("tools.slack.org.ws.users_search", "users_search", "slack", "Look up a person by name"),
  tool(
    "tools.proton_mail.org.ws.search_emails",
    "search_emails",
    "proton_mail",
    "Search your inbox",
  ),
  tool("tools.outsmartly.org.o.start_build", "start_build", "outsmartly", "Start a deploy build"),
];
const executor = { tools: { list: () => Effect.succeed(TOOLS) } } as never;
const paths = (r: { items: readonly { path: string }[] }) => r.items.map((i) => i.path);

const withEnv = (vars: Record<string, string | undefined>) => {
  for (const [k, v] of Object.entries(vars)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
};

afterEach(() => withEnv({ EXECUTOR_EMBED_URL: undefined, EXECUTOR_EMBED_TIMEOUT_MS: undefined }));

describe("embedding search", () => {
  it.effect("adds tools only the embedding finds, ranked above keyword-less noise", () =>
    Effect.gen(function* () {
      const keywordOnly = yield* searchTools(executor, "find my colleague", 5);
      expect(paths(keywordOnly)).not.toContain("slack.org.ws.users_search");

      const server = yield* Effect.promise(() => startServer());
      withEnv({ EXECUTOR_EMBED_URL: server.url });
      const hybrid = yield* searchTools(executor, "find my colleague", 5);
      expect(paths(hybrid)[0]).toBe("slack.org.ws.users_search");
      yield* Effect.promise(server.close);
    }),
  );

  it.effect("reuses cached catalogue embeddings on the next search", () =>
    Effect.gen(function* () {
      const server = yield* Effect.promise(() => startServer());
      withEnv({ EXECUTOR_EMBED_URL: server.url });
      yield* searchTools(executor, "email", 5);
      const firstCall = server.inputs.flat().length;
      yield* searchTools(executor, "deploy", 5);
      expect(firstCall).toBe(TOOLS.length + 1);
      expect(server.inputs.flat().length - firstCall).toBe(1);
      yield* Effect.promise(server.close);
    }),
  );

  // live clock: it.effect uses TestClock, under which the timeout never fires
  it.live("falls back to keyword ranking when the service errors or is slow", () =>
    Effect.gen(function* () {
      const expected = paths(yield* searchTools(executor, "search inbox", 5));

      const failing = yield* Effect.promise(() => startServer({ status: 500 }));
      withEnv({ EXECUTOR_EMBED_URL: failing.url });
      expect(paths(yield* searchTools(executor, "search inbox", 5))).toEqual(expected);
      yield* Effect.promise(failing.close);

      const slow = yield* Effect.promise(() => startServer({ delayMs: 300 }));
      withEnv({ EXECUTOR_EMBED_URL: slow.url, EXECUTOR_EMBED_TIMEOUT_MS: "50" });
      expect(paths(yield* searchTools(executor, "search inbox", 5))).toEqual(expected);
      yield* Effect.promise(slow.close);
    }),
  );

  it.effect("keeps embedding-only results inside the namespace", () =>
    Effect.gen(function* () {
      const server = yield* Effect.promise(() => startServer());
      withEnv({ EXECUTOR_EMBED_URL: server.url });
      const scoped = yield* searchTools(executor, "find my colleague", 5, {
        namespace: "proton_mail",
      });
      expect(paths(scoped).every((p) => p.startsWith("proton_mail."))).toBe(true);
      yield* Effect.promise(server.close);
    }),
  );

  it("weights keywords down for long queries and fuses by reciprocal rank", () => {
    expect(keywordWeight(3)).toBe(1);
    expect(keywordWeight(8)).toBe(0.75);
    expect(keywordWeight(20)).toBe(0.5);
    const fused = fuseRankings(["a", "b"], ["b", "c"], 1);
    expect([...fused].sort((x, y) => y[1] - x[1]).map(([p]) => p)).toEqual(["b", "a", "c"]);
  });
});
