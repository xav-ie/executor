// Optional dense retrieval for `tools.search`, fused with the keyword ranking.
//
// Enabled by EXECUTOR_EMBED_URL, an OpenAI-compatible `/v1/embeddings`
// endpoint (`{ input: string[] }` -> `{ data: [{ index, embedding }] }`).
// Unset, or on any error/timeout, search stays keyword-only.
import { Data, Effect, Option, Schema } from "effect";

const QUERY_INSTRUCTION =
  "Instruct: Given a user request, retrieve the tool that can fulfill it\nQuery: ";
const BATCH_SIZE = 32;
/** Budget for one search (catalogue fill + query). */
const DEFAULT_SEARCH_TIMEOUT_MS = 1_500;
/** Embedding-only candidates beyond this cosine rank are dropped, so `total`
 *  stays a count of plausible matches instead of the whole catalogue. */
export const EMBEDDING_CANDIDATES = 50;
const RRF_K = 60;

export type EmbeddableTool = {
  readonly path: string;
  readonly name: string;
  readonly description?: string;
};

class EmbeddingError extends Data.TaggedError("EmbeddingError")<{
  readonly reason: string;
  readonly cause?: unknown;
}> {}

const EmbeddingsResponse = Schema.Struct({
  data: Schema.Array(
    Schema.Struct({ index: Schema.Number, embedding: Schema.Array(Schema.Number) }),
  ),
});

const env = (name: string): string | undefined =>
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.[name];

export const docText = (tool: EmbeddableTool): string =>
  `${tool.path}\n${tool.name.replaceAll("_", " ")}\n${(tool.description ?? "").slice(0, 2000)}`;

// ponytail: keyed by url + doc text and never evicted; entries for removed or
// edited tools linger until restart. Fine at hundreds of tools. A cold
// catalogue that outlasts one search's budget fills across searches, since
// each finished batch stays cached.
const cache = new Map<string, Float32Array>();

let lastWarn = 0;
const warnOncePerMinute = (message: string) =>
  Effect.suspend(() => {
    const now = Date.now();
    if (now - lastWarn < 60_000) return Effect.void;
    lastWarn = now;
    return Effect.logWarning(`embedding search disabled for this call: ${message}`);
  });

const normalize = (values: readonly number[]): Float32Array => {
  const out = Float32Array.from(values);
  const norm = Math.sqrt(out.reduce((sum, v) => sum + v * v, 0)) || 1;
  return out.map((v) => v / norm);
};

const embed = (url: string, input: readonly string[]) =>
  Effect.gen(function* () {
    const res = yield* Effect.tryPromise({
      try: (signal) =>
        fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ input }),
          signal,
        }),
      catch: (cause) => new EmbeddingError({ reason: "request failed", cause }),
    });
    if (!res.ok) return yield* new EmbeddingError({ reason: `service returned ${res.status}` });
    const json = yield* Effect.tryPromise({
      try: () => res.json(),
      catch: (cause) => new EmbeddingError({ reason: "invalid JSON", cause }),
    });
    const body = yield* Schema.decodeUnknownEffect(EmbeddingsResponse)(json).pipe(
      Effect.mapError((cause) => new EmbeddingError({ reason: "unexpected response", cause })),
    );
    const vectors: Float32Array[] = [];
    for (const row of body.data) vectors[row.index] = normalize(row.embedding);
    if (vectors.length !== input.length || vectors.includes(undefined as never)) {
      return yield* new EmbeddingError({ reason: "wrong number of vectors" });
    }
    return vectors;
  });

const fillCache = (url: string, texts: readonly string[]) => {
  const missing = [...new Set(texts)].filter((t) => !cache.has(url + t));
  const batches: string[][] = [];
  for (let i = 0; i < missing.length; i += BATCH_SIZE)
    batches.push(missing.slice(i, i + BATCH_SIZE));
  return Effect.forEach(
    batches,
    (batch) =>
      embed(url, batch).pipe(
        Effect.tap((vectors) =>
          Effect.sync(() => batch.forEach((t, j) => cache.set(url + t, vectors[j]!))),
        ),
      ),
    { concurrency: 4, discard: true },
  );
};

/** Tool paths ordered by cosine similarity to the query (best first), or
 *  `null` when embeddings are disabled or unavailable for this call. Never fails. */
export const embeddingRanking = (
  tools: readonly EmbeddableTool[],
  query: string,
): Effect.Effect<string[] | null> => {
  const url = env("EXECUTOR_EMBED_URL");
  if (!url || tools.length === 0) return Effect.succeed(null);
  const timeoutMs = Number(env("EXECUTOR_EMBED_TIMEOUT_MS")) || DEFAULT_SEARCH_TIMEOUT_MS;
  const texts = tools.map(docText);
  return Effect.all([embed(url, [QUERY_INSTRUCTION + query]), fillCache(url, texts)], {
    concurrency: 2,
  }).pipe(
    Effect.timeoutOption(timeoutMs),
    Effect.flatMap(
      Option.match({
        onNone: () => warnOncePerMinute("timed out").pipe(Effect.as(null)),
        onSome: ([[q]]) =>
          Effect.succeed(
            tools
              .map((tool, i) => {
                const d = cache.get(url + texts[i]!)!;
                let dot = 0;
                for (let j = 0; j < d.length; j++) dot += d[j]! * q![j]!;
                return { path: tool.path, dot };
              })
              .sort((a, b) => b.dot - a.dot || a.path.localeCompare(b.path))
              .map((r) => r.path),
          ),
      }),
    ),
    Effect.catch((error) => warnOncePerMinute(error.reason).pipe(Effect.as(null))),
  );
};

/** Keyword weight in the fusion: full for terse queries, halved for sentences,
 *  where token overlap is mostly noise. Linear from 4 to 12 query tokens. */
export const keywordWeight = (queryTokens: number): number =>
  Math.min(1, Math.max(0.5, 1 - ((queryTokens - 4) / 8) * 0.5));

/** Weighted reciprocal rank fusion. Returns path -> fused score. */
export const fuseRankings = (
  keyword: readonly string[],
  dense: readonly string[],
  keywordW: number,
): Map<string, number> => {
  const scores = new Map<string, number>();
  keyword.forEach((p, r) => scores.set(p, (scores.get(p) ?? 0) + keywordW / (RRF_K + r + 1)));
  dense
    .slice(0, EMBEDDING_CANDIDATES)
    .forEach((p, r) => scores.set(p, (scores.get(p) ?? 0) + 1 / (RRF_K + r + 1)));
  return scores;
};
