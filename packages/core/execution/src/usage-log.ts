import { Context, Effect } from "effect";
import * as Exit from "effect/Exit";
import { isToolResult } from "@executor-js/sdk/core";
import type { SandboxToolInvoker } from "@executor-js/codemode-core";

// Append-only JSONL record of what agents search for and which tools they call
// next, written to `EXECUTOR_USAGE_LOG` when set. It exists to give search
// evals and embedding fine-tunes real (query -> tool used) labels. Arguments
// and results are never recorded: only paths, scores, outcome and timing.

const SEARCH_RESULTS_LOGGED = 10;
const ERROR_REPORT_INTERVAL_MS = 60_000;

/** MCP session the current execution belongs to. A search made in one
 *  `execute` call is usually followed by tool calls in a later one, so offline
 *  analysis joins them on this id. */
export const CurrentUsageSession = Context.Reference<string | undefined>(
  "@executor-js/execution/CurrentUsageSession",
  { defaultValue: () => undefined },
);

const usageLogPath = (): string | undefined =>
  typeof process === "undefined" ? undefined : process.env.EXECUTOR_USAGE_LOG || undefined;

let pending: string[] = [];
let draining: Promise<void> | undefined;
let lastErrorReportAt = 0;

const drain = async (path: string): Promise<void> => {
  const { appendFile } = await import("node:fs/promises");
  while (pending.length > 0) {
    const batch = pending.join("");
    pending = [];
    await appendFile(path, batch);
  }
};

/** Queue one event; never throws and never blocks the caller. */
export const recordUsage = (event: Record<string, unknown>): void => {
  const path = usageLogPath();
  if (!path) return;
  pending.push(`${JSON.stringify({ ts: new Date().toISOString(), ...event })}\n`);
  if (draining) return;
  draining = Effect.runPromise(
    Effect.tryPromise({ try: () => drain(path), catch: (cause) => cause }).pipe(
      Effect.catchCause((cause) =>
        Effect.sync(() => {
          pending = [];
          if (Date.now() - lastErrorReportAt < ERROR_REPORT_INTERVAL_MS) return;
          lastErrorReportAt = Date.now();
          console.error(`executor usage log: write to ${path} failed`, cause);
        }),
      ),
      Effect.ensuring(
        Effect.sync(() => {
          draining = undefined;
        }),
      ),
    ),
  );
};

/** Resolves once every queued event has been written (tests, shutdown). */
export const flushUsageLog = async (): Promise<void> => {
  while (draining) await draining;
};

const searchResults = (value: unknown): readonly { path: string; score: number }[] => {
  const items = (value as { readonly items?: unknown } | null)?.items;
  if (!Array.isArray(items)) return [];
  return items.slice(0, SEARCH_RESULTS_LOGGED).map((item: { path: string; score: number }) => ({
    path: item.path,
    score: item.score,
  }));
};

const usageEvent = (
  input: { readonly path: string; readonly args: unknown },
  exit: Exit.Exit<unknown, unknown>,
  durationMs: number,
): Record<string, unknown> => {
  const args = (typeof input.args === "object" && input.args !== null ? input.args : {}) as Record<
    string,
    unknown
  >;
  if (input.path === "search") {
    return {
      type: "search",
      query: args.query ?? "",
      namespace: args.namespace,
      limit: args.limit,
      offset: args.offset,
      ok: Exit.isSuccess(exit),
      results: Exit.isSuccess(exit) ? searchResults(exit.value) : [],
    };
  }
  if (input.path === "describe.tool") {
    return { type: "describe", path: args.path, ok: Exit.isSuccess(exit) };
  }
  const value = Exit.isSuccess(exit) ? exit.value : undefined;
  const failed = isToolResult(value) && !value.ok ? value.error : undefined;
  return {
    type: "call",
    path: input.path,
    ok: Exit.isSuccess(exit) && failed === undefined,
    durationMs,
    ...(failed ? { errorCode: failed.code } : {}),
  };
};

/**
 * Record every search, describe and tool call an `execute` run makes.
 * `executionId` groups the events of one run; `sessionId` groups runs.
 * A no-op wrapper when `EXECUTOR_USAGE_LOG` is unset.
 */
export const withUsageLog = (inner: SandboxToolInvoker): SandboxToolInvoker => {
  if (!usageLogPath()) return inner;
  const executionId = crypto.randomUUID();
  return {
    invoke: (input) => {
      const startedAt = Date.now();
      return inner.invoke(input).pipe(
        Effect.onExit((exit) =>
          Effect.gen(function* () {
            const sessionId = yield* CurrentUsageSession;
            recordUsage({
              executionId,
              sessionId,
              ...usageEvent(input, exit, Date.now() - startedAt),
            });
          }),
        ),
      );
    },
  };
};
