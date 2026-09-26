import { afterEach, describe, expect, it } from "@effect/vitest";
import { Effect } from "effect";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ToolResult } from "@executor-js/sdk";
import type { SandboxToolInvoker } from "@executor-js/codemode-core";

import { CurrentUsageSession, flushUsageLog, withUsageLog } from "./usage-log";

const inner: SandboxToolInvoker = {
  invoke: ({ path }) =>
    path === "search"
      ? Effect.succeed({ items: [{ path: "slack.org.w.users_search", score: 42, name: "x" }] })
      : path === "gmail.org.w.send"
        ? Effect.succeed(ToolResult.fail({ code: "oauth_reauth_required", message: "nope" }))
        : Effect.succeed(ToolResult.ok({ secret: "never logged" })),
};

const readEvents = (path: string) =>
  readFileSync(path, "utf8")
    .trim()
    .split("\n")
    // oxlint-disable-next-line executor/no-json-parse -- test reads back the raw JSONL file it asserts on
    .map((line) => JSON.parse(line));

describe("usage log", () => {
  afterEach(() => {
    delete process.env.EXECUTOR_USAGE_LOG;
  });

  it.effect("records searches and calls without args or results", () =>
    Effect.gen(function* () {
      const file = join(mkdtempSync(join(tmpdir(), "usage-log-")), "usage.jsonl");
      process.env.EXECUTOR_USAGE_LOG = file;
      const invoker = withUsageLog(inner);

      yield* invoker.invoke({ path: "search", args: { query: "find user", namespace: "slack" } });
      yield* invoker.invoke({ path: "slack.org.w.users_search", args: { query: "private" } });
      yield* invoker.invoke({ path: "gmail.org.w.send", args: {} });
      yield* Effect.promise(flushUsageLog);

      const [search, call, failed] = readEvents(file);
      expect(search).toMatchObject({
        type: "search",
        sessionId: "sess-1",
        query: "find user",
        namespace: "slack",
        results: [{ path: "slack.org.w.users_search", score: 42 }],
      });
      expect(call).toMatchObject({ type: "call", path: "slack.org.w.users_search", ok: true });
      expect(call.executionId).toBe(search.executionId);
      expect(failed).toMatchObject({ ok: false, errorCode: "oauth_reauth_required" });
      expect(readFileSync(file, "utf8")).not.toContain("private");
      expect(readFileSync(file, "utf8")).not.toContain("never logged");
    }).pipe(Effect.provideService(CurrentUsageSession, "sess-1")),
  );

  it.effect("is a no-op when EXECUTOR_USAGE_LOG is unset", () =>
    Effect.gen(function* () {
      const file = join(mkdtempSync(join(tmpdir(), "usage-log-")), "usage.jsonl");
      const invoker = withUsageLog(inner);
      expect(invoker).toBe(inner);
      yield* invoker.invoke({ path: "search", args: { query: "x" } });
      yield* Effect.promise(flushUsageLog);
      expect(existsSync(file)).toBe(false);
    }),
  );
});
