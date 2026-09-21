"""The shell interface. stdout is evidence (or JSON); diagnostics go to stderr."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .index import Index
from .inputs import read_paths
from .judge import SemanticError, resolve_backend
from .select import select, select_per


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def inputs(parser):
    parser.add_argument(
        "paths", nargs="*", help="files/directories; '-' reads stdin; a .jselect file reuses an index"
    )
    parser.add_argument("--field", help="text field or dotted path; auto-detected for common JSON/CSV fields")
    parser.add_argument(
        "--group-by", help="combine rows by this field, e.g. conversation_id (in input order)"
    )
    parser.add_argument(
        "--per",
        metavar="FIELD",
        help="one context per value of this field, e.g. company_year; needs --json (one object per line)",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "text", "lines", "jsonl", "json", "csv", "tsv"],
        default="auto",
        help="input format (default: infer from extension)",
    )
    parser.add_argument("--glob", help="only read matching files in directories, e.g. '*.py'")
    parser.add_argument(
        "--exclude", action="append", default=[], help="gitignore-style exclusion; repeatable"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1800, help="passage size in characters (default: 1800)"
    )
    parser.add_argument("--overlap", type=int, default=240, help="overlap in characters (default: 240)")


def source(args):
    return read_paths(
        args.paths or ["-"],
        field=args.field,
        format=args.format,
        glob=args.glob,
        exclude=args.exclude,
        group_by=args.group_by,
        per=args.per,
    )


def parser_for(command):
    p = Parser(
        prog="jselect" + (f" {command}" if command else ""),
        description="Select useful, source-linked evidence within a token budget.",
        epilog="Other commands: index (save a searchable collection), inspect (index stats), "
        "show (read a passage by ID), doctor (check setup).",
    )
    p.add_argument("--version", action="version", version=f"jselect {__version__}")
    p.add_argument("--json", action="store_true", help="emit a versioned JSON object, including JSON errors")
    if command == "doctor":
        p.add_argument("--api", choices=["typesafe", "openrouter", "gateway"])
        return p
    if command in {"inspect", "show"}:
        p.add_argument("path", help="saved .jselect index")
        if command == "show":
            p.add_argument("id", help="full passage ID from JSON selection output")
        return p
    if command == "index":
        inputs(p)
        p.add_argument("--output", "-o", required=True, help="path for the saved .jselect index")
        p.add_argument("--force", action="store_true", help="replace an existing index atomically")
        return p
    p.add_argument("task", help="the task, question, or information you need")
    inputs(p)
    p.add_argument(
        "--tokens", type=int, default=8000, help="maximum tokens in the evidence context (default: 8000)"
    )
    p.add_argument(
        "--encoding", default="o200k_base", help="tiktoken encoding/model, or 'bytes' (default: o200k_base)"
    )
    p.add_argument(
        "--mode",
        choices=["auto", "local", "semantic"],
        default="auto",
        help="auto uses semantic scoring when a key is configured, else local retrieval",
    )
    p.add_argument(
        "--local", action="store_const", dest="mode", const="local", help="use local retrieval; no API calls"
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=256,
        help="maximum passages retained for selection; with --scan shortlist, caps scoring (default: 256)",
    )
    p.add_argument(
        "--scan",
        choices=["shortlist", "all"],
        help="all scores every eligible passage (default: all for semantic scoring; shortlist for local)",
    )
    p.add_argument("--against", help="previous --json result or records: seek additional evidence")
    p.add_argument(
        "--diversity", type=float, default=0.7, help="penalty for repetitive text, 0..1 (default: 0.7)"
    )
    p.add_argument(
        "--threshold",
        type=float,
        help="minimum relevance score (semantic default: 0.25; 0.5 with --sample representative)",
    )
    p.add_argument("-n", "--max-items", type=int, help="also cap the number of selected passages")
    p.add_argument(
        "--sample",
        choices=["representative"],
        help="draw relevant passages at random instead of favoring the most relevant and novel",
    )
    p.add_argument("--seed", type=int, default=0, help="random seed for --sample (default: 0)")
    p.add_argument("--api", choices=["typesafe", "openrouter", "gateway"])
    p.add_argument("--model", help="override the pinned Jev model")
    p.add_argument(
        "--budget",
        type=float,
        default=0.05,
        help="semantic dollar budget; preflight estimate (default: 0.05)",
    )
    p.add_argument(
        "-j", "--concurrency", type=int, default=8, help="maximum simultaneous requests (default: 8)"
    )
    p.add_argument("--batch-size", type=int, default=8, help="passages per API request, 1..16 (default: 8)")
    p.add_argument("--timeout", type=float, default=20, help="total seconds per request, including retries")
    p.add_argument("--no-cache", action="store_true", help="disable persistent semantic score caching")
    p.add_argument("--stats", action="store_true", help="print timing and cost to stderr")
    p.add_argument("--output", "-o", help="write evidence or JSON to this file instead of stdout")
    return p


def main(argv=None, *, out=None, err=None, transport=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out, err = out or sys.stdout, err or sys.stderr
    json_mode = "--json" in argv
    command = next((arg for arg in argv if arg != "--json"), "")
    command = command if command in {"index", "inspect", "show", "doctor"} else ""
    if command:
        argv.remove(command)
    try:
        args = parser_for(command).parse_intermixed_args(argv)
        if command == "doctor":
            issue = None
            try:
                backend = resolve_backend(args.api)
            except (OSError, ValueError) as e:
                backend, issue = None, str(e)
            db = sqlite3.connect(":memory:")
            try:
                db.execute("CREATE VIRTUAL TABLE health USING fts5(text)")
            finally:
                db.close()
            payload = {
                "schema_version": 1,
                "version": __version__,
                "local_ready": True,
                "semantic_configured": backend is not None,
                "auth_verified": False,
                "api": backend.name if backend else None,
                "model": backend.model if backend else None,
                "auth_source": backend.auth_source if backend else "missing",
                "hint": issue
                or (None if backend else "Set TYPESAFE_API_KEY or OPENROUTER_API_KEY for semantic scoring."),
            }
            print(
                json.dumps(payload)
                if args.json
                else f"jselect {__version__}: local mode ready; "
                f"semantic {'configured' if backend else 'not configured'}"
                + (f"\n{payload['hint']}" if payload["hint"] else ""),
                file=out,
            )
            return 0
        if command == "index":
            with Index.build(
                source(args),
                path=args.output,
                per=args.per,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                force=args.force,
            ) as index:
                payload = {"schema_version": 1, "path": str(index.path), "stats": index.stats}
            print(
                json.dumps(payload)
                if args.json
                else f"Indexed {payload['stats']['records']:,} records → {payload['path']}",
                file=out,
            )
            return 0
        if command in {"inspect", "show"}:
            with Index(args.path) as index:
                payload = (
                    {"schema_version": 1, "path": str(index.path), "stats": index.stats}
                    if command == "inspect"
                    else {"schema_version": 1, "passage": asdict(index.get(args.id))}
                )
            print(
                json.dumps(payload, ensure_ascii=False)
                if args.json or command == "inspect"
                else payload["passage"]["text"],
                file=out,
            )
            return 0
        if args.per and not args.json:
            raise ValueError("--per writes one JSON object per line; add --json")
        index = None
        if len(args.paths) == 1 and Path(args.paths[0]).suffix == ".jselect":
            index = Index(args.paths[0])
        try:
            result = (select_per if args.per else select)(
                index or source(args),
                **({"per": args.per} if args.per else {}),
                task=args.task,
                tokens=args.tokens,
                encoding=args.encoding,
                mode=args.mode,
                candidates=args.candidates,
                scan=args.scan,
                diversity=args.diversity,
                threshold=args.threshold,
                max_items=args.max_items,
                against=args.against,
                sample=args.sample,
                seed=args.seed,
                api=args.api,
                model=args.model,
                budget=args.budget,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                timeout=args.timeout,
                cache=not args.no_cache,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                _transport=transport,
            )
        finally:
            if index:
                index.close()
        if args.per:
            output = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in result)
        else:
            output = json.dumps(result.to_dict(), ensure_ascii=False) + "\n" if args.json else result.context
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            out.write(output)
        if args.stats and args.per:
            # Scan work is shared, so every line reports the same calls and cost; do not sum them.
            stats = result[-1].stats if result else {}
            print(
                f"jselect: {len(result)} contexts, one per {args.per}; "
                f"{sum(len(r.items) for r in result)} passages; {stats.get('mode')}; "
                f"{stats.get('calls', 0)} calls; ${stats.get('cost', 0):.6f}; {stats.get('seconds', 0):.3f}s",
                file=err,
            )
        elif args.stats:
            stats = result.stats
            print(
                f"jselect: {len(result.items)} passages, {result.tokens}/{result.token_budget} tokens; "
                f"{stats.get('mode')}; {stats.get('calls', 0)} calls; ${stats.get('cost', 0):.6f}; "
                f"{stats.get('seconds', 0):.3f}s",
                file=err,
            )
            if stats.get("sample"):
                print(
                    f"jselect: {stats['sample']} sample: {stats['sample_occurrences']} of "
                    f"{stats['population_occurrences']} relevant occurrences "
                    f"({stats['sample_passages']} of {stats['population_passages']} passages); "
                    f"threshold {stats['threshold']:g}; seed {stats['seed']}; random without replacement; "
                    f"ended by {stats['sample_stop']}",
                    file=err,
                )
        if not args.json:
            for warning in result.warnings:
                print(f"jselect: {warning}", file=err)
        return 0
    except BrokenPipeError:
        return 0
    except (ValueError, OSError, sqlite3.Error, SemanticError) as e:
        if json_mode:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": {"type": type(e).__name__, "message": str(e)},
                        "stats": getattr(e, "stats", None),
                    }
                ),
                file=out,
            )
        else:
            print(f"jselect: {e}", file=err)
        return 2
    except KeyboardInterrupt:
        print("jselect: interrupted", file=err)
        return 130


def cli():
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0) from None
