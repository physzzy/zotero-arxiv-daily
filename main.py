"""Run the pinned upstream core with existing repository settings and local AI."""

import argparse
from datetime import datetime
from pathlib import Path
import os
import sys
import tempfile
import time

from construct_email import BEIJING, already_sent, receipt, send_email
from llm import LocalChatClient

UPSTREAM_REVISION = "35c23ec44617b7595b30ac8e5bf4326c6cfd2336"


def build_config(env=None):
    from omegaconf import OmegaConf

    env = os.environ if env is None else env
    required = ["ZOTERO_ID", "ZOTERO_KEY", "ARXIV_QUERY", "SMTP_SERVER",
                "SMTP_PORT", "SENDER", "RECEIVER", "SENDER_PASSWORD"]
    missing = [key for key in required if not env.get(key, "").strip()]
    if missing:
        raise ValueError("Missing configuration names: " + ", ".join(missing))
    limit = int(env.get("MAX_PAPER_NUM") or 100)
    if limit == -1:
        limit = sys.maxsize
    if limit <= 0:
        raise ValueError("MAX_PAPER_NUM must be -1 or a positive integer")
    return OmegaConf.create({
        "zotero": {"user_id": env["ZOTERO_ID"], "api_key": env["ZOTERO_KEY"],
                   "include_path": None, "ignore_path": None},
        "source": {"arxiv": {"category": env["ARXIV_QUERY"].split("+"), "include_cross_list": False}},
        "email": dict(sender=env["SENDER"], receiver=env["RECEIVER"],
                      smtp_server=env["SMTP_SERVER"], smtp_port=int(env["SMTP_PORT"]),
                      sender_password=env["SENDER_PASSWORD"]),
        "llm": {"api": {"key": "local-only", "base_url": "http://127.0.0.1:1/v1"},
                "api_mode": "chat_completion", "generation_kwargs": {"model": "local-qwen", "max_tokens": 512},
                "language": env.get("LANGUAGE") or "English"},
        "reranker": {"local": {"model": "avsolatorio/GIST-small-Embedding-v0", "encode_kwargs": {}}},
        "executor": {"debug": False, "send_empty": (env.get("SEND_EMPTY") or "").lower() in {"true", "1", "yes", "y", "on"},
                     "max_paper_num": limit, "source": ["arxiv"], "reranker": "local"},
    })


def filter_corpus(corpus, patterns):
    if not patterns:
        return corpus
    from gitignore_parser import parse_gitignore

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ignore") as stream:
        stream.write(patterns)
        stream.flush()
        matcher = parse_gitignore(stream.name, base_dir="./")
        # Preserve the legacy collection separator and gitignore negation rules.
        return [paper for paper in corpus if not any(matcher(path.replace("/", " / ")) for path in paper.paths)]


def fetch_corpus(executor, *, sleep=time.sleep):
    from loguru import logger

    for attempt in range(3):
        try:
            corpus = executor.fetch_zotero_corpus()
            corpus = filter_corpus(corpus, os.environ.get("ZOTERO_IGNORE", ""))
            if not corpus:
                raise ValueError("No usable Zotero abstracts after filtering")
            return corpus
        except Exception as exc:
            # Report only type/status, never the Zotero configuration or key.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = status in {429, 500, 502, 503, 504} or type(exc).__name__ in {"ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout"}
            if not transient or attempt == 2:
                raise RuntimeError(f"Zotero retrieval failed ({type(exc).__name__}, HTTP {status})") from None
            logger.warning("Transient Zotero failure; retrying in {}s", 15 * (attempt + 1))
            sleep(15 * (attempt + 1))


def check_runtime(executor, client, corpus, delivery_date):
    """Exercise fresh Zotero, RSS, embeddings, local AI and HTML; never SMTP."""
    from zotero_arxiv_daily.protocol import Paper
    from zotero_arxiv_daily.construct_email import render_email

    raw = executor.retrievers["arxiv"]._retrieve_raw_papers()
    paper = Paper(source="arxiv", title="Local model compatibility check",
                  authors=["Test Author"], abstract="We study electron transport in a two-dimensional tight-binding model and find an edge-localized conducting mode.",
                  full_text="Test Author, Department of Physics, Test University. We study a tight-binding model of electron transport.",
                  url="https://example.org/local-model-check", pdf_url="https://example.org/local-model-check.pdf")
    paper = executor.reranker.rerank([paper], corpus)[0]
    paper.generate_tldr(client, executor.config.llm)
    paper.generate_affiliations(client, executor.config.llm)
    if client.calls < 2 or client.failures or paper.tldr == paper.abstract:
        raise RuntimeError("Local AI compatibility check failed")
    if not paper.tldr or "TLDR:" not in render_email([paper]):
        raise RuntimeError("HTML compatibility check failed")
    return receipt("dry_run_ok", delivery_date=delivery_date, count=1,
                   zotero_count=len(corpus), rss_paper_count=len(raw), local_ai_calls=client.calls,
                   upstream_revision=UPSTREAM_REVISION, smtp_attempted=False)


def run(check=False):
    from loguru import logger
    from zotero_arxiv_daily.executor import Executor
    from zotero_arxiv_daily.construct_email import render_email

    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=False, diagnose=False)
    delivery_date = os.environ.get("DELIVERY_DATE") or datetime.now(BEIJING).date().isoformat()
    if delivery_date != datetime.now(BEIJING).date().isoformat():
        raise ValueError("Only the current Beijing date is supported; historical sending is disabled")
    if not check and already_sent(delivery_date):
        return receipt("already_sent", delivery_date=delivery_date)
    config = build_config()
    executor = Executor(config)
    client = LocalChatClient()
    executor.openai_client = client
    logger.info("Upstream revision: {}; free local Qwen summaries", UPSTREAM_REVISION)
    corpus = fetch_corpus(executor)
    if check:
        return check_runtime(executor, client, corpus, delivery_date)
    papers = executor.retrievers["arxiv"].retrieve_papers()
    if not papers and not config.executor.send_empty:
        return receipt("no_papers", delivery_date=delivery_date, zotero_count=len(corpus))
    if papers:
        papers = executor.reranker.rerank(papers, corpus)[:config.executor.max_paper_num]
        for index, paper in enumerate(papers, 1):
            paper.generate_tldr(client, config.llm)
            paper.generate_affiliations(client, config.llm)
            logger.info("AI summary {}/{} complete", index, len(papers))
        if client.failures:
            raise RuntimeError("Local model failed; digest was not sent")
    return send_email(config, render_email(papers), delivery_date=delivery_date, count=len(papers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate runtime without sending email")
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        run(check=args.check)
    except Exception as exc:
        receipt("failed", delivery_date=datetime.now(BEIJING).date().isoformat(), error_type=type(exc).__name__)
        print(f"Digest failed: {exc}", file=sys.stderr)
        sys.exit(1)
