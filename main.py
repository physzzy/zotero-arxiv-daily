import argparse
import os
import sys
import time
from tempfile import mkstemp

import arxiv
import feedparser
from dotenv import load_dotenv
from gitignore_parser import parse_gitignore
from loguru import logger
from pyzotero import zotero
from tqdm import tqdm

from construct_email import render_email, send_email
from llm import set_global_llm
from paper import ArxivPaper
from recommender import rerank_paper

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_zotero_corpus(id: str, key: str) -> list[dict]:
    zot = zotero.Zotero(id, "user", key)
    collections = zot.everything(zot.collections())
    collections = {c["key"]: c for c in collections}
    corpus = zot.everything(
        zot.items(itemType="conferencePaper || journalArticle || preprint")
    )
    corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

    def get_collection_path(col_key: str) -> str:
        parent = collections[col_key]["data"]["parentCollection"]
        if parent:
            return get_collection_path(parent) + " / " + collections[col_key]["data"]["name"]
        return collections[col_key]["data"]["name"]

    for c in corpus:
        paths = [get_collection_path(col) for col in c["data"]["collections"]]
        c["paths"] = paths
    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    _, filename = mkstemp()
    with open(filename, "w") as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir="./")
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c["paths"]]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def parse_bool_env(value: str) -> bool:
    return value.lower() in ["true", "1", "yes", "y", "on"]


def fetch_rss_with_retry(query: str, max_attempts: int = 5, base_wait: int = 5):
    rss_url = f"https://rss.arxiv.org/atom/{query}"
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Fetching arXiv RSS: {rss_url} (attempt {attempt}/{max_attempts})")
            feed = feedparser.parse(rss_url)

            if getattr(feed, "bozo", 0):
                logger.warning(f"RSS parse bozo={feed.bozo}; continuing to inspect content")

            title = getattr(feed.feed, "title", "")
            if "Feed error for query" in title:
                raise Exception(f"Invalid ARXIV_QUERY: {query}.")

            if not getattr(feed, "entries", None):
                logger.warning("RSS returned no entries.")

            return feed

        except Exception as e:
            last_err = e
            wait = min(60, base_wait * (2 ** (attempt - 1)))
            logger.warning(f"RSS fetch failed: {e}. Retrying in {wait}s...")
            if attempt < max_attempts:
                time.sleep(wait)

    raise RuntimeError(f"Failed to fetch arXiv RSS after {max_attempts} attempts: {last_err}")


def fetch_batch_with_retry(
    client: arxiv.Client,
    id_batch: list[str],
    max_attempts: int = 6,
    base_wait: int = 10,
) -> list[ArxivPaper]:
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            search = arxiv.Search(id_list=id_batch)
            return [ArxivPaper(p) for p in client.results(search)]

        except arxiv.HTTPError as e:
            last_err = e
            wait = min(180, base_wait * (2 ** (attempt - 1)))
            logger.warning(
                f"arXiv batch request failed with HTTP error: {e}. "
                f"Attempt {attempt}/{max_attempts}, retrying in {wait}s..."
            )
            if attempt < max_attempts:
                time.sleep(wait)

        except Exception as e:
            last_err = e
            wait = min(180, base_wait * (2 ** (attempt - 1)))
            logger.warning(
                f"arXiv batch request failed: {e}. "
                f"Attempt {attempt}/{max_attempts}, retrying in {wait}s..."
            )
            if attempt < max_attempts:
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch arXiv batch after {max_attempts} attempts. "
        f"batch_size={len(id_batch)}, last_error={last_err}"
    )


def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    # 官方要求 legacy API 最多约每 3 秒 1 次请求；这里放宽一点
    client = arxiv.Client(
        num_retries=3,
        delay_seconds=5.0,
        page_size=100,
    )

    feed = fetch_rss_with_retry(query)

    if debug:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(
            query="cat:cs.AI",
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break
        return papers

    all_paper_ids = [
        i.id.removeprefix("oai:arXiv.org:")
        for i in feed.entries
        if getattr(i, "arxiv_announce_type", None) == "new"
    ]

    if len(all_paper_ids) == 0:
        logger.info("No 'new' arXiv papers found in RSS feed.")
        return []

    logger.info(f"Found {len(all_paper_ids)} new paper ids from RSS.")

    # RSS 和 API 都属于 legacy API，切换前先停一下，减少 429 概率
    time.sleep(5)

    papers = []
    failed_batches = 0
    batch_size = 20  # 比原来的 50 更保守，降低被限流概率

    bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv papers")
    for start in range(0, len(all_paper_ids), batch_size):
        id_batch = all_paper_ids[start : start + batch_size]
        target_size = len(id_batch)

        try:
            batch = fetch_batch_with_retry(client, id_batch)
            papers.extend(batch)
            bar.update(target_size)

            # 两个 batch 之间再额外停一下，进一步减小限流概率
            time.sleep(3)

        except Exception as e:
            failed_batches += 1
            logger.error(f"Skipping failed batch starting at index {start}: {e}")
            bar.update(target_size)

            # 失败后也停一下，避免马上继续撞限流
            time.sleep(15)

    bar.close()

    if len(papers) == 0 and len(all_paper_ids) > 0:
        raise RuntimeError(
            f"Failed to retrieve any arXiv paper metadata. "
            f"total_ids={len(all_paper_ids)}, failed_batches={failed_batches}"
        )

    if failed_batches > 0:
        logger.warning(
            f"Completed with partial failures: failed_batches={failed_batches}, "
            f"retrieved_papers={len(papers)}"
        )
    else:
        logger.info(f"Successfully retrieved {len(papers)} arXiv papers.")

    return papers


parser = argparse.ArgumentParser(description="Recommender system for academic papers")


def add_argument(*args, **kwargs):
    def get_env(key: str, default=None):
        v = os.environ.get(key)
        if v == "" or v is None:
            return default
        return v

    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get("dest", args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)

    if env_value is not None:
        if kwargs.get("type") == bool:
            env_value = parse_bool_env(env_value)
        elif kwargs.get("type") is not None:
            env_value = kwargs.get("type")(env_value)
        parser.set_defaults(**{arg_full_name: env_value})


if __name__ == "__main__":
    add_argument("--zotero_id", type=str, help="Zotero user ID")
    add_argument("--zotero_key", type=str, help="Zotero API key")
    add_argument(
        "--zotero_ignore",
        type=str,
        help="Zotero collection to ignore, using gitignore-style pattern.",
    )
    add_argument(
        "--send_empty",
        type=bool,
        help="If get no arxiv paper, send empty email",
        default=False,
    )
    add_argument(
        "--max_paper_num",
        type=int,
        help="Maximum number of papers to recommend",
        default=100,
    )
    add_argument("--arxiv_query", type=str, help="Arxiv search query")
    add_argument("--smtp_server", type=str, help="SMTP server")
    add_argument("--smtp_port", type=int, help="SMTP port")
    add_argument("--sender", type=str, help="Sender email address")
    add_argument("--receiver", type=str, help="Receiver email address")
    add_argument("--sender_password", type=str, help="Sender email password")
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    assert (not args.use_llm_api or args.openai_api_key is not None)

    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")

    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n{args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")

    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)

    if len(papers) == 0:
        logger.info(
            "No new papers found. Yesterday maybe a holiday and no one submit their work :). "
            "If this is not the case, please check the ARXIV_QUERY."
        )
        if not args.send_empty:
            sys.exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[: args.max_paper_num]

        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(
                api_key=args.openai_api_key,
                base_url=args.openai_api_base,
                model=args.model_name,
                lang=args.language,
            )
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers)
    logger.info("Sending email...")
    send_email(
        args.sender,
        args.receiver,
        args.sender_password,
        args.smtp_server,
        args.smtp_port,
        html,
    )
    logger.success(
        "Email sent successfully! If you don't receive the email, "
        "please check the configuration and the junk box."
    )
