"""Recover missed arXiv mailings using the existing Zotero ranking and mail layout."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, make_msgid
from html import escape
import json
import os
from pathlib import Path
import re
import smtplib
import ssl
from zoneinfo import ZoneInfo


def announcement_window(day: date) -> tuple[datetime, datetime]:
    """Normal arXiv announcement day in UTC/Asia; delayed moderation may differ."""
    if day.weekday() > 4:
        raise ValueError("There is no regular Saturday/Sunday UTC announcement.")
    deadline = day - timedelta(days=3 if day.weekday() == 0 else 1)
    previous = deadline - timedelta(days=3 if day.weekday() == 1 else 1)
    eastern = ZoneInfo("America/New_York")
    return tuple(
        datetime.combine(value, time(14), eastern).astimezone(timezone.utc)
        for value in (previous, deadline)
    )


def work_dates(start_text: str, end_text: str) -> list[str]:
    start, end = date.fromisoformat(start_text), date.fromisoformat(end_text)
    if end < start or (end - start).days > 30:
        raise ValueError("Use an ascending range of at most 31 calendar days.")
    if end > datetime.now(ZoneInfo("Asia/Singapore")).date():
        raise ValueError("Future announcement dates cannot be backfilled.")
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
        if (start + timedelta(days=offset)).weekday() < 5
    ]


def fetch_results(day: date):
    import arxiv

    start, end = announcement_window(day)
    categories = [value.strip() for value in os.environ["ARXIV_QUERY"].split("+")]
    if not categories or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]*", value) for value in categories
    ):
        raise ValueError("ARXIV_QUERY must contain category names joined with '+'.")
    category_query = " OR ".join(
        f"cat:{value} OR cat:{value}.*" for value in categories
    )
    query = (
        f"({category_query}) AND submittedDate:"
        f"[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    )
    client = arxiv.Client(page_size=100, num_retries=5, delay_seconds=5)
    search = arxiv.Search(
        query=query, max_results=5000,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )
    found = list(client.results(search))
    if len(found) >= 5000:
        raise RuntimeError("Historical query reached its result limit; nothing was sent.")
    # The API range is inclusive; make adjacent mailing windows strictly disjoint.
    unique = {}
    for result in found:
        selected_primary = any(
            result.primary_category == value
            or result.primary_category.startswith(value + ".")
            for value in categories
        )
        # Match the daily RSS's 'new' entries, excluding cross-list-only matches.
        if selected_primary and start <= result.published < end:
            identifier = re.sub(r"v\d+$", "", result.get_short_id())
            unique[identifier] = result
    return list(unique.values()), start, end


def prepare_papers(results, mode: str):
    from main import filter_corpus, get_zotero_corpus
    from paper import ArxivPaper
    from recommender import rerank_paper

    class AbstractPaper(ArxivPaper):
        @property
        def tldr(self):
            return escape(self.summary)

        @property
        def affiliations(self):
            return None

        @property
        def code_url(self):
            return None

    corpus = get_zotero_corpus(os.environ["ZOTERO_ID"], os.environ["ZOTERO_KEY"])
    if os.environ.get("ZOTERO_IGNORE"):
        corpus = filter_corpus(corpus, os.environ["ZOTERO_IGNORE"])
    if not corpus:
        raise RuntimeError("The current Zotero corpus has no eligible abstracts.")
    paper_type = ArxivPaper if mode == "full" else AbstractPaper
    papers = rerank_paper([paper_type(result) for result in results], corpus)
    maximum = int(os.environ.get("MAX_PAPER_NUM") or "100")
    if maximum != -1:
        if maximum < 1:
            raise ValueError("MAX_PAPER_NUM must be positive or -1.")
        papers = papers[:maximum]
    return papers


def configure_llm():
    from llm import set_global_llm

    use_api = os.environ.get("USE_LLM_API", "").lower() in ("true", "1", "yes", "y", "on")
    language = os.environ.get("LANGUAGE") or "English"
    if use_api:
        set_global_llm(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1",
            model=os.environ.get("MODEL_NAME") or "gpt-4o", lang=language,
        )
    else:
        set_global_llm(lang=language)


def send_backfill(day: date, html: str):
    sender, receiver = os.environ["SENDER"], os.environ["RECEIVER"]
    message = MIMEText(html, "html", "utf-8")
    message["From"] = formataddr(("GitHub Action", sender))
    message["To"] = receiver
    message["Subject"] = Header(f"[Backfill] Daily arXiv {day:%Y/%m/%d}", "utf-8").encode()
    message["Message-ID"] = make_msgid()
    message["X-Arxiv-Backfill-Date"] = day.isoformat()
    host, port = os.environ["SMTP_SERVER"], int(os.environ["SMTP_PORT"])
    context = ssl.create_default_context()
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, timeout=90, context=context)
    else:
        server = smtplib.SMTP(host, port, timeout=90)
    accepted = False
    try:
        if port != 465:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
        server.login(sender, os.environ["SENDER_PASSWORD"])
        refused = server.sendmail(sender, [receiver], message.as_string())
        if refused:
            raise RuntimeError("SMTP did not accept the configured recipient.")
        accepted = True
    finally:
        try:
            server.quit()
        except (OSError, smtplib.SMTPException):
            # A failed QUIT after acceptance must not turn a sent email into a retry.
            if not accepted:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs=2, metavar=("START", "END"))
    parser.add_argument("--day")
    parser.add_argument("--format", choices=("full", "abstract"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", default=os.environ.get("RUNNER_TEMP", "/tmp"))
    args = parser.parse_args()
    if args.dates:
        days = work_dates(*args.dates)
        if not days:
            parser.error("The date range has no regular announcement days.")
        print(json.dumps(days))
        return
    if not args.day:
        parser.error("Specify --day or --dates.")
    day = date.fromisoformat(args.day)
    results, start, end = fetch_results(day)
    report = {
        "day": day.isoformat(), "candidate_count": len(results),
        "submission_start_utc": start.isoformat(),
        "submission_end_exclusive_utc": end.isoformat(),
        "date_basis": "original submission window for the normal announcement schedule",
        "format": args.format,
    }
    if args.dry_run:
        report["status"] = "dry_run"
    elif not results:
        report["status"] = "no_papers"
    else:
        from construct_email import render_email

        papers = prepare_papers(results, args.format)
        if args.format == "full":
            configure_llm()
        note = (
            f"<h2>Backfill: {day:%Y/%m/%d}</h2>"
            f"<p>Ranked using the current Zotero library. "
            f"Original submission window: {start:%Y-%m-%d %H:%M} UTC "
            f"to {end:%Y-%m-%d %H:%M} UTC (end excluded). "
            "Grouped by the normal arXiv announcement schedule; moderation delays "
            "can change the actual announcement day.</p>"
        )
        if args.format == "abstract":
            note += "<p>TLDR fields contain the original abstracts for this fast backfill.</p>"
        html = render_email(papers).replace("<body>", "<body>" + note, 1)
        send_backfill(day, html)
        report.update(
            status="smtp_accepted", selected_count=len(papers),
            sent_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    output = Path(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{day.isoformat()}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
