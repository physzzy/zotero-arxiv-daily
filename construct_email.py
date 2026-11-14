"""Timed TLS delivery and receipts; SMTP acceptance is recorded before QUIT."""

import json
import io
import os
from pathlib import Path
import smtplib
import ssl
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from zoneinfo import ZoneInfo
import urllib.request
import urllib.parse
import zipfile

BEIJING = ZoneInfo("Asia/Shanghai")


def already_sent(delivery_date):
    """Read earlier receipts under the existing Actions token; fail closed."""
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository:
        return False

    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if urllib.parse.urlparse(newurl).scheme != "https":
                raise RuntimeError("Unsafe artifact redirect")
            result = super().redirect_request(req, fp, code, msg, headers, newurl)
            result.remove_header("Authorization")
            return result

    opener = urllib.request.build_opener(SafeRedirect())

    def read(url, limit):
        if urllib.parse.urlparse(url).hostname != "api.github.com":
            raise RuntimeError("Unexpected GitHub API destination")
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        with opener.open(request, timeout=30) as response:
            data = response.read(limit + 1)
        if len(data) > limit:
            raise RuntimeError("Artifact exceeds receipt size limit")
        return data

    prefix = f"arxiv-delivery-{delivery_date}-"
    url = f"https://api.github.com/repos/{repository}/actions/artifacts?per_page=100"
    for artifact in json.loads(read(url, 2_000_000))["artifacts"]:
        if not artifact["name"].startswith(prefix) or artifact["expired"]:
            continue
        with zipfile.ZipFile(io.BytesIO(read(artifact["archive_download_url"], 1_000_000))) as archive:
            info = archive.getinfo("delivery.json")
            if info.file_size > 100_000:
                raise RuntimeError("Invalid receipt size")
            prior = json.loads(archive.read(info))
        if prior.get("status") == "smtp_accepted" and prior.get("date") == delivery_date:
            return True
    return False


def receipt(status, *, delivery_date, count=0, **details):
    result = dict(status=status, date=delivery_date, paper_count=count,
                  recorded_at=datetime.now(timezone.utc).isoformat(), **details)
    path = Path(os.environ.get("DELIVERY_RECEIPT", "_maintenance/delivery.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("DELIVERY_RECEIPT " + json.dumps(result, ensure_ascii=False), flush=True)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"status={status}\ndate={delivery_date}\n")
    return result


def wait_until_delivery_time(delivery_date, *, now=None, sleep=time.sleep):
    timed = os.environ.get("GITHUB_EVENT_NAME") == "schedule" or os.environ.get("DELIVERY_MODE") == "afternoon"
    if not timed:
        return
    now = now or datetime.now(timezone.utc)
    target = datetime.fromisoformat(delivery_date).replace(hour=15, tzinfo=BEIJING)
    delay = max(0, (target - now).total_seconds())
    if delay:
        print(f"Digest ready; waiting {delay:.0f}s until 15:00 Beijing", flush=True)
        sleep(delay)


def send_email(config, html, *, delivery_date, count, smtp_factory=None, ssl_factory=None):
    wait_until_delivery_time(delivery_date)
    message = EmailMessage()
    message["From"] = config.email.sender
    message["To"] = config.email.receiver
    message["Subject"] = f"Daily arXiv {delivery_date.replace('-', '/')}"
    message["Date"] = format_datetime(datetime.now(BEIJING))
    message["Message-ID"] = make_msgid()
    message.set_content("Your daily arXiv digest is included as HTML.")
    message.add_alternative(html, subtype="html")
    smtp_factory = smtp_factory or smtplib.SMTP
    ssl_factory = ssl_factory or smtplib.SMTP_SSL
    context = ssl.create_default_context()
    server = None
    accepted = False
    try:
        port = int(config.email.smtp_port)
        if port == 465:
            server = ssl_factory(config.email.smtp_server, port, timeout=60, context=context)
        else:
            server = smtp_factory(config.email.smtp_server, port, timeout=60)
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
        server.login(config.email.sender, config.email.sender_password)
        refused = server.send_message(message, config.email.sender, [config.email.receiver])
        if refused:
            raise RuntimeError("SMTP refused the recipient")
        accepted = True
        # No resend after the server has acknowledged DATA, even if QUIT fails.
        result = receipt("smtp_accepted", delivery_date=delivery_date, count=count,
                         message_id=message["Message-ID"])
        return result
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                try:
                    server.close()
                except Exception:
                    pass
                if accepted:
                    print("SMTP accepted the message; connection cleanup failed", flush=True)
