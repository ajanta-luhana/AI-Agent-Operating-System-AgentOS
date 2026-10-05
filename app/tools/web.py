"""Web search + fetch. Fetched content is UNTRUSTED data (prompt-injection risk)."""
import ipaddress, socket
from urllib.parse import urlparse
import httpx
from app.config import settings

UA = {"User-Agent": "Mozilla/5.0 (AgentOS research bot)"}


def search(query: str, n: int = 5) -> list[dict]:
    if settings.searxng_url:
        r = httpx.get(f"{settings.searxng_url}/search", params={"q": query, "format": "json"}, timeout=20)
        return [{"title": x["title"], "url": x["url"]} for x in r.json().get("results", [])[:n]]
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    return [{"title": x.get("title", ""), "url": x["href"]} for x in DDGS().text(query, max_results=n)]


def _safe_url(url: str) -> bool:
    """Basic SSRF guard: http(s) only, no private/loopback/link-local targets."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except socket.gaierror:
        return False
    return True


def fetch_text(url: str, limit: int = 20000) -> str:
    if not _safe_url(url):
        return ""
    try:
        r = httpx.get(url, headers=UA, timeout=15, follow_redirects=True)
        if "text/html" not in r.headers.get("content-type", "") or len(r.content) > 2_000_000:
            return ""
        import trafilatura
        return (trafilatura.extract(r.text) or "")[:limit]
    except Exception:
        return ""
