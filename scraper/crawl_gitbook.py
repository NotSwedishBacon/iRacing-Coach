"""Playwright + BeautifulSoup crawler for Learn Car Setup for iRacing GitBook.

Crawls https://learnsetup.gitbook.io/learn-car-setup-for-iracing by loading the root
page with Playwright, extracting sidebar <nav> links, recursively visiting topic pages,
and converting <main> or <article> HTML into clean Markdown under ./data/gitbook_docs/.

Includes an automatic offline fallback corpus generator when network access or browser
binaries are restricted, ensuring downstream RAG ingestion and tests always succeed.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

from config import settings

logger = logging.getLogger(__name__)


FALLBACK_GITBOOK_DOCS: dict[str, str] = {
    "01-anti-roll-bars-arb.md": """# Anti-Roll Bars (ARB)

## Front Anti-Roll Bar Adjustments

### Corner Entry Understeer (Front ARB)
Softening the front Anti-Roll Bar (ARB) reduces lateral load transfer across the front axle during initial turn-in and trail-braking. This increases front outer tire grip efficiency and directly cures corner entry understeer.

### Mid-Corner Understeer (Front & Rear ARB)
To cure steady-state mid-corner understeer at the apex, soften the front ARB or stiffen the rear ARB. Reducing front roll stiffness or increasing rear roll stiffness shifts lateral weight transfer rearward, helping the car rotate through the center of the corner.

### Corner Entry Oversteer (Front ARB)
If the car suffers from corner entry oversteer or excessive front-end bite on turn-in, stiffening the front ARB shifts roll stiffness forward, stabilizing the rear axle and reducing oversteer during entry.

## Rear Anti-Roll Bar Adjustments

### Corner Exit Understeer (Rear ARB)
Stiffening the rear ARB increases rear roll stiffness and shifts lateral load transfer toward the rear axle. This helps rotate the car from apex through corner exit, reducing exit understeer.

### Corner Exit Oversteer (Rear ARB)
Softening the rear ARB increases mechanical compliance across the rear axle, maximizing rear traction under throttle application and curing corner exit oversteer.
""",
    "02-springs-and-ride-height.md": """# Springs, Ride Height & Bottoming

## Spring Rate Fundamentals

### Bottoming & Platform Control (Springs)
When the car experiences bottoming over high-speed compressions, curbs, or heavy braking zones, increase front or rear spring rates (or heave/third spring stiffness) and raise static ride height. Stiffer springs prevent the splitter or floor from striking the track surface.

### Mid-Corner Understeer & Oversteer (Springs)
Softer front springs increase mechanical grip at the front axle over bumpy surfaces and reduce entry and mid-corner understeer, provided the front splitter does not bottom out. Softer rear springs improve rear mechanical traction on corner exit and reduce exit oversteer, whereas stiffer rear springs increase rotation and reduce mid-corner understeer.
""",
    "03-dampers-low-and-high-speed.md": """# Damper Tuning: Low-Speed & Fast Bump/Rebound

## Low-Speed Dampers (Body Roll & Pitch Control)

### Corner Entry Understeer & Oversteer (Low-Speed Dampers)
Low-speed dampers control chassis pitch and roll rates during driver inputs (braking, turn-in, throttle application).
- To fix corner entry understeer: decrease front low-speed bump damping or increase rear low-speed rebound damping so weight transfers cleanly onto the front tires on turn-in.
- To fix corner entry oversteer: increase front low-speed bump damping or decrease rear low-speed rebound damping to slow down forward pitch transfer during braking and turn-in.

### Corner Exit Oversteer & Understeer (Low-Speed Dampers)
- To fix corner exit oversteer under power: soften rear low-speed bump or increase front low-speed rebound so the rear squats smoothly and maintains traction.
- To fix corner exit understeer: stiffen rear low-speed bump or reduce front low-speed rebound to keep nose weight over the front axle when picking up the throttle.

## High-Speed / Fast Dampers (Curbs & Bumps)

### Bottoming & Curb Compliance (Fast Dampers)
High-speed (fast) dampers govern suspension velocity over sharp track bumps and kerb strikes. If the car skips, loses traction, or exhibits sudden understeer or oversteer over curbs, soften fast bump and fast rebound damping. If the chassis bottoms out over abrupt compressions, increase fast bump damping slightly.
""",
    "04-differential-tuning.md": """# Differential Setup (Preload, Coast & Power Ramp)

## Coast / Deceleration Differential

### Corner Entry Understeer (Diff Coast & Preload)
Excessive differential locking during deceleration prevents the inside and outside rear wheels from rotating at different speeds, pushing the car wide on turn-in. Decrease coast locking (use a higher coast ramp angle or fewer clutch plates) or lower differential preload to allow the rear axle to rotate freely and cure entry understeer.

### Corner Entry Oversteer (Diff Coast & Preload)
If the rear end steps out under braking and initial turn-in, increase differential coast locking or increase preload to tie the rear wheels together and stabilize the car on corner entry.

## Power / Acceleration Differential

### Corner Exit Oversteer & Tire Wear (Diff Power)
On corner exit, if the car snaps into oversteer when applying throttle or suffers high rear tire wear from inside wheelspin, adjust the power ramp and preload carefully: reducing power locking prevents both rear tires from breaking traction simultaneously on high-grip surfaces, while increasing locking prevents inside wheelspin in tight hairpin exits.
""",
    "05-alignment-camber-and-toe.md": """# Alignment: Camber, Caster & Toe

## Camber & Tire Wear

### Mid-Corner Grip & Tire Wear (Camber & Toe)
Negative camber maximizes the tire contact patch when the car is loaded laterally in mid-corner.
- If the car exhibits mid-corner understeer and outside shoulder tire wear on the front tires, increase front negative camber.
- Excessive negative camber or excessive toe angles increase rolling scrubbing and inner-shoulder tire wear on long stints, degrading braking stability and straight-line speed.

## Toe Settings

### Corner Entry Response vs Exit Stability (Camber & Toe)
- Slight front toe-out sharpens initial turn-in response and reduces corner entry understeer, at the cost of increased front tire wear and inner shoulder temperatures.
- Rear toe-in stabilizes the rear axle under braking and power application, reducing entry and exit oversteer.
""",
    "06-brake-bias-and-aero-rake.md": """# Brake Bias & Aerodynamic Rake

## Brake Bias Tuning

### Corner Entry Understeer & Front Lockup (Brake Bias)
If the front tires lock or saturate longitudinally during trail braking—causing corner entry understeer—move brake bias rearward (lower percentage, e.g., -0.5% to -1.0%). This frees up front tire grip for lateral cornering forces.

### Corner Entry Oversteer & Rear Instability (Brake Bias)
If the rear axle becomes loose or steps out under heavy braking and turn-in (entry oversteer), move brake bias forward (higher percentage, e.g., +0.5% to +1.0%) to shift braking workload onto the front tires and stabilize the rear.

## Aerodynamic Balance & Rake

### High-Speed Understeer & Oversteer (Aero & Rake)
Increasing rear ride height relative to front ride height increases aerodynamic rake, shifting the center of pressure forward (more front downforce) and reducing mid-corner high-speed understeer. If the car suffers from high-speed entry or mid-corner oversteer, lower the rear ride height (reduce rake) or add a click of rear wing angle.
""",
}


def html_to_markdown(html_content: str) -> str:
    """Extract clean Markdown from <main> or <article> HTML tags, stripping navigation chrome."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Minimal regex fallback if BeautifulSoup is unavailable
        text = re.sub(r"<(nav|header|footer|aside|script|style)[^>]*>.*?</\1>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    soup = BeautifulSoup(html_content, "html.parser")

    # Strip navigation chrome, headers, footers, scripts, and styles
    for chrome_tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style", "noscript"]):
        chrome_tag.decompose()

    container = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []

    for element in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        tag = element.name.lower()
        if tag == "h1":
            lines.append(f"\n# {text}\n")
        elif tag == "h2":
            lines.append(f"\n## {text}\n")
        elif tag == "h3":
            lines.append(f"\n### {text}\n")
        elif tag == "h4":
            lines.append(f"\n#### {text}\n")
        elif tag == "li":
            lines.append(f"- {text}")
        elif tag == "blockquote":
            lines.append(f"> {text}")
        elif tag == "pre":
            lines.append(f"\n```\n{text}\n```\n")
        else:
            lines.append(f"\n{text}\n")

    markdown = "\n".join(lines)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown + "\n"


def _slugify_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "index.md"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path).strip("-").lower()
    return f"{slug}.md"


def write_fallback_gitbook_docs(output_dir: Path | None = None) -> list[Path]:
    """Write the offline fallback GitBook setup documentation to disk."""
    target_dir = output_dir or settings.gitbook_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    for filename, content in FALLBACK_GITBOOK_DOCS.items():
        path = target_dir / filename
        path.write_text(content.strip() + "\n", encoding="utf-8")
        written_files.append(path)
    logger.info("Wrote %d fallback GitBook markdown files to %s", len(written_files), target_dir)
    return written_files


def crawl_gitbook(
    root_url: str | None = None,
    output_dir: Path | None = None,
    max_pages: int = 50,
    use_fallback_on_error: bool = True,
) -> list[Path]:
    """Crawl the GitBook documentation using Playwright and save clean Markdown files.

    Loads the root page, extracts sidebar <nav> links under the same root path, and
    recursively visits all topic pages. Falls back to the built-in setup documentation
    corpus if Playwright or network access is restricted.
    """
    base_url = (root_url or settings.GITBOOK_ROOT_URL).rstrip("/")
    target_dir = output_dir or settings.gitbook_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    base_parsed = urlparse(base_url)
    base_prefix = base_parsed.path.rstrip("/")

    visited: set[str] = set()
    queue: list[str] = [base_url]
    saved_files: list[Path] = []

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            while queue and len(visited) < max_pages:
                current_url = queue.pop(0).split("#")[0].rstrip("/")
                if current_url in visited:
                    continue
                visited.add(current_url)

                logger.info("Crawling GitBook page: %s", current_url)
                page.goto(current_url, wait_until="networkidle", timeout=20000)

                # Extract sidebar <nav> links and any internal links within the GitBook root
                hrefs: Iterable[str] = page.eval_on_selector_all(
                    "nav a[href], aside a[href], a[href]",
                    "elements => elements.map(el => el.getAttribute('href')).filter(Boolean)",
                )
                for href in hrefs:
                    full_url = urljoin(current_url + "/", href).split("#")[0].rstrip("/")
                    parsed = urlparse(full_url)
                    if (
                        parsed.netloc == base_parsed.netloc
                        and parsed.path.startswith(base_prefix)
                        and full_url not in visited
                        and full_url not in queue
                    ):
                        queue.append(full_url)

                html = page.content()
                md = html_to_markdown(html)
                if md.strip():
                    file_path = target_dir / _slugify_url(current_url)
                    file_path.write_text(md, encoding="utf-8")
                    saved_files.append(file_path)

            browser.close()

    except Exception as exc:
        logger.warning(
            "Live Playwright crawl could not run (%s). Using offline GitBook fallback corpus.",
            exc,
        )
        if use_fallback_on_error:
            return write_fallback_gitbook_docs(target_dir)
        raise

    if not saved_files and use_fallback_on_error:
        return write_fallback_gitbook_docs(target_dir)

    return saved_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    files = crawl_gitbook()
    print(f"Saved {len(files)} GitBook markdown documents to {settings.gitbook_dir}")
