"""Open FishSTOP as a real browser session and wake it when necessary."""

from __future__ import annotations

import os
import time
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


APP_URL = os.environ.get(
    "FISHSTOP_URL",
    "https://fishstop-eml.streamlit.app/",
)
WAKE_BUTTON = "Yes, get this app back up!"
# This class belongs to the public download page and is independent from its
# marketing copy, which can change without making the app unavailable.
APP_READY_SELECTOR = "section.download-section"
FAILURE_SCREENSHOT = Path("artifacts/keep-awake-failure.png")


def wait_until_loaded(page, timeout_seconds: int = 180) -> bool:
    """Return whether a sleeping app was woken after its UI becomes visible."""
    deadline = time.monotonic() + timeout_seconds
    wake_requested = False
    wake_button = page.get_by_role("button", name=WAKE_BUTTON, exact=True)

    while time.monotonic() < deadline:
        try:
            if not wake_requested and wake_button.is_visible():
                print("FishSTOP is sleeping; requesting wake-up.")
                wake_button.click()
                wake_requested = True

            for frame in page.frames:
                if frame.locator(APP_READY_SELECTOR).is_visible():
                    return wake_requested
        except PlaywrightError:
            # Streamlit replaces frames while moving from the sleep page
            # to the application. Retry against the new page structure.
            pass

        page.wait_for_timeout(1_000)

    raise RuntimeError(
        "FishSTOP did not finish loading within 180 seconds "
        f"(url={page.url!r}, title={page.title()!r})."
    )


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
            if wait_until_loaded(page):
                print("FishSTOP woke and loaded successfully.")
            else:
                print("FishSTOP is already awake.")
        except Exception:
            FAILURE_SCREENSHOT.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(FAILURE_SCREENSHOT), full_page=True)
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
