import asyncio, os, base64, time, sys, traceback
from playwright.async_api import async_playwright

SCRIPT_VERSION = "screenshot_loop_v3_inmemory"

INTERVAL = float(os.getenv("SCREENSHOT_INTERVAL", "0.2"))  # every few hundred ms
TARGET_URL = os.getenv("SCREENSHOT_URL", "https://example.com")
OUTPUT_DIR = os.getenv("SCREENSHOT_DIR", "/app/screenshots")
LATEST_FILE = os.getenv("SCREENSHOT_FILE", "latest.png")  # still maintain a latest pointer
CDP_ENDPOINT = os.getenv("CDP_ENDPOINT", "http://127.0.0.1:9222")
MAX_HISTORY = int(os.getenv("SCREENSHOT_MAX_FILES", "300"))  # rotation cap

os.makedirs(OUTPUT_DIR, exist_ok=True)

async def main():
    print(f"[screenshot_loop] Starting {SCRIPT_VERSION}", flush=True)
    async with async_playwright() as pw:
        # Connect to existing chromium over CDP with retries
        retries = int(os.getenv("SCREENSHOT_CDP_RETRIES", "40"))
        delay = float(os.getenv("SCREENSHOT_CDP_DELAY", "0.25"))
        browser = None
        for attempt in range(1, retries + 1):
            try:
                browser = await pw.chromium.connect_over_cdp(f"{CDP_ENDPOINT}")
                print(f"[screenshot_loop] Connected to existing Chromium via CDP {CDP_ENDPOINT}", flush=True)
                break
            except Exception as e:
                if attempt == 1:
                    print(f"[screenshot_loop] Waiting for CDP {CDP_ENDPOINT} ...", flush=True)
                if attempt == retries:
                    print(f"[screenshot_loop] CDP connection failed after {retries} attempts: {e}. Falling back to launching a headless browser.", flush=True)
                    try:
                        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                        print("[screenshot_loop] Launched fallback Chromium instance (no persistent CDP)", flush=True)
                    except Exception as le:
                        print(f"[screenshot_loop] Fallback launch failed: {le}", flush=True)
                        raise
                else:
                    await asyncio.sleep(delay)
        if browser is None:
            raise RuntimeError("Could not establish any Chromium instance")
        contexts = browser.contexts
        if contexts:
            ctx = contexts[0]
        else:
            ctx = await browser.new_context()
        pages = ctx.pages
        if pages:
            page = pages[0]
        else:
            page = await ctx.new_page()
            await page.goto(TARGET_URL)
        latest_path = os.path.join(OUTPUT_DIR, LATEST_FILE)
        while True:
            epoch_ms = int(time.time() * 1000)
            hist_filename = f"shot_{epoch_ms}.png"
            hist_path = os.path.join(OUTPUT_DIR, hist_filename)
            tmp_latest = latest_path + ".tmp"
            try:
                # Take screenshot in-memory to avoid Playwright path / mime quirks
                data = await page.screenshot(full_page=False, type="png")
                # Write history file
                with open(hist_path, 'wb') as f:
                    f.write(data)
                # Update latest atomically
                try:
                    with open(tmp_latest, 'wb') as f:
                        f.write(data)
                    os.replace(tmp_latest, latest_path)
                except Exception as e:
                    print(f"[screenshot_loop] latest copy error {e}", flush=True)
                    try:
                        if os.path.exists(tmp_latest):
                            os.remove(tmp_latest)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[screenshot_loop] screenshot error {e}", flush=True)
                traceback.print_exc()
            # Rotation
            try:
                files = sorted(
                    [f for f in os.listdir(OUTPUT_DIR) if f.startswith('shot_') and f.endswith('.png')]
                )
                if len(files) > MAX_HISTORY:
                    for old in files[: len(files) - MAX_HISTORY]:
                        try:
                            os.remove(os.path.join(OUTPUT_DIR, old))
                        except:  # noqa
                            pass
            except Exception:
                pass
            await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
