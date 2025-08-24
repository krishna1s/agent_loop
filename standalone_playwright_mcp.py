import asyncio
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from typing import Optional, Dict, Any

class StandalonePlaywrightMCP:

    async def get_dom_snapshot(self) -> str:
        """
        Return a DOM snapshot using Playwright's page.snapshot() if available, else fallback to page.content().
        """
        # Playwright's page.snapshot() is available in recent versions (>=1.39)
        # It returns a string with the DOM snapshot (not a screenshot)
        try:
            if hasattr(self.page, "snapshot"):
                # Use Playwright's native DOM snapshot
                return await self.page.snapshot()
            else:
                # Fallback: use page.content() (HTML only, not a full snapshot)
                return await self.page.content()
        except Exception as e:
            return f"[Error] Failed to get DOM snapshot: {e}"
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.browser_config = {
            "headless": False,
            "devtools": True,
            "slow_mo": 500,
            "viewport": {"width": 1280, "height": 720},
            "args": [
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--allow-running-insecure-content"
            ]
        }

    async def initialize_browser(self):
        if not self.playwright:
            self.playwright = await async_playwright().start()
        if not self.browser:
            self.browser = await self.playwright.chromium.launch(
                channel="msedge",
                headless=self.browser_config["headless"],
                devtools=self.browser_config["devtools"],
                slow_mo=self.browser_config["slow_mo"],
                args=self.browser_config["args"]
            )
        if not self.context:
            self.context = await self.browser.new_context(
                viewport=self.browser_config["viewport"]
            )
        if not self.page:
            self.page = await self.context.new_page()
        return {"success": True, "message": "Browser initialized"}

    async def close_browser(self):
        if self.page:
            await self.page.close()
            self.page = None
        if self.context:
            await self.context.close()
            self.context = None
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

    async def navigate_to_url(self, url: str) -> str:
        await self.page.goto(url)
        return f"Navigated to {url}"

    async def click_element(self, selector: str) -> str:
        await self.page.click(selector)
        return f"Clicked element {selector}"

    async def type_text(self, selector: str, text: str) -> str:
        await self.page.fill(selector, text)
        return f"Typed '{text}' in {selector}"

    async def take_screenshot(self, filename: str = None) -> str:
        if not filename:
            filename = "screenshot.png"
        await self.page.screenshot(path=filename)
        return f"Screenshot saved as {filename}"

    async def evaluate_javascript(self, script: str) -> Any:
        return await self.page.evaluate(script)

    async def wait_for_condition(self, condition: str, timeout: int = 30) -> str:
        await self.page.wait_for_selector(condition, timeout=timeout*1000)
        return f"Waited for {condition}"

    async def hover_element(self, selector: str) -> str:
        await self.page.hover(selector)
        return f"Hovered over {selector}"

    async def select_dropdown_option(self, selector: str, option: str) -> str:
        await self.page.select_option(selector, option)
        return f"Selected {option} in {selector}"

    async def press_key(self, key: str) -> str:
        await self.page.keyboard.press(key)
        return f"Pressed key {key}"
