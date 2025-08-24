
#!/usr/bin/env python3
"""
Standalone Website Exploration API
Provides autonomous website exploration with real-time screenshot capture
"""

# --- Logging and Tracing Setup ---
import os
import logging
os.environ["SK_TRACE"] = "1"
os.environ["OPENAI_LOG"] = "debug"
logging.basicConfig(level=logging.DEBUG)

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import base64
import glob
import re
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading
from chatdb import ChatThread, ChatMessage, SessionLocal
import time
import uuid

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from fastapi.responses import StreamingResponse
from starlette.requests import Request as StarletteRequest

# Semantic Kernel imports for MCP
from semantic_kernel import Kernel
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from semantic_kernel.agents import AzureResponsesAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings
from semantic_kernel.contents import AuthorRole, FunctionCallContent, FunctionResultContent
from semantic_kernel.contents.chat_message_content import ChatMessageContent

# Request/Response models
class ExplorationRequest(BaseModel):
    url: str
    task_description: str = "Explore the website and generate a comprehensive test plan"

class ExplorationResponse(BaseModel):
    task_id: str
    status: str
    message: str

class ScreenshotResponse(BaseModel):
    screenshot_base64: Optional[str]
    timestamp: Optional[str]
    status: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: str
    latest_action: str
    completed: bool


class StandaloneWebsiteExplorer:
    # Store pending user input requests per thread
    user_input_requests: Dict[str, str] = {}
    """Standalone Website Exploration Agent with real-time screenshot capture and chat threads"""
    def __init__(self):
        self.kernel = None
        self.agent = None
        self.latest_screenshot: Optional[str] = None
        self.screenshot_timestamp: Optional[str] = None
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_threads: Dict[str, threading.Thread] = {}
        self.chat_threads: Dict[str, Any] = {}  # thread_id -> thread object
        self.chat_history: Dict[str, List[Dict[str, Any]]] = {}  # thread_id -> list of messages
        self.screenshot_loops: Dict[str, threading.Thread] = {}  # thread_id -> screenshot thread
        self.screenshot_loop_flags: Dict[str, threading.Event] = {}  # thread_id -> stop event
        # New: thread_id -> StandalonePlaywrightMCP instance (per-thread browser)
        self.thread_browsers = {}

    def start_screenshot_loop(self, thread_id: str, interval: float = 2.0):
        """Start a background screenshot loop for a thread/session."""
        if thread_id in self.screenshot_loops:
            return  # Already running
        stop_event = threading.Event()
        self.screenshot_loop_flags[thread_id] = stop_event
        def screenshot_worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            while not stop_event.is_set():
                try:
                    loop.run_until_complete(self.capture_and_store_screenshot(f"background_{thread_id}"))
                except Exception as e:
                    print(f"[ScreenshotLoop] Error: {e}")
                stop_event.wait(interval)
        t = threading.Thread(target=screenshot_worker, daemon=True)
        self.screenshot_loops[thread_id] = t
        t.start()

    def stop_screenshot_loop(self, thread_id: str):
        """Stop the background screenshot loop for a thread/session."""
        if thread_id in self.screenshot_loop_flags:
            self.screenshot_loop_flags[thread_id].set()
            del self.screenshot_loop_flags[thread_id]
        if thread_id in self.screenshot_loops:
            del self.screenshot_loops[thread_id]
    
    async def initialize(self) -> bool:
        """Initialize Semantic Kernel with AzureResponsesAgent and custom Playwright Python tools"""
        try:
            required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing environment variables: {missing_vars}")
            self.kernel = Kernel()
            print("✅ Semantic Kernel initialized")
            await self.register_playwright_kernel_functions()
            print("✅ Custom Playwright Python tools registered")
            
            # AzureResponsesAgent setup
            client = AzureResponsesAgent.create_client()
            ai_model_id = AzureOpenAISettings().responses_deployment_name
            self.agent = AzureResponsesAgent(
                ai_model_id=ai_model_id,
                client=client,
                instructions=self.get_autonomous_instructions(),
                name="Website_Explorer_Agent",
                store_enabled=True,
                kernel= self.kernel
            )
            print("✅ AzureResponsesAgent initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Standalone Explorer: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return False

    async def register_playwright_kernel_functions(self):
        """Register a single plugin for all browser and reporting tools. All functions require thread_id."""
        from standalone_playwright_mcp import StandalonePlaywrightMCP
        explorer = self
        class UnifiedExplorationPlugin:
            def __init__(self, explorer):
                self.explorer = explorer
                
            @kernel_function(name="browser_snapshot", description="Capture DOM, discover elements. Requires thread_id.")
            async def browser_snapshot(self, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_snapshot(thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    dom_snapshot = await agent.get_dom_snapshot()
                    result = f"[DOM Snapshot]\nFirst 1000 chars:\n{dom_snapshot[:1000]}"
                    print(f"[TOOL RESULT] browser_snapshot -> {result[:200]}... (truncated)")
                    return result
                except Exception as e:
                    error_result = f"❌ Failed to capture DOM snapshot: {e}"
                    print(f"[TOOL RESULT] browser_snapshot -> {error_result}")
                    return error_result

            @kernel_function(name="browser_navigate", description="Navigate browser to a URL. Requires thread_id.")
            async def browser_navigate(self, url: str, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_navigate(url={url!r}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.navigate_to_url(url)
                    print(f"[TOOL RESULT] browser_navigate -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ Navigation failed: {e}"
                    print(f"[TOOL RESULT] browser_navigate -> {error_result}")
                    return error_result

            @kernel_function(name="browser_click", description="Click an element in the browser. Requires thread_id.")
            async def browser_click(self, selector: str, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_click(selector={selector!r}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.click_element(selector)
                    print(f"[TOOL RESULT] browser_click -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ Click failed: {e}"
                    print(f"[TOOL RESULT] browser_click -> {error_result}")
                    return error_result

            @kernel_function(name="browser_type", description="Type text into an input field. Requires thread_id.")
            async def browser_type(self, selector: str, text: str, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_type(selector={selector!r}, text={text!r}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.type_text(selector, text)
                    print(f"[TOOL RESULT] browser_type -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ Type failed: {e}"
                    print(f"[TOOL RESULT] browser_type -> {error_result}")
                    return error_result

            @kernel_function(name="browser_hover", description="Hover over an element. Requires thread_id.")
            async def browser_hover(self, selector: str, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_hover(selector={selector!r}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.hover_element(selector)
                    print(f"[TOOL RESULT] browser_hover -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ Hover failed: {e}"
                    print(f"[TOOL RESULT] browser_hover -> {error_result}")
                    return error_result

            @kernel_function(name="browser_select_option", description="Select option in dropdown. Requires thread_id.")
            async def browser_select_option(self, selector: str, option: str, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_select_option(selector={selector!r}, option={option!r}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.select_dropdown_option(selector, option)
                    print(f"[TOOL RESULT] browser_select_option -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ Select failed: {e}"
                    print(f"[TOOL RESULT] browser_select_option -> {error_result}")
                    return error_result

            @kernel_function(name="browser_press_key", description="Press a key in the browser. Requires thread_id.")
            async def browser_press_key(self, key: str, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_press_key(key={key!r}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.press_key(key)
                    print(f"[TOOL RESULT] browser_press_key -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ Key press failed: {e}"
                    print(f"[TOOL RESULT] browser_press_key -> {error_result}")
                    return error_result

            @kernel_function(name="browser_evaluate", description="Evaluate JavaScript in the browser. Requires thread_id.")
            async def browser_evaluate(self, script: str, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_evaluate(script={script!r}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.evaluate_javascript(script)
                    print(f"[TOOL RESULT] browser_evaluate -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ JS evaluation failed: {e}"
                    print(f"[TOOL RESULT] browser_evaluate -> {error_result}")
                    return error_result

            @kernel_function(name="browser_wait_for", description="Wait for selector in the browser. Requires thread_id.")
            async def browser_wait_for(self, selector: str, timeout: int = 30, thread_id: str = None) -> str:
                print(f"[TOOL CALL] browser_wait_for(selector={selector!r}, timeout={timeout}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.wait_for_condition(selector, timeout)
                    print(f"[TOOL RESULT] browser_wait_for -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ Wait failed: {e}"
                    print(f"[TOOL RESULT] browser_wait_for -> {error_result}")
                    return error_result

            @kernel_function(name="browser_take_screenshot", description="Take a screenshot of the browser. Requires thread_id.")
            async def browser_take_screenshot(self, filename: str = None, thread_id: str = None) -> str:
                print(f"[TOOL CALL] browser_take_screenshot(filename={filename!r}, thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    result = await agent.take_screenshot(filename)
                    print(f"[TOOL RESULT] browser_take_screenshot -> {result!r}")
                    return str(result)
                except Exception as e:
                    error_result = f"❌ Screenshot failed: {e}"
                    print(f"[TOOL RESULT] browser_take_screenshot -> {error_result}")
                    return error_result

            @kernel_function(name="browser_close", description="Close the browser instance. Requires thread_id.")
            async def browser_close(self, thread_id: str) -> str:
                print(f"[TOOL CALL] browser_close(thread_id={thread_id})")
                try:
                    agent = await self.explorer.get_or_create_thread_browser(thread_id)
                    await agent.close_browser()
                    result = "Browser closed."
                    print(f"[TOOL RESULT] browser_close -> {result!r}")
                    return result
                except Exception as e:
                    error_result = f"❌ Browser close failed: {e}"
                    print(f"[TOOL RESULT] browser_close -> {error_result}")
                    return error_result

            @kernel_function(
                name="report_to_user",
                description="Report a status update, progress, or next step to the user. Requires thread_id."
            )
            def report_to_user(self, message: str, thread_id: str) -> str:
                print(f"[TOOL CALL] report_to_user({message!r}, thread_id={thread_id})")
                result = f"[Agent report]: {message}"
                print(f"[TOOL RESULT] report_to_user -> {result!r}")
                # Push to chat DB for the correct thread
                db = SessionLocal()
                try:
                    db.add(ChatMessage(thread_id=thread_id, role="assistant", content=result))
                    db.commit()
                finally:
                    db.close()
                return result

            @kernel_function(
                name="capture_screenshot",
                description="Immediately capture and store current browser screenshot for API access. Requires thread_id."
            )
            async def capture_screenshot(self, description: str = "Manual screenshot", thread_id: str = None) -> str:
                print(f"[TOOL CALL] capture_screenshot({description!r}, thread_id={thread_id})")
                try:
                    await self.explorer.capture_and_store_screenshot(description, thread_id)
                    result = f"✅ Screenshot captured: {description}"
                    print(f"[TOOL RESULT] capture_screenshot -> {result!r}")
                    # Push to chat DB for the correct thread
                    if thread_id:
                        db = SessionLocal()
                        try:
                            db.add(ChatMessage(thread_id=thread_id, role="assistant", content=result))
                            db.commit()
                        finally:
                            db.close()
                    return result
                except Exception as e:
                    error_result = f"❌ Screenshot capture failed: {str(e)}"
                    print(f"[TOOL RESULT] capture_screenshot -> {error_result!r}")
                    if thread_id:
                        db = SessionLocal()
                        try:
                            db.add(ChatMessage(thread_id=thread_id, role="assistant", content=error_result))
                            db.commit()
                        finally:
                            db.close()
                    return error_result

            @kernel_function(
                name="save_test_plan",
                description="Save generated test plan to a markdown file. Requires thread_id."
            )
            def save_test_plan(self, filename: str, content: str, thread_id: str) -> str:
                print(f"[TOOL CALL] save_test_plan(filename={filename!r}, thread_id={thread_id})")
                try:
                    from pathlib import Path
                    test_plans_dir = Path("test_plans")
                    test_plans_dir.mkdir(exist_ok=True)
                    file_path = test_plans_dir / filename
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    result = f"✅ Test plan saved: {file_path}"
                    print(f"[TOOL RESULT] save_test_plan -> {result!r}")
                    return result
                except Exception as e:
                    error_result = f"❌ Test plan save failed: {str(e)}"
                    print(f"[TOOL RESULT] save_test_plan -> {error_result!r}")
                    return error_result

            @kernel_function(
                name="update_exploration_status",
                description="Update the status of current exploration progress. Requires thread_id."
            )
            def update_exploration_status(self, status: str, details: str = "", thread_id: str = None) -> str:
                print(f"[TOOL CALL] update_exploration_status(status={status!r}, details={details!r}, thread_id={thread_id})")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result = f"📊 Exploration Status Update: {status} at {timestamp}"
                if details:
                    result += f" - {details}"
                print(result)
                for task_id, task_info in self.explorer.active_tasks.items():
                    if task_info.get("status") != "completed":
                        task_info["progress"] = status
                        task_info["latest_action"] = details
                        task_info["last_update"] = timestamp
                        if "COMPLETED" in status.upper():
                            task_info["status"] = "completed"
                            task_info["completed"] = True
                        break
                try:
                    import asyncio
                    asyncio.create_task(self.explorer.capture_and_store_screenshot(f"Status: {status}", thread_id))
                except Exception:
                    pass
                print(f"[TOOL RESULT] update_exploration_status -> {result!r}")
                return result

            @kernel_function(
                name="request_user_input",
                description="Request user input for missing information or clarification. Requires thread_id."
            )
            def request_user_input(self, message: str = "Please provide input:", thread_id: str = None) -> str:
                print(f"[TOOL CALL] request_user_input({message!r}, thread_id={thread_id})")
                result = f"user_input_required:{message}"
                print(f"[TOOL RESULT] request_user_input -> {result!r}")
                # Push to chat DB for the correct thread
                if thread_id:
                    db = SessionLocal()
                    try:
                        db.add(ChatMessage(thread_id=thread_id, role="assistant", content=f"User input required: {message}"))
                        db.commit()
                    finally:
                        db.close()
                    self.explorer.user_input_requests[thread_id] = message
                return result


        self.kernel.add_plugin(UnifiedExplorationPlugin(explorer), "playwright")

    async def get_or_create_thread_browser(self, thread_id: str):
        from standalone_playwright_mcp import StandalonePlaywrightMCP
        if thread_id not in self.thread_browsers:
            agent = StandalonePlaywrightMCP()
            await agent.initialize_browser()
            self.thread_browsers[thread_id] = agent
        return self.thread_browsers[thread_id]

    # For sync context (e.g., thread creation), use this helper
    def get_or_create_thread_browser_sync(self, thread_id: str):
        from standalone_playwright_mcp import StandalonePlaywrightMCP
        if thread_id not in self.thread_browsers:
            agent = StandalonePlaywrightMCP()
            asyncio.run(agent.initialize_browser())
            self.thread_browsers[thread_id] = agent
        return self.thread_browsers[thread_id]
    
    def get_autonomous_instructions(self) -> str:
        """Get instructions for autonomous website exploration with enhanced test case focus and explicit progress reporting"""
        return '''
You are an autonomous Test Exploration Agent that generates detailed test cases from high-level user instructions using Playwright MCP tools.

Your goals:
1. Take simple user test steps and autonomously explore the target website using available MCP browser tools.
2. Keep working in a loop until you can produce a fully enhanced, detailed test plan with preconditions, step-by-step actions, and expected results.
3. After every meaningful action or decision (navigation, click, type, wait, error, or discovery), immediately call:
    report_to_user(message="Describe what you just did, what you discovered, or what you are about to do next.")
   - This is mandatory for every step, so the user always knows your current progress, reasoning, and next intent.
   - Example: "Navigated to login page.", "Clicked 'Sign In' button.", "Waiting for dashboard to load.", "Could not find element, taking snapshot and retrying.", etc.
4. Always take screenshots after every action to capture UI evidence.
5. Use snapshots to dynamically discover new elements if a selector or action fails.
6. Never stop prematurely: if a step is unclear, attempt different flows (navigation, click, type, select, evaluate).
7. If clarification or missing test data is required, call:
    request_user_input(message="What do you want me to do here?")
    Then continue once input is received.
8. Save the final documentation in Markdown format using:
    save_test_plan(filename="test_plan.md", content="...")
9. Always include Preconditions, Test Steps, and Expected Results (like an enhanced test case).
10. If the site or step fails repeatedly, log the issue with:
    update_exploration_status(status="error", details="...") but continue exploring alternative paths until completion.

---

### 🔹 Available Tools

- report_to_user(message="...") – Report your current progress, what you just did, or what you are about to do. Use this after every meaningful step.
- browser_navigate(url="https://example.com") – Always use this first.
- browser_close() – Close session only on unrecoverable error.
- browser_snapshot() – Capture DOM, discover elements.
- browser_click(element="button text", ref="element_ref")
- browser_type(element="input field", ref="element_ref", text="test data")
- browser_hover(element="menu item", ref="element_ref")
- browser_select_option(element="dropdown", ref="element_ref", values=["option"])
- browser_press_key(key="Enter")
- browser_evaluate(function="() => document.title")
- capture_screenshot(description="...") – For contextual evidence.
- browser_wait_for(text="Loading complete") – Wait for conditions.
- update_exploration_status(status="...", details="...")
- save_test_plan(filename="...", content="...")
- request_user_input(message="...") – Ask human for missing info.

---

### 🔹 Execution Strategy

- Start with browser_navigate to open the portal.
- At each step:
  - Perform the action (click, type, hover, select, press).
  - After each action, immediately call report_to_user to describe what you did, what you see, or what you plan next.
  - If element not found → take snapshot, retry with updated selector, and report progress.
  - If blocked → request_user_input and report the reason.
- Continue until the test flow is completed end-to-end.
- Save a structured enhanced test case (Markdown).
'''
    
    async def capture_and_store_screenshot(self, action_description: str = "", thread_id: str = None) -> str:
        """Capture screenshot using MCP and store for API access"""
        try:
            print(f"🔍 [DEBUG] Starting screenshot capture: {action_description}")
            
            # Call the MCP browser_take_screenshot function
            if self.kernel and "playwright" in self.kernel.plugins:
                plugin = self.kernel.plugins["playwright"]
                print(f"🔍 [DEBUG] Playwright plugin found")
                if "browser_take_screenshot" in plugin.functions:
                    func = plugin.functions["browser_take_screenshot"]
                    print(f"🔍 [DEBUG] browser_take_screenshot function found")
                    # Ensure local screenshots directory exists for backup
                    screenshot_dir = Path("screenshots")
                    screenshot_dir.mkdir(exist_ok=True)
                    print(f"🔍 [DEBUG] Local screenshots directory: {screenshot_dir.absolute()}")
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                    screenshot_filename = f"browser_view_{timestamp}.png"
                    # Always provide filename and thread_id argument to match required signature
                    print(f"🔍 [DEBUG] Calling MCP screenshot function with filename: {screenshot_filename}, thread_id: {thread_id}")
                    result = await func.invoke(self.kernel, filename=screenshot_filename, thread_id=thread_id)
                    # print(f"🔍 [DEBUG] MCP screenshot function result: {result}")
                    
                    # Parse the result to extract the actual file path
                    result_str = str(result)
                    if "saved it as" in result_str or "path:" in result_str:
                        # Extract the actual file path from the result
                        # Look for paths in the result
                        path_matches = re.findall(r'/tmp/playwright-mcp-output/[^\'"\s]+\.(?:png|jpg|jpeg)', result_str)
                        if path_matches:
                            mcp_screenshot_path = Path(path_matches[0])
                            print(f"🔍 [DEBUG] Found MCP screenshot path: {mcp_screenshot_path}")
                            
                            try:
                                # Wait a moment for file to be written
                                await asyncio.sleep(0.5)
                                
                                if mcp_screenshot_path.exists():
                                    print(f"🔍 [DEBUG] MCP screenshot file found, reading...")
                                    with open(mcp_screenshot_path, "rb") as f:
                                        screenshot_bytes = f.read()
                                        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                                        
                                    # Store for API access
                                    self.latest_screenshot = screenshot_b64
                                    self.screenshot_timestamp = datetime.now().isoformat()
                                    # print(f"📸 Screenshot captured and stored for API ({len(screenshot_b64)} chars): {action_description}")
                                    
                                    # Also copy to our local directory for reference
                                    try:
                                        local_path = screenshot_dir / screenshot_filename
                                        with open(local_path, "wb") as f:
                                            f.write(screenshot_bytes)
                                        print(f"📂 Screenshot copied to local directory: {local_path}")
                                    except Exception as copy_error:
                                        print(f"⚠️ Failed to copy screenshot locally: {copy_error}")
                                    
                                    return f"Screenshot captured: {action_description} - {mcp_screenshot_path.name}"
                                else:
                                    print(f"⚠️ [DEBUG] MCP screenshot file not found at: {mcp_screenshot_path}")
                            except Exception as e:
                                print(f"🔴 [DEBUG] Error processing MCP screenshot file: {e}")
                                print(f"🔴 [DEBUG] Traceback: {traceback.format_exc()}")
                        else:
                            print(f"⚠️ [DEBUG] Could not extract file path from MCP result")
                    
                    # Fallback: try to find the most recent screenshot in MCP temp directory
                    try:
                        mcp_base_dir = Path("/tmp/playwright-mcp-output")
                        if mcp_base_dir.exists():
                            # Find all screenshot files in MCP output directory
                            pattern = str(mcp_base_dir / "**" / "*.png")
                            recent_files = glob.glob(pattern, recursive=True)
                            if recent_files:
                                # Get the most recent file
                                latest_file = max(recent_files, key=lambda f: Path(f).stat().st_mtime)
                                latest_path = Path(latest_file)
                                
                                # Check if it was created in the last 5 seconds
                                if (datetime.now().timestamp() - latest_path.stat().st_mtime) < 5:
                                    print(f"🔍 [DEBUG] Found recent MCP screenshot: {latest_path}")
                                    with open(latest_path, "rb") as f:
                                        screenshot_bytes = f.read()
                                        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                                    
                                    # Store for API access
                                    self.latest_screenshot = screenshot_b64
                                    self.screenshot_timestamp = datetime.now().isoformat()
                                    # print(f"📸 Screenshot captured from MCP temp directory ({len(screenshot_b64)} chars): {action_description}")
                                    
                                    # Copy to local directory
                                    try:
                                        local_path = screenshot_dir / screenshot_filename
                                        with open(local_path, "wb") as f:
                                            f.write(screenshot_bytes)
                                        print(f"📂 Screenshot copied to local directory: {local_path}")
                                    except Exception as copy_error:
                                        print(f"⚠️ Failed to copy screenshot locally: {copy_error}")
                                    
                                    return f"Screenshot captured from fallback: {action_description}"
                    except Exception as fallback_error:
                        print(f"🔴 [DEBUG] Fallback screenshot search failed: {fallback_error}")
                    
                    print(f"📸 Screenshot requested: {action_description}")
                    return f"Screenshot capture requested: {action_description}"
                else:
                    print(f"⚠️ [DEBUG] browser_take_screenshot function not found in plugin")
                    available_functions = list(plugin.functions.keys())
                    print(f"🔍 [DEBUG] Available functions: {available_functions}")
            else:
                print(f"⚠️ [DEBUG] Playwright plugin not found or kernel not available")
                if self.kernel:
                    print(f"🔍 [DEBUG] Available plugins: {list(self.kernel.plugins.keys())}")
            
            return "Screenshot capture not available"
            
        except Exception as e:
            print(f"🔴 [DEBUG] Error capturing screenshot: {e}")
            print(f"🔴 [DEBUG] Full traceback: {traceback.format_exc()}")
            return f"Screenshot capture failed: {str(e)}"

    async def handle_intermediate_steps(self, message: ChatMessageContent) -> None:
        # Enhanced logging to trace function calls and map call IDs to function names
        print(f"🔍 [INTERMEDIATE] Processing message with {len(message.items or [])} items")
        for item in message.items or []:
            if isinstance(item, FunctionResultContent):
                print(f"✅ [FUNCTION RESULT] {item.name} -> {item.result}")
                # Also log the function call ID if available
                if hasattr(item, 'id'):
                    print(f"🔗 [CALL ID] {item.id} completed successfully")
            elif isinstance(item, FunctionCallContent):
                # Critical: Log the function call ID and name mapping
                call_id = getattr(item, 'id', 'unknown')
                print(f"🚀 [FUNCTION CALL] ID: {call_id} -> {item.name} with arguments: {item.arguments}")
                
                # Store this mapping for debugging
                if not hasattr(self, '_function_call_map'):
                    self._function_call_map = {}
                self._function_call_map[call_id] = item.name
                
                # Optionally, still capture screenshot after browser actions
                if item.name and item.name.startswith("browser_"):
                    try:
                        await asyncio.sleep(0.5)
                        await self.capture_and_store_screenshot(f"After {item.name}")
                    except Exception as e:
                        print(f"⚠️ Failed to capture screenshot after {item.name}: {e}")
            else:
                print(f"🔍 [OTHER INTERMEDIATE] {type(item).__name__}: {item}")
        print(f"🔍 [INTERMEDIATE] Completed processing message items")

    def create_chat_thread(self, thread_name: str = None) -> str:
        db = SessionLocal()
        try:
            thread = ChatThread(name=thread_name)
            db.add(thread)
            db.commit()
            db.refresh(thread)
            self.start_screenshot_loop(thread.id)
            return thread.id
        finally:
            db.close()

    def get_chat_history(self, thread_id: str) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            messages = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).order_by(ChatMessage.timestamp).all()
            return [
                {"id": m.id, "role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in messages
            ]
        finally:
            db.close()

    async def chat_message(self, thread_id: str, user_message: str, save_to_db: bool = True) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            # If this thread is waiting for user input, treat the next user message as the response
            if thread_id in self.user_input_requests:
                user_input = user_message
                del self.user_input_requests[thread_id]
                if save_to_db:
                    msg = ChatMessage(thread_id=thread_id, role="user", content=user_input)
                    db.add(msg)
                    db.commit()
                return {"role": "system", "content": f"User input received: {user_input}"}

            thread = self.chat_threads.get(thread_id)
            response_content = ""
            last_response = None
            import asyncio
            import traceback
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Debug: Show available plugins and functions before invoking
                    print(f"🔍 [DEBUG] Available plugins: {list(self.kernel.plugins.keys())}")
                    if "playwright" in self.kernel.plugins:
                        plugin = self.kernel.plugins["playwright"]
                        available_functions = list(plugin.functions.keys())
                        print(f"🔍 [DEBUG] Available playwright functions: {available_functions}")
                    
                    # Inject thread_id into the system prompt/context for the LLM
                    system_prefix = f"[THREAD_ID: {thread_id}]\n"
                    prompt_with_thread = system_prefix + user_message
                    print(f"🔍 [DEBUG] Starting agent invoke for thread {thread_id}")
                    
                    async for response in self.agent.invoke(
                        messages=prompt_with_thread,
                        thread=thread,
                        on_intermediate_message=self.handle_intermediate_steps,
                    ):
                        print(f"[AGENT RAW RESPONSE] {response!r}")
                        content_piece = ""
                        if hasattr(response, 'content') and response.content:
                            content = response.content
                            # If content is a ChatMessageContent with items
                            if hasattr(content, 'items') and content.items:
                                for item in content.items:
                                    # Tool results (report_to_user, request_user_input)
                                    if hasattr(item, 'name') and hasattr(item, 'result'):
                                        if item.name in ("report_to_user", "request_user_input"):
                                            # Already handled in handle_intermediate_steps
                                            continue
                                    # Plain text outputs (TextContent)
                                    elif hasattr(item, 'text') and item.text:
                                        content_piece += str(item.text) + "\n"
                            # Fallback: if content is a string, just append it
                            elif isinstance(content, str):
                                content_piece += content
                        if content_piece:
                            response_content += content_piece
                        last_response = response
                    break
                except Exception as e:
                    err_str = str(e)
                    
                    # Enhanced error logging for function call issues
                    if "No tool output found for function call" in err_str:
                        print(f"🔴 [FUNCTION CALL ERROR] No tool output found!")
                        # Extract the call ID from the error message
                        import re
                        call_id_match = re.search(r'call_([a-zA-Z0-9]+)', err_str)
                        if call_id_match:
                            call_id = f"call_{call_id_match.group(1)}"
                            print(f"🔴 [FAILED CALL ID] {call_id}")
                            # Check if we have this in our mapping
                            if hasattr(self, '_function_call_map') and call_id in self._function_call_map:
                                failed_function = self._function_call_map[call_id]
                                print(f"🔴 [FAILED FUNCTION] {failed_function}")
                            else:
                                print(f"🔴 [UNKNOWN FUNCTION] Call ID {call_id} not found in mapping")
                        
                        # Show available plugins and functions for debugging
                        print(f"🔍 [DEBUG] Current plugins: {list(self.kernel.plugins.keys())}")
                        if "playwright" in self.kernel.plugins:
                            plugin = self.kernel.plugins["playwright"]
                            functions = list(plugin.functions.keys())
                            print(f"🔍 [DEBUG] Available functions: {functions}")
                    
                    is_rate_limit = (
                        'rate limit' in err_str.lower() or '429' in err_str or 'RateLimitError' in err_str
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    elif is_rate_limit:
                        print(f"[Agent] Rate limit exceeded after {max_retries} attempts.")
                        return {"role": "system", "content": "[Error] OpenAI API rate limit exceeded after retries. Please wait a minute and try again."}
                    else:
                        print(f"[Agent] Exception: {err_str}\n{traceback.format_exc()}")
                        return {"role": "system", "content": f"[Error] Agent failed: {err_str}\n{traceback.format_exc()}"}
            print(f"[Agent] User: {user_message}\n[Agent] Assistant: {response_content}")
            if save_to_db:
                db.add(ChatMessage(thread_id=thread_id, role="user", content=user_message))
                db.add(ChatMessage(thread_id=thread_id, role="assistant", content=response_content))
                db.commit()
            self.chat_threads[thread_id] = last_response.thread if last_response else thread
            return {"role": "assistant", "content": response_content}
        finally:
            db.close()


# Global explorer instance
explorer = StandaloneWebsiteExplorer()

# FastAPI app

app = FastAPI(title="Website Exploration API", version="1.0.0")
from fastapi import Request
# --- Chat endpoints for chat-based UI ---
class ChatThreadCreateRequest(BaseModel):
    name: str = None

class ChatMessageRequest(BaseModel):
    thread_id: str
    message: str

class ChatMessageResponse(BaseModel):
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    thread_id: str
    history: List[Dict[str, Any]]

@app.post("/chat/thread", response_model=dict)
async def create_chat_thread(request: ChatThreadCreateRequest):
    thread_id = explorer.create_chat_thread(request.name)
    return {"thread_id": thread_id}

@app.post("/chat/message", response_model=ChatMessageResponse)
async def chat_message(request: ChatMessageRequest):
    result = await explorer.chat_message(request.thread_id, request.message)
    return ChatMessageResponse(**result)

@app.get("/chat/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_chat_history(thread_id: str):
    history = explorer.get_chat_history(thread_id)
    return ChatHistoryResponse(thread_id=thread_id, history=history)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API compatibility endpoints for React frontend ---
@app.get("/api/threads")
async def api_get_threads():
    db = SessionLocal()
    try:
        threads = db.query(ChatThread).all()
        result = []
        for thread in threads:
            # Get first and last message timestamps if available
            if thread.messages:
                created_at = min(m.timestamp for m in thread.messages).isoformat()
                updated_at = max(m.timestamp for m in thread.messages).isoformat()
            else:
                created_at = updated_at = thread.created_at.isoformat()
            result.append({
                "id": thread.id,
                "name": thread.name or f"Exploration_{thread.id[:8]}",
                "created_at": created_at,
                "updated_at": updated_at
            })
        return {"success": True, "threads": result}
    finally:
        db.close()

@app.post("/api/threads")
async def api_create_thread():
    thread_id = explorer.create_chat_thread()
    return {"success": True, "thread_id": thread_id}

@app.get("/api/threads/{thread_id}/messages")
async def api_get_thread_messages(thread_id: str):
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).order_by(ChatMessage.timestamp).all()
        result = [
            {"id": m.id, "role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in messages
        ]
        return {"success": True, "messages": result}
    finally:
        db.close()

@app.delete("/api/threads/{thread_id}")
async def api_delete_thread(thread_id: str):
    db = SessionLocal()
    try:
        thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if thread:
            db.delete(thread)
            db.commit()
            explorer.stop_screenshot_loop(thread_id)
            return {"success": True}
        return {"success": False, "error": "Thread not found"}
    finally:
        db.close()

@app.post("/api/message")
async def api_post_message(data: dict):
    thread_id = data.get("thread_id")
    message = data.get("message")
    if not thread_id:
        thread_id = explorer.create_chat_thread()
    result = await explorer.chat_message(thread_id, message)
    return {"success": True, "response": result["content"], "thread_id": thread_id}

@app.post("/api/message/stream")
async def api_post_message_stream(request: StarletteRequest):
    from chatdb import SessionLocal, ChatMessage
    body = await request.json()
    thread_id = body.get("thread_id")
    message = body.get("message")
    if not thread_id:
        thread_id = explorer.create_chat_thread()
    db = SessionLocal()
    async def event_stream():
        # Save user message to DB
        db.add(ChatMessage(thread_id=thread_id, role="user", content=message))
        db.commit()
        result = await explorer.chat_message(thread_id, message, save_to_db=False)
        # If error, yield error and stop
        if isinstance(result, dict) and result.get("role") == "system" and result.get("content", "").startswith("[Error]"):
            print(f"[EventStream] Sending error to frontend: {result['content']}")
            yield f"data: {{\"error\": \"{result['content'].replace('\\', '\\\\').replace('"', '\\"')}\"}}\n"
            yield f"data: {{\"complete\": true}}\n"
            db.close()
            return
        content = result["content"]
        agent_response = ""
        for i in range(0, len(content), 50):
            chunk = content[i:i+50]
            agent_response += chunk
            print(f"[EventStream] Sending chunk to frontend: {chunk}")
            yield f"data: {{\"content\": \"{chunk.replace('\\', '\\\\').replace('"', '\\"')}\"}}\n"
        # Save agent response to DB
        db.add(ChatMessage(thread_id=thread_id, role="assistant", content=agent_response))
        db.commit()
        print("[EventStream] Sending complete to frontend.")
        yield f"data: {{\"complete\": true}}\n"
        db.close()
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.on_event("startup")
async def startup_event():
    """Initialize the explorer on startup"""
    success = await explorer.initialize()
    if not success:
        print("❌ Failed to initialize explorer")
        raise Exception("Failed to initialize explorer")
    print("✅ Explorer initialized successfully")

@app.get("/screenshot", response_model=ScreenshotResponse)
async def get_latest_screenshot():
    """Get the latest screenshot as base64"""
    try:
        if explorer.latest_screenshot:
            return ScreenshotResponse(
                screenshot_base64=explorer.latest_screenshot,
                timestamp=explorer.screenshot_timestamp,
                status="available"
            )
        else:
            return ScreenshotResponse(
                screenshot_base64=None,
                timestamp=None,
                status="no_screenshot_available"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get screenshot: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "explorer_initialized": explorer.agent is not None,
        "agent_ready": explorer.agent is not None,
        "active_tasks": len(explorer.active_tasks),
        "latest_screenshot_available": explorer.latest_screenshot is not None,
        "timestamp": datetime.now().isoformat()
    }

# --- WebSocket endpoint for real-time screenshot streaming ---
class ScreenshotWebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.last_sent_screenshot: str = ""

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                pass

screenshot_ws_manager = ScreenshotWebSocketManager()

@app.websocket("/ws/screenshots")
async def websocket_screenshots(websocket: WebSocket):
    await screenshot_ws_manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(2)
            # Send the latest screenshot if available and changed
            if explorer.latest_screenshot and explorer.latest_screenshot != screenshot_ws_manager.last_sent_screenshot:
                await websocket.send_json({
                    "type": "screenshot",
                    "data": explorer.latest_screenshot,
                    "timestamp": explorer.screenshot_timestamp
                })
                screenshot_ws_manager.last_sent_screenshot = explorer.latest_screenshot
            else:
                # Send a keepalive ping to prevent disconnect
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        screenshot_ws_manager.disconnect(websocket)
    except Exception:
        screenshot_ws_manager.disconnect(websocket)

if __name__ == "__main__":
    print("🚀 Starting Standalone Website Exploration API...")
    print("📋 Available endpoints:")
    print("  POST /start_exploration - Start exploration task")
    print("  GET /task_status/{task_id} - Get task status")
    print("  GET /screenshot - Get latest screenshot")
    print("  GET /active_tasks - List all tasks")
    print("  DELETE /task/{task_id} - Stop task")
    print("  GET /health - Health check")
    print()
    print("🌐 API will be available at: http://localhost:8001")
    print("📖 API docs will be available at: http://localhost:8001/docs")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
