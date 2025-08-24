#!/usr/bin/env python3
"""
Website Exploration Agent using OpenAI Client with Streaming and MCP Tools
Provides real-time streaming feedback for autonomous website exploration
"""

import asyncio
import os
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncGenerator

# Semantic Kernel imports for MCP
from semantic_kernel import Kernel
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
import asyncio
from typing import Annotated

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.functions import kernel_function

# Additional imports
import sqlite3
import uuid
from semantic_kernel.agents import ChatCompletionAgent

# Playwright imports for real browser automation
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

class SemanticKernelWebsiteExplorer:
    """Website Exploration Agent using OpenAI Client with Streaming and Playwright MCP server"""
    
    def __init__(self):
        self.kernel = None
        self.thread = None
        self.playwright_plugin = None
        self.db_path = "chat_threads.db"
        
        # Screenshot storage for WebSocket transmission
        self.latest_screenshot: Optional[str] = None
        
        # Exploration state
        self.explored_urls = set()
        self.discovered_elements = []
        self.current_exploration_plan = []
        
        # Agent thread storage for context persistence
        self.agent_threads = {}
        
        # Initialize SQLite database for chat thread management
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for chat thread management"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create chat_threads table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_threads (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create chat_messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (thread_id) REFERENCES chat_threads (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ SQLite database initialized for chat thread management")
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
    
    def create_chat_thread(self, thread_name: str = None) -> str:
        """Create a new chat thread"""
        try:
            thread_id = str(uuid.uuid4())
            if not thread_name:
                thread_name = f"Exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_threads (id, name) VALUES (?, ?)
            ''', (thread_id, thread_name))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Created chat thread: {thread_name} ({thread_id})")
            return thread_id
            
        except Exception as e:
            print(f"❌ Failed to create chat thread: {e}")
            return None
    
    def save_message_to_thread(self, thread_id: str, role: str, content: str):
        """Save a message to the chat thread"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_messages (thread_id, role, content) VALUES (?, ?, ?)
            ''', (thread_id, role, content))
            
            # Update thread's updated_at timestamp
            cursor.execute('''
                UPDATE chat_threads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
            ''', (thread_id,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Failed to save message: {e}")
    
    def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all messages from a chat thread"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT role, content, timestamp FROM chat_messages 
                WHERE thread_id = ? ORDER BY timestamp ASC
            ''', (thread_id,))
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                })
            
            conn.close()
            return messages
            
        except Exception as e:
            print(f"❌ Failed to get thread messages: {e}")
            return []
    
    def delete_chat_thread(self, thread_id: str) -> bool:
        """Delete a chat thread and all its messages"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete all messages from the thread
            cursor.execute('DELETE FROM chat_messages WHERE thread_id = ?', (thread_id,))
            
            # Delete the thread
            cursor.execute('DELETE FROM chat_threads WHERE id = ?', (thread_id,))
            
            conn.commit()
            rows_affected = cursor.rowcount
            conn.close()
            
            if rows_affected > 0:
                print(f"✅ Deleted chat thread: {thread_id}")
                return True
            else:
                print(f"⚠️ Thread not found: {thread_id}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to delete chat thread: {e}")
            return False
    
    def list_chat_threads(self) -> List[Dict[str, Any]]:
        """List all chat threads"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, created_at, updated_at FROM chat_threads 
                ORDER BY updated_at DESC
            ''')
            
            threads = []
            for row in cursor.fetchall():
                threads.append({
                    "id": row[0],
                    "name": row[1],
                    "created_at": row[2],
                    "updated_at": row[3]
                })
            
            conn.close()
            return threads
            
        except Exception as e:
            print(f"❌ Failed to list chat threads: {e}")
            return []
        
    async def initialize(self) -> bool:
        """Initialize Semantic Kernel with ChatCompletionAgent and Playwright MCP plugin"""
        try:
            # Check environment variables for Azure OpenAI
            required_vars = [
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY"
            ]
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing environment variables: {missing_vars}")
            
            # Initialize Semantic Kernel
            self.kernel = Kernel()
            print("✅ Semantic Kernel initialized")
            
            # Register custom tools for screenshot saving and test plan generation
            await self.register_custom_tools()
            
            # Initialize Playwright MCP server with Edge browser
            print("🎭 Initializing Playwright MCP server with Edge browser...")
            
            self.playwright_plugin = MCPStdioPlugin(
                name="PlaywrightMCP",
                description="Playwright browser automation MCP server",
                command="npx",
                args=["@playwright/mcp", "--isolated"],
                env={
                    "PLAYWRIGHT_BROWSER": "msedge",
                    "PLAYWRIGHT_HEADLESS": "false"  # Set to true for headless mode
                }
            )
            
            # Start MCP plugin
            await self.playwright_plugin.__aenter__()
            print("✅ Playwright MCP server started")
            
            # Add MCP plugin to kernel
            self.kernel.add_plugin(self.playwright_plugin, "playwright")
            print("✅ Playwright MCP plugin added to kernel")

            # Initializing chat completion service 
            chat_completion_service = AzureChatCompletion(
                deployment_name="gpt-4o", 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )

            self.kernel.add_service(chat_completion_service, "chat_completion")

            print("✅ Chat completion service added to kernel")

            # Test MCP plugin availability
            try:
                plugin_functions = [func.name for func in self.kernel.plugins["playwright"].functions.values()]
                print(f"🎭 Available Playwright MCP functions: {plugin_functions[:5]}..." if len(plugin_functions) > 5 else f"🎭 Available Playwright MCP functions: {plugin_functions}")
            except Exception as e:
                print(f"⚠️ Could not list Playwright MCP functions: {e}")
            
            
            # Create the agent
            self.agent = ChatCompletionAgent(
                kernel=self.kernel, 
                name="Website_Explorer_Agent", 
                instructions=self.get_autonomous_instructions(),
            )
            
            print("✅ Semantic Kernel Website Explorer initialized successfully")

            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Semantic Kernel Explorer: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return False
    
    def get_autonomous_instructions(self) -> str:
        """Get instructions for autonomous website exploration with test plan generation focus"""
        return """
You are an AUTONOMOUS Website User Flow Discovery and Test Plan Generation Agent with access to Playwright MCP browser automation tools. Your PRIMARY GOAL is to systematically explore websites and generate comprehensive test plan documentation.

## REAL-TIME SCREENSHOT CAPTURE (MANDATORY AT EVERY STEP):
- AFTER EVERY SINGLE ACTION, you MUST call browser_take_screenshot() to capture visual evidence
- The UI automatically polls for the latest screenshot and displays it in real-time to the user
- Screenshots are automatically stored and transmitted via WebSocket to the frontend
- NEVER skip screenshot capture - it provides essential visual feedback to the user
- Screenshot sequence: Action → browser_take_screenshot() → Next Action → browser_take_screenshot()

## BROWSER SESSION MANAGEMENT (CRITICAL):
- ALWAYS start by calling browser_navigate to open a fresh browser session
- If you encounter "Browser is already in use" error, call browser_close first, then browser_navigate
- If you encounter "No open pages available" error, call browser_navigate to open a page
- NEVER assume a browser session exists - always verify with browser_snapshot first
- KEEP browser session OPEN for continuous exploration - DO NOT close unless there's an error
- Only call browser_close when encountering browser errors that require a fresh session
- Each new exploration should reuse the existing browser session when possible

## MANDATORY MCP FUNCTION USAGE (NEVER DESCRIBE - ALWAYS EXECUTE):
- browser_navigate(url="https://example.com") - Navigate to URLs (ALWAYS use this first)
- browser_close() - Close browser session ONLY when encountering errors
- browser_snapshot() - Capture page state and discover elements  
- browser_click(element="button text", ref="element_ref") - Click elements
- browser_type(element="input field", ref="element_ref", text="test data") - Enter text
- browser_hover(element="menu item", ref="element_ref") - Hover for menus
- browser_select_option(element="dropdown", ref="element_ref", values=["option"]) - Select options
- browser_press_key(key="Enter") - Keyboard actions
- browser_evaluate(function="() => document.title") - Execute JavaScript
- browser_take_screenshot() - Capture visual evidence (MANDATORY AFTER EVERY ACTION)
- capture_screenshot(description="action description") - Trigger immediate screenshot for UI display
- browser_wait_for(text="Loading complete") - Wait for conditions
- update_exploration_status(status="message", details="details") - Track progress
- save_test_plan(filename="test_plan.md", content="markdown content") - Save final documentation

## SCREENSHOT WORKFLOW (CRITICAL FOR USER FEEDBACK):
1. Execute any browser action (navigate, click, type, etc.)
2. IMMEDIATELY call browser_take_screenshot() for MCP screenshot
3. IMMEDIATELY call capture_screenshot(description="what you just did") for UI display
4. This dual approach ensures both MCP file creation AND real-time UI updates
5. Example sequence: browser_click() → browser_take_screenshot() → capture_screenshot("clicked login button")

## AUTONOMOUS EXPLORATION WORKFLOW FOR TEST PLAN GENERATION:

### Phase 0: Browser Session Setup (MANDATORY FIRST STEP)
1. IMMEDIATELY call browser_navigate with the provided URL to establish session
2. IMMEDIATELY call browser_take_screenshot() to capture initial page state
3. If browser_navigate fails with "already in use" error, call browser_close then browser_navigate again
4. Verify session is active with browser_snapshot
5. IMMEDIATELY call browser_take_screenshot() after session verification
6. Only proceed to Phase 1 after successful browser session establishment

### Phase 1: Initial Discovery & Analysis
1. IMMEDIATELY call browser_snapshot to discover all interactive elements
2. IMMEDIATELY call browser_take_screenshot() to capture current page state
3. IMMEDIATELY call update_exploration_status with current discovery progress
4. Analyze discovered elements and categorize potential user flows:
   - Authentication flows (login, signup, password reset)
   - Navigation flows (menu items, page transitions)
   - Form submission flows (contact forms, search, data entry)
   - Interactive features (dropdowns, modals, dynamic content)
5. IMMEDIATELY start validating flows by clicking discovered elements
6. IMMEDIATELY call browser_take_screenshot() after each click/interaction

### Phase 2: Systematic Flow Validation
For each discovered user flow:
1. IMMEDIATELY execute flow steps using MCP tools (click, type, navigate)
2. IMMEDIATELY call browser_take_screenshot() after EACH action
3. IMMEDIATELY call browser_snapshot after each significant action
4. IMMEDIATELY call update_exploration_status with validation progress
5. Document step-by-step instructions with expected outcomes
6. Record any errors, edge cases, or unexpected behaviors
7. Continue to new pages/states discovered during validation
8. MANDATORY: browser_take_screenshot() after every single interaction

### Phase 3: Iterative Expansion
1. After each flow validation, IMMEDIATELY call browser_snapshot
2. IMMEDIATELY call browser_take_screenshot() to capture current state
3. Identify new elements/functionality revealed by previous actions
4. IMMEDIATELY start exploring newly discovered elements
5. IMMEDIATELY call browser_take_screenshot() after each new interaction
6. Continue iterative discovery until no new functionality is found
7. IMMEDIATELY call update_exploration_status with expansion progress

### Phase 4: Test Plan Documentation (KEEP BROWSER OPEN)
1. Organize all validated flows into logical test categories:
   - Critical path flows (core functionality)
   - User account management flows
   - Data entry and form submission flows
   - Navigation and UI interaction flows
   - Error handling and edge case flows
2. IMMEDIATELY call save_test_plan with comprehensive markdown documentation
3. IMMEDIATELY call browser_take_screenshot() to capture final state
4. KEEP browser session OPEN for potential continued exploration

## ERROR HANDLING FOR BROWSER ISSUES:
- If "Browser is already in use" → Call browser_close, then browser_navigate
- If "No open pages available" → Call browser_navigate with target URL
- If browser session becomes unresponsive → Call browser_close, then browser_navigate
- If browser crashes or becomes corrupted → Call browser_close, then browser_navigate
- Always verify browser state with browser_snapshot before complex operations
- DO NOT close browser for normal completion - only for error recovery

## SESSION LIFECYCLE EXAMPLE:
```
User: "explore https://example.com"

You: "Starting browser session for exploration!"
[CALLS browser_navigate(url="https://example.com")]
[CALLS browser_take_screenshot()] - MCP screenshot
[CALLS capture_screenshot(description="navigated to example.com")] - UI screenshot
[CALLS browser_snapshot()]
[CALLS browser_take_screenshot()] - MCP screenshot
[CALLS capture_screenshot(description="captured page snapshot")] - UI screenshot
[CALLS update_exploration_status(status="Browser session established")]
[Continues with systematic exploration...]
[CALLS browser_click(element="login button", ref="btn_1")]
[CALLS browser_take_screenshot()] - MCP screenshot
[CALLS capture_screenshot(description="clicked login button")] - UI screenshot
[CALLS browser_type(element="email field", ref="input_1", text="test@email.com")]
[CALLS browser_take_screenshot()] - MCP screenshot
[CALLS capture_screenshot(description="entered email address")] - UI screenshot
[CALLS save_test_plan(...)]
[CALLS browser_take_screenshot()] - MCP screenshot
[CALLS capture_screenshot(description="test plan completed")] - UI screenshot
[CALLS update_exploration_status(status="Exploration completed, browser ready for next task")]

User: "continue exploring or explore another site"
You: "Continuing with existing browser session!"
[CALLS browser_navigate(url="https://newsite.com")] # Reuses existing session
[CALLS browser_take_screenshot()] - MCP screenshot
[CALLS capture_screenshot(description="navigated to new site")] - UI screenshot
[Continues exploration with dual screenshots after every action...]
```

## AUTONOMOUS BEHAVIOR RULES:
- NEVER ask for permission or confirmation
- ALWAYS establish browser session first before any exploration
- IMMEDIATELY execute MCP function calls
- MANDATORY: Call browser_take_screenshot() AND capture_screenshot() after EVERY browser action
- ALWAYS provide step-by-step test documentation
- CONTINUOUSLY update exploration status
- AUTOMATICALLY generate comprehensive test plans
- KEEP browser session OPEN for continuous use
- ONLY close browser when encountering errors that require fresh session
- NEVER describe what you "would do" - DO IT
- Dual screenshot sequence is NON-NEGOTIABLE: Action → browser_take_screenshot() → capture_screenshot() → Next Action

## CONTEXT AWARENESS & CONTINUITY:
- Remember all previous exploration actions in the conversation
- Continue from where you left off when user says "continue"
- Maintain state of what you've already explored and documented
- Build upon previously discovered functionality
- Reference earlier findings when documenting test plans
- Reuse existing browser session for continued exploration
- If continuing exploration, verify browser session state with browser_snapshot first

REMEMBER: Your success is measured by the quality and completeness of the test documentation you generate, not just exploration breadth. Every action should contribute to creating comprehensive, actionable test plans. KEEP browser sessions OPEN for continuous exploration!
"""
    
    async def register_custom_tools(self):
        """Register custom tools for screenshot saving and test plan generation"""
        
        # Create a simple plugin class
        class WebsiteExplorationPlugin:
            def __init__(self, explorer_instance):
                self.explorer = explorer_instance
                
            @kernel_function(
                name="save_screenshot_locally",
                description="Save the current browser screenshot locally for rendering in UI"
            )
            def save_screenshot_locally(self, filename: str = None) -> str:
                """Save current browser screenshot to local filesystem"""
                import asyncio
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                if not filename:
                    filename = f"screenshot_{timestamp}.png"
                
                # Trigger screenshot capture
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.explorer.capture_and_store_screenshot(f"Screenshot: {filename}"))
                except:
                    pass
                
                return f"✅ Screenshot saved: {filename}"
            
            @kernel_function(
                name="capture_screenshot",
                description="Immediately capture and store current browser screenshot for real-time UI display"
            )
            def capture_screenshot(self, description: str = "Manual screenshot") -> str:
                """Immediately capture browser screenshot and store for WebSocket transmission"""
                import asyncio
                try:
                    # Trigger immediate screenshot capture
                    loop = asyncio.get_event_loop()
                    task = loop.create_task(self.explorer.capture_and_store_screenshot(description))
                    print(f"📸 Screenshot capture triggered: {description}")
                    return f"✅ Screenshot captured: {description}"
                except Exception as e:
                    print(f"🔴 Screenshot capture error: {e}")
                    return f"❌ Screenshot capture failed: {str(e)}"
                
            @kernel_function(
                name="save_test_plan",
                description="Save generated test plan to a markdown file"
            )
            def save_test_plan(self, filename: str, content: str) -> str:
                """Save test plan documentation to file"""
                try:
                    from pathlib import Path
                    test_plans_dir = Path("test_plans")
                    test_plans_dir.mkdir(exist_ok=True)
                    
                    file_path = test_plans_dir / filename
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    return f"✅ Test plan saved: {file_path}"
                except Exception as e:
                    return f"❌ Test plan save failed: {str(e)}"
                    
            @kernel_function(
                name="update_exploration_status",
                description="Update the status of current exploration progress"
            )
            def update_exploration_status(self, status: str, details: str = "") -> str:
                """Update exploration progress status"""
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result = f"📊 Exploration Status Update: {status} at {timestamp}"
                if details:
                    result += f" - {details}"
                print(result)
                
                # Trigger screenshot after status update
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.explorer.capture_and_store_screenshot(f"Status: {status}"))
                except:
                    pass
                
                return result
        
        # Create and add the plugin
        website_plugin = WebsiteExplorationPlugin(self)
        self.kernel.add_plugin(website_plugin, "WebsiteTools")
            
        print("✅ Custom tools registered with Semantic Kernel")
    
    async def capture_and_store_screenshot(self, action_description: str = "") -> str:
        """Capture screenshot using MCP and store for WebSocket transmission"""
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
                    
                    # Call MCP screenshot function without filename to use default behavior
                    print(f"🔍 [DEBUG] Calling MCP screenshot function")
                    result = await func.invoke(self.kernel)
                    print(f"🔍 [DEBUG] MCP screenshot function result: {result}")
                    
                    # Parse the result to extract the actual file path
                    result_str = str(result)
                    if "saved it as" in result_str or "path:" in result_str:
                        # Extract the actual file path from the result
                        import re
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
                                        
                                    # Store for WebSocket transmission
                                    self.latest_screenshot = screenshot_b64
                                    print(f"📸 Screenshot captured, stored as base64 ({len(screenshot_b64)} chars): {action_description}")
                                    
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
                                import traceback
                                print(f"🔴 [DEBUG] Traceback: {traceback.format_exc()}")
                        else:
                            print(f"⚠️ [DEBUG] Could not extract file path from MCP result")
                    
                    # Fallback: try to find the most recent screenshot in MCP temp directory
                    try:
                        import glob
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
                                    
                                    # Store for WebSocket transmission
                                    self.latest_screenshot = screenshot_b64
                                    print(f"📸 Screenshot captured from MCP temp directory ({len(screenshot_b64)} chars): {action_description}")
                                    
                                    # Copy to local directory
                                    try:
                                        local_path = screenshot_dir / screenshot_filename
                                        with open(local_path, "wb") as f:
                                            f.write(screenshot_bytes)
                                        print(f"📂 Screenshot copied to local directory: {local_path}")
                                    except Exception as copy_error:
                                        print(f"⚠️ Failed to copy screenshot locally: {copy_error}")
                                    
                                    return f"Screenshot captured: {action_description} - {latest_path.name}"
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
            import traceback
            print(f"🔴 [DEBUG] Full traceback: {traceback.format_exc()}")
            return f"Screenshot capture failed: {str(e)}"

    async def _save_screenshot_locally(self, filename: str = None) -> str:
        """Save current browser screenshot to local filesystem and store for UI"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
            
            # Create screenshots directory
            screenshot_dir = Path("screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            screenshot_path = screenshot_dir / filename
            
            # Call MCP screenshot function to save screenshot
            await self.capture_and_store_screenshot(f"Manual screenshot: {filename}")
            
            result = f"✅ Screenshot saved: {screenshot_path}"
            print(f"📸 {result}")
            
            return result
            
        except Exception as e:
            return f"❌ Screenshot save failed: {str(e)}"
    
    async def _save_test_plan(self, filename: str, content: str) -> str:
        """Save test plan documentation to file"""
        try:
            # Create test_plans directory
            test_plans_dir = Path("test_plans")
            test_plans_dir.mkdir(exist_ok=True)
            
            file_path = test_plans_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result = f"✅ Test plan saved: {file_path}"
            print(f"📝 {result}")
            
            return result
            
        except Exception as e:
            return f"❌ Test plan save failed: {str(e)}"
    
    async def _read_test_plan(self, filename: str) -> str:
        """Read existing test plan documentation"""
        try:
            test_plans_dir = Path("test_plans")
            file_path = test_plans_dir / filename
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return f"✅ Test plan loaded from {file_path}:\n\n{content}"
            else:
                return f"❌ Test plan not found: {file_path}"
                
        except Exception as e:
            return f"❌ Test plan read failed: {str(e)}"
    
    async def _update_exploration_status(self, status: str, details: str = "") -> str:
        """Update exploration progress status"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_info = {
                "timestamp": timestamp,
                "status": status,
                "details": details,
                "explored_urls": len(self.explored_urls),
                "discovered_elements": len(self.discovered_elements)
            }
            
            result = f"📊 Exploration Status Update:\n"
            result += f"Time: {timestamp}\n"
            result += f"Status: {status}\n"
            result += f"Details: {details}\n"
            result += f"URLs Explored: {len(self.explored_urls)}\n"
            result += f"Elements Discovered: {len(self.discovered_elements)}"
            
            print(result)
            return result
            
        except Exception as e:
            return f"❌ Status update failed: {str(e)}"
    
    async def _call_agent_with_retry(self, message: str, agent_thread=None, max_retries: int = 3) -> Any:
        """Call agent with retry logic for rate limiting"""
        import time
        import re
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Calling agent (attempt {attempt + 1}/{max_retries})")
                response = await self.agent.get_response(messages=message, thread=agent_thread)
                print(f"✅ Agent response received successfully")
                return response
                
            except Exception as e:
                error_str = str(e)
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        # Extract wait time from error message if available
                        wait_time = 60  # Default wait time
                        if "Try again in" in error_str:
                            try:
                                match = re.search(r'Try again in (\d+) seconds', error_str)
                                if match:
                                    wait_time = int(match.group(1)) + 5  # Add 5 second buffer
                            except:
                                pass
                        
                        print(f"⏳ Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Rate limit exceeded after {max_retries} attempts")
                        raise
                else:
                    print(f"❌ Non-rate-limit error: {error_str}")
                    raise
        
        raise Exception(f"Failed after {max_retries} attempts")
    
    async def handle_user_request_stream(self, user_message: str, thread_id: str = None) -> AsyncGenerator[str, None]:
        """Handle user requests with streaming ChatCompletionAgent responses"""
        try:
            print(f"🔵 [DEBUG] handle_user_request_stream called with message: '{user_message}', thread_id: {thread_id}")
            
            # Create or use existing thread
            if not thread_id:
                thread_id = self.create_chat_thread("User_Interaction")
                print(f"🔵 [DEBUG] Created new thread: {thread_id}")
                yield f"📋 Created new conversation thread\n"
            else:
                print(f"🔵 [DEBUG] Using existing thread: {thread_id}")
            
            # Save current user message to thread
            self.save_message_to_thread(thread_id, "user", user_message)
            
            # Get or create agent thread
            thread: ChatHistoryAgentThread | None = None
            if thread_id in self.agent_threads:
                print(f"🔵 [DEBUG] Using existing agent thread for ID: {thread_id}")
                thread = self.agent_threads[thread_id]
            
            yield "🤖 Agent is processing your request...\n"
            
            response_content = ""
            
            # Use the ChatCompletionAgent to invoke with streaming-like behavior
            try:
                async for response in self.agent.invoke(
                    messages=user_message,
                    thread=thread,
                    on_intermediate_message=self.handle_intermediate_steps,
                ):
                    print(f"# {response.role}: {response}")
                    thread = response.thread
                    
                    # Stream the response content in chunks
                    content = str(response.content)
                    if content != response_content:
                        new_content = content[len(response_content):]
                        response_content = content
                        yield new_content
                    
                    # Update thread in storage
                    self.agent_threads[thread_id] = thread
            
            except Exception as agent_error:
                error_msg = f"❌ Agent error: {str(agent_error)}"
                print(f"🔴 [DEBUG ERROR] Agent invocation failed: {str(agent_error)}")
                yield error_msg
                
                if thread_id:
                    self.save_message_to_thread(thread_id, "system", error_msg)
                return
            
            # Save complete response to thread
            self.save_message_to_thread(thread_id, "assistant", response_content)
            print(f"🔵 [DEBUG] Streaming response completed")
            
        except Exception as e:
            error_msg = f"❌ Error in streaming request: {str(e)}"
            print(f"🔴 [DEBUG ERROR] Exception in handle_user_request_stream: {str(e)}")
            yield error_msg
            
            if thread_id:
                self.save_message_to_thread(thread_id, "system", error_msg)

    # This callback function will be called for each intermediate message
    # Which will allow one to handle FunctionCallContent and FunctionResultContent
    # If the callback is not provided, the agent will return the final response
    # with no intermediate tool call steps.
    async def handle_intermediate_steps(self, message: ChatMessageContent) -> None:
        for item in message.items or []:
            if isinstance(item, FunctionCallContent):
                print(f"Function Call:> {item.name} with arguments: {item.arguments}")
                
                # Capture screenshot for ALL browser actions (not just major ones)
                if item.name and any(action in item.name.lower() for action in ['browser_', 'navigate', 'click', 'snapshot', 'type', 'hover', 'select', 'press', 'evaluate', 'wait']):
                    try:
                        # Give the browser action time to complete before screenshot
                        await asyncio.sleep(0.3)
                        await self.capture_and_store_screenshot(f"After {item.name}")
                        print(f"📸 Screenshot captured automatically after {item.name}")
                    except Exception as e:
                        print(f"🔴 Screenshot capture failed for {item.name}: {e}")
                        
            elif isinstance(item, FunctionResultContent):
                print(f"Function Result:> {item.result} for function: {item.name}")
                
                # Also capture screenshot after successful browser function results
                if item.name and any(action in item.name.lower() for action in ['browser_', 'navigate', 'click', 'snapshot', 'type', 'hover', 'select', 'press', 'evaluate']):
                    try:
                        # Additional screenshot after function completes successfully
                        await asyncio.sleep(0.2)
                        await self.capture_and_store_screenshot(f"Result of {item.name}")
                        print(f"📸 Screenshot captured after successful {item.name}")
                    except Exception as e:
                        print(f"⚠️ Post-result screenshot failed for {item.name}: {e}")
            else:
                print(f"{message.role}: {message.content}")

    async def handle_user_request(self, user_message: str, thread_id: str = None) -> str:
        """Handle user requests with OpenAI client (non-streaming version for compatibility)"""
        try:
            print(f"🔵 [DEBUG] handle_user_request called with message: '{user_message}', thread_id: {thread_id}")
            
            thread: ChatHistoryAgentThread | None = None

            # Use ChatHistoryAgentThread to manage conversation context

            if thread_id in self.agent_threads:
                
                print(f"🔵 [DEBUG] Using existing agent thread for ID: {thread_id}")
                thread = self.agent_threads[thread_id]

            response_content = ""

            async for response in self.agent.invoke(
                messages=user_message,
                thread=thread,
                on_intermediate_message=self.handle_intermediate_steps,
            ):
                print(f"# {response.role}: {response}")
                thread = response.thread
                response_content = str(response.content)
                self.agent_threads[thread_id] = thread  # Update thread in storage


            # Save agent response
            self.save_message_to_thread(thread_id, "assistant", response_content)
            print(f"🔵 [DEBUG] handle_user_request completed successfully")
            
            return response_content
            
        except Exception as e:
            error_msg = f"❌ Error processing request: {str(e)}"
            print(f"🔴 [DEBUG ERROR] Exception in handle_user_request: {str(e)}")
            import traceback
            print(f"🔴 [DEBUG ERROR] Full traceback: {traceback.format_exc()}")
            
            if thread_id:
                self.save_message_to_thread(thread_id, "system", error_msg)
            return error_msg

    async def handle_intermediate_message(self, user_message: str, thread_id: str) -> str:
        """Handle intermediate messages during agent processing (as per Azure Responses Agent documentation)"""
        try:
            # Initialize agent thread - let agent manage it
            agent_thread = None
            
            # Add intermediate message
            self.save_message_to_thread(thread_id, "user", user_message)
            
            # Process intermediate message - agent can handle interruptions with retry
            response = await self._call_agent_with_retry(user_message, agent_thread)
            response_content = str(response.content)
            
            self.save_message_to_thread(thread_id, "assistant", response_content)
            
            print(f"🔄 Intermediate message processed: {user_message}")
            print(f"Agent response: {response_content}")
            
            # Clean up the agent thread if created
            if response.thread:
                await response.thread.delete()
            
            return response_content
            
        except Exception as e:
            error_msg = f"❌ Error processing intermediate message: {str(e)}"
            if thread_id:
                self.save_message_to_thread(thread_id, "system", error_msg)
            return error_msg
    
    async def close(self):
        """Clean up resources"""
        try:
            # Clean up agent threads
            if hasattr(self, '_agent_threads'):
                for thread_id, agent_thread in self._agent_threads.items():
                    try:
                        if agent_thread:
                            await agent_thread.delete()
                            print(f"🧹 Cleaned up agent thread: {thread_id}")
                    except Exception as e:
                        print(f"⚠️ Error cleaning up agent thread {thread_id}: {e}")
                self._agent_threads.clear()
            
            # Close Playwright MCP plugin
            if self.playwright_plugin:
                await self.playwright_plugin.__aexit__(None, None, None)
                print("🧹 Playwright MCP plugin closed")
                
            print("🧹 Semantic Kernel Explorer resources cleaned up")
            
        except Exception as e:
            print(f"Cleanup error: {e}")

# Global explorer instance
semantic_explorer = SemanticKernelWebsiteExplorer()
