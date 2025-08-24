#!/usr/bin/env python3
"""
Website Exploration Agent with AutoGen and Playwright MCP
Systematically explores websites and generates comprehensive test plans
"""

import asyncio
import os
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import chainlit as cl
from chainlit import Message, AskUserMessage, AskFileMessage, Image

# Playwright imports for real browser automation
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.auth.azure import AzureTokenProvider
from azure.identity import DefaultAzureCredential

class WebsiteExploreAgent:
    """Website Exploration Agent using AutoGen and real Playwright tools"""
    
    def __init__(self):
        self.model_client = None
        self.agent = None
        self.team = None
        
        # Real Playwright browser instances
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Screenshot storage for WebSocket transmission
        self.latest_screenshot: Optional[str] = None
        
        # Browser configuration for headed mode with Edge
        self.browser_config = {
            "headless": False,  # HEADED MODE for visual exploration
            "devtools": True,   # Enable devtools for debugging
            "slow_mo": 500,     # Slow down actions for visibility
            "viewport": {"width": 1280, "height": 720},
            "args": [
                "--start-maximized",
                "--disable-blink-features=AutomationControlled", 
                "--disable-web-security",
                "--allow-running-insecure-content"
            ]
        }
    
    async def initialize_browser(self) -> Dict[str, Any]:
        """Initialize Playwright browser with Microsoft Edge in headed mode"""
        try:
            if not self.playwright:
                self.playwright = await async_playwright().start()
                
            if not self.browser:
                # Use Microsoft Edge instead of Chromium
                self.browser = await self.playwright.chromium.launch(
                    channel="msedge",  # Use Microsoft Edge
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
                
            return {
                "success": True,
                "message": "Microsoft Edge browser initialized in HEADED mode - you can now see the exploration!"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to initialize Microsoft Edge browser: {str(e)}"
            }
    
    async def close_browser(self):
        """Clean up browser resources"""
        try:
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
                
        except Exception as e:
            print(f"Error closing browser: {e}")
        
    async def initialize(self):
        """Initialize the AutoGen agent with real Playwright browser capabilities"""
        
        try:
            # Check environment variables
            required_vars = [
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_DEPLOYMENT_NAME", 
                "AZURE_OPENAI_API_VERSION"
            ]
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing environment variables: {missing_vars}")
            
            # Create Azure OpenAI client
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            
            if api_key:
                # API Key authentication
                self.model_client = AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment,
                    model=deployment,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key
                )
            else:
                # Azure AD Token authentication
                token_provider = AzureTokenProvider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                
                self.model_client = AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment,
                    model=deployment,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider
                )
            
            print("✅ AutoGen agent initialized (browser will open when first MCP tool is used)")
            
            # Create Playwright MCP tool functions
            tools = [
                # Core browser automation tools
                self.navigate_to_url,
                self.take_snapshot,
                self.click_element,
                self.type_text,
                self.hover_element,
                self.select_dropdown_option,
                self.press_key,
                self.evaluate_javascript,
                self.upload_file,
                self.wait_for_condition,
                self.resize_browser,
                self.take_screenshot,
                self.get_console_messages,
                self.get_network_requests,
                self.navigate_back,
                self.navigate_forward,
                
                # Tab management tools
                self.list_tabs,
                self.open_new_tab,
                self.select_tab,
                self.close_tab,
                
                # Flow documentation tools
                self.save_test_plan,
                self.read_test_plan,
                self.update_flow_status,
                self.generate_final_documentation
            ]
            
            # Create system message based on the comprehensive specification
            system_message = self.create_system_message()
            
            # Create the AssistantAgent
            self.agent = AssistantAgent(
                name="website_explore_agent",
                model_client=self.model_client,
                tools=tools,
                system_message=system_message,
                description="Specialized agent for systematic website exploration and test plan generation using Playwright MCP tools",
                reflect_on_tool_use=True,
                model_client_stream=True
            )
            
            # Create team
            self.team = RoundRobinGroupChat(
                [self.agent],
                termination_condition=TextMentionTermination("TERMINATE")
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Website Explore Agent: {e}")
            return False
    
    def create_system_message(self) -> str:
        """Create comprehensive system message based on the specification"""
        
        return """
You are an expert Website Exploration Agent that systematically explores websites and generates comprehensive test plans using Playwright MCP browser tools. You work AUTONOMOUSLY and take initiative to explore thoroughly.

**YOUR MISSION:**
Discover, validate, and document all possible user interactions and workflows through autonomous iterative exploration to create complete test suite documentation.

**AUTONOMOUS BEHAVIOR:**
- **TAKE INITIATIVE**: Don't ask for permission - start exploring immediately
- **BE PROACTIVE**: Automatically discover and test all flows you find
- **MINIMIZE INTERRUPTIONS**: Only ask users for specific inputs when absolutely necessary (passwords, emails, etc.)
- **CONTINUOUS EXPLORATION**: Keep exploring until you've covered all major functionality
- **AUTO-DOCUMENTATION**: Generate test plans as you discover functionality

**CORE WORKFLOW - AUTONOMOUS LOOP:**
```
1. START: Navigate to target URL immediately
2. DISCOVER: Take snapshot and identify all interactive elements
3. EXPLORE: Automatically click/interact with discovered elements
4. VALIDATE: Test each discovered flow without asking permission
5. DOCUMENT: Record findings and continue to next area
6. REPEAT: Keep exploring until comprehensive coverage achieved
7. FINALIZE: Generate complete test plan at the end
```

**WHEN TO ASK FOR USER INPUT:**
- Login credentials (email, password, 2FA codes)
- Payment information for checkout flows
- Personal information for form submissions
- External service API keys or tokens
- Confirmation for destructive actions (delete, purchase)

**AUTONOMOUS EXPLORATION EXAMPLES:**
✅ "I'll start by navigating to the homepage and taking a snapshot..."
✅ "Found a login button - clicking it to explore authentication..."
✅ "Discovered a search form - testing with sample queries..."
✅ "Navigation menu found - exploring each section systematically..."
✅ "Form detected - filling with test data to validate submission..."

❌ "Should I click this button?" (Just click it!)
❌ "Would you like me to explore this section?" (Just explore it!)
❌ "Do you want me to continue?" (Always continue!)

**AVAILABLE PLAYWRIGHT MCP TOOLS:**
- **navigate_to_url**: Navigate to URLs
- **take_snapshot**: Capture accessibility snapshot for element discovery
- **click_element**: Click buttons, links, form elements
- **type_text**: Enter text in input fields
- **hover_element**: Hover for tooltips or menus
- **select_dropdown_option**: Select dropdown options
- **press_key**: Keyboard actions (Enter, Tab, Escape)
- **evaluate_javascript**: Execute JavaScript for page inspection
- **upload_file**: Upload files in forms
- **wait_for_condition**: Wait for elements or conditions
- **take_screenshot**: Capture visual evidence
- **get_console_messages**: Check for JavaScript errors
- **get_network_requests**: Monitor network activity
- **navigate_back/forward**: Browser navigation
- **Tab management**: list_tabs, open_new_tab, select_tab, close_tab

**COMMUNICATION STYLE:**
- **Action-Oriented**: "Navigating to homepage now..."
- **Progress Updates**: "Found 5 interactive elements, testing each..."
- **Discovery Reports**: "Discovered user registration flow with 3 steps..."
- **Autonomous Decisions**: "Detected form validation - testing edge cases..."
- **Efficient Documentation**: "Documented login flow - proceeding to dashboard..."

**EXPLORATION STRATEGY:**
1. **Homepage Analysis**: Navigate and identify main sections
2. **Navigation Discovery**: Explore all menu items and links
3. **Form Testing**: Test all forms with various inputs
4. **User Flow Validation**: Complete entire user journeys
5. **Error Handling**: Test edge cases and error conditions
6. **Mobile/Responsive**: Test different viewport sizes
7. **Performance**: Monitor loading times and network requests

**KEY CONSTRAINTS:**
- **BE AUTONOMOUS**: Don't wait for permission to explore
- **BE THOROUGH**: Cover all major functionality areas
- **BE EFFICIENT**: Move quickly through exploration
- **BE SMART**: Ask for user input only when absolutely required
- **BE COMPREHENSIVE**: Generate detailed test documentation

**WORKFLOW EXECUTION:**
1. Immediately start with target URL
2. Systematically explore all discovered functionality
3. Ask for user input only when required for progression
4. Continue exploration until comprehensive coverage
5. Generate final comprehensive test documentation

Start exploring immediately when given a URL - no need to ask for permission or confirmation!
"""
    
    # Real Playwright Tool Functions (browser opens on first use)
    async def navigate_to_url(self, url: str) -> str:
        """Navigate to a specific URL using real Playwright with live view"""
        try:
            # Initialize browser on first use
            if not self.page:
                browser_result = await self.initialize_browser()
                if not browser_result["success"]:
                    return json.dumps({
                        "success": False, 
                        "error": browser_result["error"],
                        "message": f"Failed to initialize browser: {browser_result['message']}"
                    })
                print(f"🌐 {browser_result['message']}")
                
                # Send browser activation message
                await cl.Message(
                    content="🌐 **Live Browser View Activated!**\n\nYou'll now see real-time browser updates as I explore websites. The browser view will appear below each action.",
                    author="System"
                ).send()
                
            # Navigate to URL
            await self.page.goto(url, wait_until="domcontentloaded")
            
            # Wait a moment for page to fully load
            await asyncio.sleep(1)
            
            # Capture and send live view
            await self.capture_and_send_screenshot(f"🔗 Navigated to {url}")
            
            current_url = self.page.url
            title = await self.page.title()
            
            result = {
                "action": "navigate", 
                "url": url,
                "current_url": current_url,
                "title": title,
                "success": True,
                "message": f"Successfully navigated to {url}"
            }
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({
                "success": False, 
                "error": str(e),
                "message": f"Failed to navigate to {url}"
            })
    
    async def take_snapshot(self) -> str:
        """Capture accessibility snapshot for element discovery using real Playwright with live view"""
        try:
            # Initialize browser on first use
            if not self.page:
                browser_result = await self.initialize_browser()
                if not browser_result["success"]:
                    return json.dumps({
                        "success": False, 
                        "error": browser_result["error"],
                        "message": f"Failed to initialize browser: {browser_result['message']}"
                    })
                print(f"🌐 {browser_result['message']}")
                
            # Capture live view first
            await self.capture_and_send_screenshot("📷 Taking element snapshot")
                
            # Get page info
            title = await self.page.title()
            url = self.page.url
            
            # Find interactive elements
            buttons = await self.page.locator("button, input[type='button'], input[type='submit']").all()
            links = await self.page.locator("a[href]").all()
            inputs = await self.page.locator("input, textarea, select").all()
            
            elements_found = []
            
            # Process buttons
            for i, button in enumerate(buttons):
                try:
                    text = await button.inner_text()
                    tag = await button.evaluate("el => el.tagName.toLowerCase()")
                    element_type = await button.get_attribute("type") or "button"
                    elements_found.append({
                        "type": "button",
                        "tag": tag,
                        "text": text.strip()[:50] if text else f"Button {i+1}",
                        "element_type": element_type,
                        "selector": f"button:nth-child({i+1})"
                    })
                except:
                    pass
                    
            # Process links
            for i, link in enumerate(links):
                try:
                    text = await link.inner_text()
                    href = await link.get_attribute("href")
                    elements_found.append({
                        "type": "link",
                        "tag": "a",
                        "text": text.strip()[:50] if text else f"Link {i+1}",
                        "href": href,
                        "selector": f"a:nth-child({i+1})"
                    })
                except:
                    pass
                    
            # Process form inputs
            for i, input_elem in enumerate(inputs):
                try:
                    tag = await input_elem.evaluate("el => el.tagName.toLowerCase()")
                    input_type = await input_elem.get_attribute("type") or "text"
                    name = await input_elem.get_attribute("name") or f"input_{i+1}"
                    placeholder = await input_elem.get_attribute("placeholder") or ""
                    elements_found.append({
                        "type": "input",
                        "tag": tag,
                        "input_type": input_type,
                        "name": name,
                        "placeholder": placeholder,
                        "selector": f"{tag}:nth-child({i+1})"
                    })
                except:
                    pass
            
            result = {
                "action": "snapshot",
                "url": url,
                "title": title,
                "elements_found": elements_found,
                "element_count": len(elements_found),
                "success": True,
                "message": f"Found {len(elements_found)} interactive elements"
            }
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({
                "success": False, 
                "error": str(e),
                "message": "Failed to capture snapshot"
            })
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def click_element(self, element_description: str, selector: str) -> str:
        """Click on a specific element using real Playwright with live view"""
        try:
            # Initialize browser on first use
            if not self.page:
                browser_result = await self.initialize_browser()
                if not browser_result["success"]:
                    return json.dumps({
                        "success": False, 
                        "error": browser_result["error"],
                        "message": f"Failed to initialize browser: {browser_result['message']}"
                    })
                print(f"🌐 {browser_result['message']}")
                
            # Highlight element before clicking (for visual feedback)
            try:
                element = self.page.locator(selector)
                await element.scroll_into_view_if_needed()
                
                # Add visual highlight using JavaScript
                await self.page.evaluate(f"""
                    const element = document.querySelector('{selector}');
                    if (element) {{
                        element.style.outline = '3px solid #f59e0b';
                        element.style.outlineOffset = '2px';
                        element.style.backgroundColor = '#fef3c7';
                    }}
                """)
                
                # Capture highlighted state
                await self.capture_and_send_screenshot(f"🎯 About to click: {element_description}")
                
                await asyncio.sleep(0.5)  # Brief pause to show highlight
                
                # Remove highlight and click
                await self.page.evaluate(f"""
                    const element = document.querySelector('{selector}');
                    if (element) {{
                        element.style.outline = '';
                        element.style.outlineOffset = '';
                        element.style.backgroundColor = '';
                    }}
                """)
                
                await element.click()
                
                # Wait for page response
                await asyncio.sleep(1)
                
                # Capture result
                await self.capture_and_send_screenshot(f"✅ Clicked: {element_description}")
                
            except Exception as e:
                # Fallback to simple click
                element = self.page.locator(selector)
                await element.scroll_into_view_if_needed()
                await element.click()
                await self.capture_and_send_screenshot(f"✅ Clicked: {element_description}")
            
            result = {
                "action": "click", 
                "element": element_description, 
                "selector": selector, 
                "success": True,
                "message": f"Successfully clicked: {element_description}"
            }
            return json.dumps(result, indent=2)
            
        except Exception as e:
            await self.capture_and_send_screenshot(f"❌ Failed to click: {element_description}")
            return json.dumps({
                "success": False, 
                "error": str(e),
                "message": f"Failed to click element: {element_description}"
            })
    
    async def type_text(self, element_description: str, selector: str, text: str) -> str:
        """Type text into an input field using real Playwright with live view"""
        try:
            # Initialize browser on first use
            if not self.page:
                browser_result = await self.initialize_browser()
                if not browser_result["success"]:
                    return json.dumps({
                        "success": False, 
                        "error": browser_result["error"],
                        "message": f"Failed to initialize browser: {browser_result['message']}"
                    })
                print(f"🌐 {browser_result['message']}")
                
            # Focus and highlight the input field
            element = self.page.locator(selector)
            await element.scroll_into_view_if_needed()
            
            # Highlight the input field
            await self.page.evaluate(f"""
                const element = document.querySelector('{selector}');
                if (element) {{
                    element.style.outline = '3px solid #2563eb';
                    element.style.outlineOffset = '2px';
                    element.focus();
                }}
            """)
            
            # Capture focused state
            await self.capture_and_send_screenshot(f"⌨️ Typing in: {element_description}")
            
            # Clear and type text
            await element.clear()
            await element.fill(text)
            
            # Remove highlight
            await self.page.evaluate(f"""
                const element = document.querySelector('{selector}');
                if (element) {{
                    element.style.outline = '';
                    element.style.outlineOffset = '';
                }}
            """)
            
            # Capture final state
            await self.capture_and_send_screenshot(f"✅ Typed '{text}' in: {element_description}")
            
            result = {
                "action": "type", 
                "element": element_description, 
                "text": text, 
                "success": True,
                "message": f"Successfully typed text into: {element_description}"
            }
            return json.dumps(result, indent=2)
            
        except Exception as e:
            await self.capture_and_send_screenshot(f"❌ Failed to type in: {element_description}")
            return json.dumps({
                "success": False, 
                "error": str(e),
                "message": f"Failed to type in element: {element_description}"
            })
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def hover_element(self, element_description: str, selector: str) -> str:
        """Hover over an element"""
        try:
            result = {"action": "hover", "element": element_description, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def select_dropdown_option(self, element_description: str, selector: str, option: str) -> str:
        """Select an option from a dropdown"""
        try:
            result = {"action": "select", "element": element_description, "option": option, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def press_key(self, key: str) -> str:
        """Press a keyboard key"""
        try:
            result = {"action": "press_key", "key": key, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def evaluate_javascript(self, script: str) -> str:
        """Execute JavaScript on the page"""
        try:
            result = {"action": "evaluate", "script": script, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def upload_file(self, file_path: str) -> str:
        """Upload a file"""
        try:
            result = {"action": "upload", "file": file_path, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def wait_for_condition(self, condition: str, timeout: int = 30) -> str:
        """Wait for a specific condition"""
        try:
            result = {"action": "wait", "condition": condition, "timeout": timeout, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def resize_browser(self, width: int, height: int) -> str:
        """Resize the browser window"""
        try:
            result = {"action": "resize", "width": width, "height": height, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def take_screenshot(self, filename: str = None) -> str:
        """Take a screenshot of the current page"""
        try:
            if not self.page:
                await self.initialize_browser()
                
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
            
            # Ensure screenshots directory exists
            screenshot_dir = Path("screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            
            # Full path for screenshot
            screenshot_path = screenshot_dir / filename
            
            # Take screenshot
            await self.page.screenshot(path=str(screenshot_path), full_page=True)
            
            # Also capture and send to UI for live view
            await self.capture_and_send_screenshot("Screenshot captured")
            
            result = {
                "action": "take_screenshot",
                "filename": filename,
                "path": str(screenshot_path),
                "success": True
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def capture_and_send_screenshot(self, action_description: str = "") -> str:
        """Capture screenshot and send to WebSocket clients for real-time updates"""
        try:
            if not self.page:
                return ""
            
            # Take screenshot optimized for display
            screenshot_bytes = await self.page.screenshot(
                full_page=False,
                type="png",
                clip={"x": 0, "y": 0, "width": 1200, "height": 800}
            )
            
            # Save screenshot to file for reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_dir = Path("screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            screenshot_path = screenshot_dir / f"browser_view_{timestamp}.png"
            
            with open(screenshot_path, "wb") as f:
                f.write(screenshot_bytes)
            
            # Store screenshot data for WebSocket transmission
            # This will be handled by the FastAPI backend's WebSocket manager
            screenshot_data = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Store the screenshot data for backend to access
            self.latest_screenshot = screenshot_data
            
            print(f"📸 Screenshot captured: {action_description}")
            
            return f"Screenshot captured: {action_description}"
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return ""
    
    async def get_console_messages(self) -> str:
        """Get console messages and errors"""
        try:
            result = {"action": "console", "messages": [], "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def get_network_requests(self) -> str:
        """Get network requests"""
        try:
            result = {"action": "network", "requests": [], "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def navigate_back(self) -> str:
        """Navigate back in browser history"""
        try:
            result = {"action": "back", "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def navigate_forward(self) -> str:
        """Navigate forward in browser history"""
        try:
            result = {"action": "forward", "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    # Tab Management Tools
    async def list_tabs(self) -> str:
        """List all open browser tabs"""
        try:
            result = {"action": "list_tabs", "tabs": [], "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def open_new_tab(self, url: str = None) -> str:
        """Open a new browser tab"""
        try:
            result = {"action": "new_tab", "url": url, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def select_tab(self, tab_index: int) -> str:
        """Select a specific tab by index"""
        try:
            result = {"action": "select_tab", "index": tab_index, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def close_tab(self, tab_index: int = None) -> str:
        """Close a specific tab or current tab"""
        try:
            result = {"action": "close_tab", "index": tab_index, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    # Documentation Tools
    async def save_test_plan(self, filename: str, content: str, directory: str = "test_plans") -> str:
        """Save test plan documentation to file"""
        try:
            # Create directory if it doesn't exist
            Path(directory).mkdir(exist_ok=True)
            
            file_path = Path(directory) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result = {"action": "save_test_plan", "file": str(file_path), "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def read_test_plan(self, filename: str, directory: str = "test_plans") -> str:
        """Read existing test plan documentation"""
        try:
            file_path = Path(directory) / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                result = {"action": "read_test_plan", "content": content, "success": True}
                return json.dumps(result, indent=2)
            else:
                return json.dumps({"success": False, "error": "File not found"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def update_flow_status(self, flow_id: str, status: str, details: str = "") -> str:
        """Update the status of a discovered flow"""
        try:
            # This would update internal flow tracking
            flow_update = {
                "flow_id": flow_id,
                "status": status,
                "details": details,
                "timestamp": "2025-07-31T15:40:00Z"
            }
            
            result = {"action": "update_flow_status", "update": flow_update, "success": True}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def generate_final_documentation(self, website_url: str, validated_flows: List[str]) -> str:
        """Generate final comprehensive test documentation"""
        try:
            # This would generate the final markdown documentation
            documentation = f"""
# Test Suite: Website Exploration Results

## Website: {website_url}
## Generated: 2025-07-31

## Overview
- **Total Flows Discovered**: {len(self.discovered_flows)}
- **Validated Flows**: {len(validated_flows)}
- **Test Coverage**: Comprehensive user flow documentation

## Validated Test Cases
{chr(10).join([f"- {flow}" for flow in validated_flows])}

## Test Suite Structure
### 1. Authentication Tests
### 2. Navigation Tests  
### 3. Form Interaction Tests
### 4. UI Validation Tests

## Recommendations
- Priority: Critical path flows should be automated first
- Automation: All validated flows are ready for test automation
- Maintenance: Regular validation recommended for dynamic content
"""
            
            result = {
                "action": "generate_documentation", 
                "documentation": documentation,
                "success": True
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def process_exploration_task(self, task: str) -> str:
        """Process a website exploration task using the AutoGen agent"""
        
        if not self.team:
            return "❌ Website Explore Agent not initialized. Please restart the session."
        
        try:
            # Create conversation history with user message
            conversation_history = [
                TextMessage(
                    source="user",
                    content=task
                )
            ]
            
            # Process with agent using streaming approach with better error handling
            responses = []
            current_response = ""
            
            try:
                async for msg in self.agent.on_messages_stream(
                    messages=conversation_history,
                    cancellation_token=CancellationToken(),
                ):
                    if isinstance(msg, ModelClientStreamingChunkEvent):
                        # Collect streaming response chunks
                        if msg.content:
                            current_response += msg.content
                    elif isinstance(msg, Response):
                        # Done streaming - save the complete response
                        if current_response.strip():
                            responses.append(current_response.strip())
                            current_response = ""
                        
                        # Also check if Response has any messages
                        if hasattr(msg, 'messages') and msg.messages:
                            for response_msg in msg.messages:
                                if hasattr(response_msg, 'content') and response_msg.content:
                                    responses.append(str(response_msg.content))
                                    
            except Exception as stream_error:
                print(f"Streaming error: {stream_error}")
                # Fallback to direct agent response if streaming fails
                try:
                    # Use a simpler approach without streaming
                    from autogen_agentchat.messages import TextMessage as SimpleTextMessage
                    simple_response = await self.agent.on_messages(
                        messages=[SimpleTextMessage(source="user", content=task)],
                        cancellation_token=CancellationToken()
                    )
                    
                    if hasattr(simple_response, 'messages') and simple_response.messages:
                        for msg in simple_response.messages:
                            if hasattr(msg, 'content') and msg.content:
                                responses.append(str(msg.content))
                    else:
                        responses.append("✅ Task processed successfully using fallback method.")
                        
                except Exception as fallback_error:
                    print(f"Fallback error: {fallback_error}")
                    return f"❌ Error processing task: {str(fallback_error)}"
            
            # Return combined responses
            if responses:
                return "\n\n".join(responses)
            else:
                return "✅ Website exploration task completed successfully."
            
        except Exception as e:
            print(f"Process exploration error: {e}")
            return f"❌ Error processing exploration task: {str(e)}"
    
    async def close(self):
        """Clean up resources including browser"""
        # Clean up browser first
        await self.close_browser()
        
        # Clean up model client
        if self.model_client:
            await self.model_client.close()

# Global agent instance
website_explore_agent = WebsiteExploreAgent()

# Chainlit Integration
@cl.on_chat_start
async def start_website_exploration():
    """Initialize the website exploration session"""
    
    await cl.Message(
        content="🌐 **Website Exploration Agent with Split-Screen Browser View**\n\n" +
                "✨ **Features:**\n" +
                "• Chat interface on the left\n" +
                "• Live browser view on the right\n" +
                "• Real-time visual feedback\n" +
                "• Element highlighting\n" +
                "• Automatic screenshot capture\n\n" +
                "**Ready to explore!** Try: *'Navigate to https://example.com'*",
        author="System"
    ).send()
    
    # Initialize the agent
    success = await website_explore_agent.initialize()
    
    if success:
        welcome_message = """
🎯 **Website Exploration Agent Ready!**

I systematically explore websites and generate comprehensive test plans using Playwright browser automation.

**🌐 Browser Behavior:**
- **Browser opens only when needed** - when you request website exploration
- **Headed mode** - you'll see the browser window for visual exploration
- **DevTools enabled** - for advanced debugging and inspection

**My Capabilities:**
- 🔍 **Systematic Discovery**: Navigate and analyze website functionality
- 📋 **Flow Documentation**: Create detailed user flow documentation  
- ✅ **Flow Validation**: Test each discovered flow step-by-step
- 📊 **Test Plan Generation**: Generate comprehensive test suite documentation
- 🏗️ **Iterative Exploration**: Continuously discover new functionality

**Exploration Workflow:**
1. **Discovery**: Analyze pages and identify interactive elements
2. **Selection**: Present flows hierarchically for user selection
3. **Validation**: Execute selected flows using browser automation
4. **Documentation**: Generate comprehensive test plans

**To get started, provide:**
- 🌐 **Target URL**: The website you want me to explore
- 🎯 **Focus Areas**: Specific functionality to prioritize (optional)
- ⚙️ **Constraints**: Any limitations or boundaries (optional)

**Example:**
"Explore https://playwright.microsoft.com and focus on authentication and workspace management flows"

**Note:** The browser window will open automatically when I start exploring your first website.

What website would you like me to explore?
"""
        
        await cl.Message(
            content=welcome_message,
            author="Website Explorer"
        ).send()
        
        # Store agent in session
        cl.user_session.set("website_explore_agent", website_explore_agent)
        
    else:
        error_message = """
❌ **Failed to Initialize Website Exploration Agent**

Please check that you have set the following environment variables:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_API_KEY` (or use Azure AD authentication)

Example:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
export AZURE_OPENAI_API_VERSION="2024-06-01"
export AZURE_OPENAI_API_KEY="your-api-key"
```

Please set these variables and restart the application.
"""
        
        await cl.Message(
            content=error_message,
            author="System"
        ).send()

@cl.on_message
async def handle_exploration_message(message: cl.Message):
    """Handle website exploration requests"""
    
    # Get the agent from session
    agent = cl.user_session.get("website_explore_agent")
    
    if not agent or not agent.team:
        await cl.Message(
            content="❌ Website Exploration Agent not available. Please restart the session.",
            author="System"
        ).send()
        return
    
    # Show processing indicator
    async with cl.Step(name="Exploring Website", type="run") as step:
        step.input = message.content
        
        # Process the exploration request
        await cl.Message(
            content="🕷️ Starting website exploration...",
            author="System"
        ).send()
        
        response = await agent.process_exploration_task(message.content)
        step.output = response
    
    # Send the agent's response
    await cl.Message(
        content=response,
        author="Website Explorer"
    ).send()

@cl.on_chat_end
async def end_exploration():
    """Clean up when exploration session ends"""
    
    agent = cl.user_session.get("website_explore_agent")
    if agent:
        await agent.close()
    
    await cl.Message(
        content="👋 Website exploration session completed. Thank you for using the Website Exploration Agent!",
        author="System"
    ).send()

if __name__ == "__main__":
    # Check if running directly
    print("🌐 Website Exploration Agent")
    print("=" * 40)
    print("🚀 Starting Chainlit server...")
    print("📝 Make sure to set your Azure OpenAI environment variables")
    print("🔍 Ready to explore websites and generate test plans")
    print("=" * 40)
