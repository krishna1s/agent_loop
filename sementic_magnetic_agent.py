#!/usr/bin/env python3
"""
Magnetic One Test Orchestrator with Chainlit Integration
Uses Microsoft's Magnetic One orchestration pattern with specialized agents working together
to create comprehensive test plans through dynamic collaboration and coordination.
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
import uuid
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Chainlit imports
import chainlit as cl

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, MagenticOrchestration, StandardMagenticManager
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.connectors.mcp import MCPSsePlugin

# MCP imports
import httpx
import asyncio
import json
from typing import Any, Dict, List, Optional

class MagneticTestOrchestrator:
    """Magnetic One Test Orchestrator for enhanced test plan generation using specialized agents"""
    
    def __init__(self):
        self.kernel = None
        self.test_agent = None
        self.orchestration = None
        self.manager = None
        self.runtime = None
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.current_message = None  # Store current Chainlit message for updates
        self.response_parts = []  # Store streaming response parts
        
    async def initialize(self) -> bool:
        """Initialize Semantic Kernel with Magnetic One orchestration and specialized test planning agent"""
        try:
            # Validate required environment variables
            required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing environment variables: {missing_vars}")
            
            # Initialize Semantic Kernel
            self.kernel = Kernel()
            
            # Create Azure OpenAI settings
            azure_settings = AzureOpenAISettings()
            
            # Initialize Azure Chat Completion service
            azure_chat_completion = AzureChatCompletion(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            )
            
            # Add the service to kernel
            self.kernel.add_service(azure_chat_completion)
            
            # Register Playwright MCP plugins
            await self.register_playwright_mcp_plugins()
            
            # Create test planning agent
            await self.create_test_agent()
            
            # Initialize Magnetic Manager for orchestration
            self.manager = StandardMagenticManager(chat_completion_service=azure_chat_completion)
            
            # Create runtime for agent execution
            self.runtime = InProcessRuntime()
            self.runtime.start()
            
            # Create Magnetic Orchestration with the test planning agent and callback
            self.orchestration = MagenticOrchestration(
                members=[self.test_agent],
                manager=self.manager,
                agent_response_callback=self.agent_response_callback
            )
            
            print("✅ Magnetic One Test Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Magnetic One Test Orchestrator: {e}")
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return False
    
    def agent_response_callback(self, message: ChatMessageContent) -> None:
        """Callback to handle agent responses and stream to UI"""
        try:
            if hasattr(message, 'content') and message.content:
                # Store the response part
                agent_name = getattr(message, 'name', 'Agent')
                response_part = f"**{agent_name}**: {message.content}"
                self.response_parts.append(response_part)
                
                # Print to console for debugging
                print(f"🤖 {response_part}")
                
                # Stream the response to UI if we have a current message
                if self.current_message:
                    # Build the complete streaming response
                    complete_response = "\n\n".join(self.response_parts)
                    
                    # Update UI asynchronously
                    asyncio.create_task(self._update_ui_message(complete_response))
                
        except Exception as e:
            print(f"❌ Error in agent callback: {e}")
    
    async def _update_ui_message(self, content: str):
        """Update the UI message with agent progress"""
        try:
            if self.current_message:
                # Replace the content with streaming response
                self.current_message.content = f"🔄 **Live Agent Response:**\n\n{content}"
                await self.current_message.update()
        except Exception as e:
            print(f"❌ Error updating UI: {e}")
    
    async def register_playwright_mcp_plugins(self):
        """Register Playwright MCP SSE plugin for browser automation and analysis"""
        try:
            # Create MCP SSE plugin for Playwright server
            playwright_plugin = MCPSsePlugin(
                name="PlaywrightMCP",
                description="Playwright browser automation and testing tools",
                url=f"http://localhost:8931/sse",
            )
            
            # Connect to the MCP server
            await playwright_plugin.connect()
            
            # Add the plugin to the kernel
            self.kernel.add_plugin(playwright_plugin)
            
            print(f"✅ Playwright MCP SSE plugin connected successfully to port {os.getenv('PLAYWRIGHT_MCP_PORT', '8931')}")
            
        except Exception as e:
            print(f"❌ Failed to connect to Playwright MCP server: {e}")
            print("⚠️  Make sure the Playwright MCP server is running on the specified port")
            # Don't fail initialization if MCP server is not available
            pass
    
    async def create_test_agent(self):
        """Create test planning agent with comprehensive system prompt using the kernel"""
        
        # Test Planning Agent with comprehensive system prompt
        # Use the kernel that has MCP plugins registered
        self.test_agent = ChatCompletionAgent(
            name="TestPlannerAgent",
            description="Comprehensive test planning agent for website testing",
            kernel=self.kernel,  # Use the kernel with MCP plugins
            instructions="""
You are an autonomous Test Exploration Agent that generates detailed test cases from high-level user instructions using MCP Playwright tools.

Your thinking should be thorough and so it's fine if it's very long. You can think step by step before and after each action you decide to take.

You MUST iterate and keep going until the problem is solved.

Your goals:
1. Take simple user test steps and autonomously explore the target website using available MCP tools.
2. Keep working in a loop until you can produce a fully enhanced, detailed test plan with preconditions, step-by-step actions, and expected results.
3. After every meaningful action or decision (navigation, click, type, wait, error, or discovery), inform user of your progress:
   - Example: "Navigated to login page.", "Clicked 'Sign In' button.", "Waiting for dashboard to load.", "Could not find element, taking snapshot and retrying.", etc.
4. Always take screenshots after every action to capture UI evidence.
5. Use snapshots to dynamically discover new elements if a selector or action fails.
6. Never stop prematurely: if a step is unclear, attempt different flows (navigation, click, type, select, evaluate).
7. If clarification or missing test data is required, then ask user for input - once you receive input, continue with the iteration.
8. Return the final documentation in Markdown format.
9. Always include Preconditions, Test Steps, and Expected Results (like an enhanced test case).

### Available MCP Tools

Use the following MCP tools for browser automation:
- mcp_playwright_browser_navigate: Navigate to URLs
- mcp_playwright_browser_snapshot: Capture page snapshots for element discovery
- mcp_playwright_browser_click: Click on elements
- mcp_playwright_browser_type: Type text into input fields
- mcp_playwright_browser_take_screenshot: Take screenshots
- mcp_playwright_browser_hover: Hover over elements
- mcp_playwright_browser_select_option: Select dropdown options
- mcp_playwright_browser_press_key: Press keyboard keys
- mcp_playwright_browser_evaluate: Execute JavaScript
- mcp_playwright_browser_wait_for: Wait for conditions
- mcp_playwright_browser_resize: Resize browser window
- mcp_playwright_browser_navigate_back/forward: Navigate browser history
- mcp_playwright_browser_tab_*: Tab management tools

### Execution Strategy

- Start with mcp_playwright_browser_navigate to open the portal.
- At each step:
  - Perform the action using appropriate MCP tools.
  - If element not found → use mcp_playwright_browser_snapshot, retry with updated selector, and report progress.
  - If blocked → ask user for further input

You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.

- Continue until the test flow is completed end-to-end.
- Save a structured enhanced test case (Markdown).

### Important: Use MCP Tools Actively

You have access to Playwright MCP tools through the kernel. Always use these tools to:
1. Navigate to websites using mcp_playwright_browser_navigate
2. Take snapshots using mcp_playwright_browser_snapshot to understand page structure
3. Interact with elements using mcp_playwright_browser_click, mcp_playwright_browser_type
4. Capture evidence using mcp_playwright_browser_take_screenshot
5. Handle dynamic content and complex scenarios

Always perform actual browser automation - don't just describe what should be done!
            """
        )
        
        print("✅ Test planning agent created successfully")
    
    async def start_runtime(self):
        """Start the InProcess runtime"""
        try:
            if not self.runtime:
                self.runtime = InProcessRuntime()
                self.runtime.start()
                print("✅ Runtime started successfully")
        except Exception as e:
            print(f"❌ Failed to start runtime: {e}")
    
    async def chat_with_agent(self, user_message: str, ui_message=None) -> str:
        """Process user message directly using Magnetic One orchestration with real-time streaming"""
        try:
            # Store the UI message for updates
            self.current_message = ui_message
            self.response_parts = []
            
            if ui_message:
                ui_message.content = "🤖 Starting Magnetic One orchestration..."
                await ui_message.update()
            
            # Start the runtime
            if not self.runtime:
                await self.start_runtime()
            
            if ui_message:
                ui_message.content = "🎯 Agent is analyzing your request..."
                await ui_message.update()
            
            # Execute Magnetic One orchestration with streaming via callback
            # The agent_response_callback will handle real-time updates
            orchestration_result = await self.orchestration.invoke(
                task=user_message,
                runtime=self.runtime
            )
            
            # Get the final result
            final_result = await orchestration_result.get()
            
            # Return the accumulated response or final result
            if self.response_parts:
                complete_response = "\n\n".join(self.response_parts)
                return complete_response
            
            return final_result if final_result else "Task completed successfully."
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            print(f"❌ Chat error: {e}")
            print(f"❌ Full traceback: {traceback.format_exc()}")
            
            if ui_message:
                ui_message.content = f"❌ Error: {str(e)}"
                await ui_message.update()
            
            return error_response
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.runtime:
                await self.runtime.stop_when_idle()
                print("✅ Runtime stopped successfully")
        except Exception as e:
            print(f"⚠️ Error stopping runtime: {e}")

# Global orchestrator instance
magnetic_orchestrator = MagneticTestOrchestrator()

# Chainlit event handlers
@cl.on_chat_start
async def start():
    """Initialize the Magnetic One orchestrator when chat starts"""
    await cl.Message(content="🚀 Initializing Magnetic One Test Orchestrator...").send()
    
    success = await magnetic_orchestrator.initialize()
    if not success:
        await cl.Message(content="❌ Failed to initialize Magnetic One orchestrator").send()
        return
    
    await cl.Message(
        content="✅ **Magnetic One Test Orchestrator** is ready!\n\n"
        "I'm your expert test planning agent powered by Microsoft's Magnetic One orchestration.\n\n"
        "**What I can help you with:**\n"
        "- 🎯 Execute live website testing scenarios using Playwright MCP tools\n"
        "- 📋 Create detailed test cases with real browser automation\n"
        "- 🔧 Provide real-time progress updates as I work\n"
        "- 🚀 Analyze websites and generate comprehensive test plans\n"
        "- 💡 Handle complex multi-step testing workflows\n\n"
        "**Key Features:**\n"
        "- **Live Browser Automation**: Real Playwright interactions\n"
        "- **Real-time Progress**: See exactly what I'm doing\n"
        "- **Kernel Integration**: MCP tools available through Semantic Kernel\n"
        "- **Evidence Capture**: Screenshots and snapshots\n\n"
        "**Example test scenario:**\n"
        "```\n"
        "I want to test my website login functionality:\n"
        "1. go to playwright.microsoft.com\n"
        "2. Click on sign in button\n"
        "3. Enter email and password\n"
        "4. if login successful report success\n"
        "```\n\n"
        "Just describe your testing scenario and I'll execute it step by step!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and process them through Magnetic One orchestration"""
    user_message = message.content.strip()
    
    try:
        # Create a progress message that will be updated in real-time
        progress_msg = await cl.Message(content="🤖 Processing your request with Magnetic One orchestration...").send()
        
        # Pass message directly to agent for intelligent processing with UI updates
        response = await magnetic_orchestrator.chat_with_agent(user_message, progress_msg)
        
        # Send the final response as a new message
        await cl.Message(content=f"**🎯 Final Result:**\n\n{response}").send()
        
    except Exception as e:
        await cl.Message(content=f"❌ **Error processing request:**\n\n{str(e)}").send()

@cl.on_stop
async def on_stop():
    """Clean up resources when chat stops"""
    await magnetic_orchestrator.cleanup()

if __name__ == "__main__":
    print("🚀 Starting Magnetic One Test Orchestrator with Chainlit...")
    print("🌐 Chainlit app will be available at: http://localhost:8000")
    print("📋 Features:")
    print("  - Magnetic One orchestration for expert test planning")
    print("  - Playwright MCP integration through Semantic Kernel")
    print("  - Interactive chat interface with real-time progress")
    print("  - Live browser automation with evidence capture")
    print("  - Kernel-based plugin system")
    print()
    
    # Run chainlit
    import subprocess
    subprocess.run(["chainlit", "run", __file__, "--port", "8080"])
