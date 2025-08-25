#!/usr/bin/env python3
"""
AutoGen Team with Chainlit Integration
Converts the FastAPI/WebSocket app_team.py to use Chainlit for a cleaner chat interface
while maintaining all the existing AutoGen MagenticOneGroupChat functionality.
"""

import json
import logging
import os
import asyncio
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from pathlib import Path

# Chainlit imports
import chainlit as cl

# AutoGen imports
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent, ModelClientStreamingChunkEvent
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams
from azure.identity import DefaultAzureCredential

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import aiofiles
import yaml
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoGenChainlit")

class DateTimeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def safe_json_dumps(obj):
    """Safely serialize objects to JSON, handling datetime objects."""
    return json.dumps(obj, cls=DateTimeJSONEncoder)

def safe_model_dump(message):
    """Safely dump a Pydantic model to dict, handling datetime serialization."""
    try:
        logger.debug(f"safe_model_dump called with: type={type(message)}, message={message}")
        
        # Check if message is None
        if message is None:
            logger.error("safe_model_dump received None message")
            return {
                "content": "None message received",
                "source": "system_error",
                "type": "error"
            }
        
        # Check if message has model_dump method
        if not hasattr(message, 'model_dump'):
            logger.error(f"Message {type(message)} does not have model_dump method")
            return {
                "content": str(message),
                "source": "system_error", 
                "type": type(message).__name__
            }
        
        # Get the model dump
        data = message.model_dump()
        
        # Convert any datetime objects to ISO format strings
        def convert_datetime(obj):
            if isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        return convert_datetime(data)
    except Exception as e:
        logger.error(f"Error in safe_model_dump: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to basic dict representation
        return {
            "content": str(getattr(message, 'content', '')),
            "source": str(getattr(message, 'source', 'unknown')),
            "type": type(message).__name__
        }

class WebsiteExploreAgent:
    """Website Exploration Agent using AutoGen and MCP Playwright tools"""
    
    def __init__(self):
        self.model_client = None
        self.agent = None
        self.team = None
        self.workbench = None
        self.runtime = None
        
        # MCP server configuration
        self.playwright_server_params = SseServerParams(
            url="http://localhost:8931/sse",
        )
    
    async def initialize(self):
        """Initialize the AutoGen agent with MCP Playwright capabilities"""
        
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
                    model='gpt-4o',
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
                    model='gpt-4o',
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider
                )
            
            # Initialize MCP workbench
            self.workbench = McpWorkbench(self.playwright_server_params)
            await self.workbench.start()
            
            print("✅ MCP Playwright server connected")
            
            # Get tools from MCP workbench (await the coroutine)
            mcp_tools = await self.workbench.list_tools()
            print(f"🔧 Retrieved {len(mcp_tools)} MCP tools")
            
            # Create the AssistantAgent with MCP workbench
            self.agent = AssistantAgent(
                name="TestPlanGenerator",
                model_client=self.model_client,
                system_message=self.get_system_message(),
                workbench=self.workbench,
                description="Specialized agent for systematic website exploration and test plan generation using MCP Playwright tools",
                reflect_on_tool_use=True,
                model_client_stream=True
            )
            
            print("✅ AutoGen agent initialized with MCP Playwright tools")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Website Explore Agent: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"❌ Failed to initialize Website Explore Agent: {e}")
            return False

    async def cleanup(self):
        """Clean up MCP workbench resources"""
        try:
            if self.workbench:
                await self.workbench.stop()
                self.workbench = None
                print("✅ MCP workbench stopped")
        except Exception as e:
            logger.error(f"Error cleaning up MCP workbench: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"Error cleaning up MCP workbench: {e}")

    def get_system_message(self) -> str:
        """Create the comprehensive system message for the agent"""
        return '''You are an autonomous Test Exploration Agent that generates detailed test cases from high-level user instructions using MCP Playwright tools.

Your thinking should be thorough and so it's fine if it's very long. You can think step by step before and after each action you decide to take.

You MUST iterate and keep going until the problem is solved.

Your goals:
1. Take simple user test steps and autonomously explore the target website using available MCP tools.
2. Keep working in a loop until you can produce a fully enhanced, detailed test plan with preconditions, step-by-step actions, and expected results.
3. After every meaningful action or decision (navigation, click, type, wait, error, or discovery), inform user of your progress :
   - Example: "Navigated to login page.", "Clicked 'Sign In' button.", "Waiting for dashboard to load.", "Could not find element, taking snapshot and retrying.", etc.
4. Always take screenshots after every action to capture UI evidence.
5. Use snapshots to dynamically discover new elements if a selector or action fails.
6. Never stop prematurely: if a step is unclear, attempt different flows (navigation, click, type, select, evaluate).
7. If clarification or missing test data is required, then ask user for input
    once you receive input, continue with the iteration.
8. Return the final documentation in Markdown format.
9. Always include Preconditions, Test Steps, and Expected Results (like an enhanced test case).

### 🔹 Available MCP Tools

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

### 🔹 Execution Strategy

- Start with mcp_playwright_browser_navigate to open the portal.
- At each step:
  - Perform the action using appropriate MCP tools.
  - If element not found → use mcp_playwright_browser_snapshot, retry with updated selector, and report progress.
  - If blocked → ask user for futher input

You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.

- Continue until the test flow is completed end-to-end.
- Save a structured enhanced test case (Markdown).'''

# Global variables for team management
website_agent = None
team = None
state_path = "team_state.json"
history_path = "team_history.json"

async def get_team(user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]]) -> MagenticOneGroupChat:
    """Create and return the AutoGen team with user input function"""
    global website_agent
    
    try:
        # Initialize the WebsiteExploreAgent if not already done
        if website_agent is None:
            website_agent = WebsiteExploreAgent()
            await website_agent.initialize()
        
        # Check if initialization was successful
        if website_agent.agent is None:
            raise RuntimeError("Failed to initialize WebsiteExploreAgent - agent is None")
        
        agent = website_agent.agent

        user_proxy = UserProxyAgent(
            name="user",
            input_func=user_input_func,  # Use the user input function.
        )
        
        # Log the participants before creating the team
        logger.debug(f"Creating team with participants: agent={agent}, user_proxy={user_proxy}")
        logger.debug(f"Agent name: {getattr(agent, 'name', 'NO_NAME')}")
        logger.debug(f"User proxy name: {getattr(user_proxy, 'name', 'NO_NAME')}")
        
        team = MagenticOneGroupChat(
            [agent, user_proxy],
            model_client=website_agent.model_client
        )
        
        return team
    except Exception as e:
        logger.error(f"Error creating team: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

async def get_history() -> list[dict[str, Any]]:
    """Get chat history from file."""
    if not os.path.exists(history_path):
        return []
    try:
        async with aiofiles.open(history_path, "r") as file:
            return json.loads(await file.read())
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return []

async def save_history(history: list[dict[str, Any]]):
    """Save chat history to file."""
    try:
        async with aiofiles.open(history_path, "w") as file:
            await file.write(safe_json_dumps(history))
    except Exception as e:
        logger.error(f"Error saving history: {e}")

async def save_team_state(team: MagenticOneGroupChat):
    """Save team state to file."""
    try:
        async with aiofiles.open(state_path, "w") as file:
            state = await team.save_state()
            await file.write(safe_json_dumps(state))
    except Exception as e:
        logger.error(f"Error saving team state: {e}")

# Chainlit event handlers
@cl.on_chat_start
async def start():
    """Initialize the AutoGen team when chat starts"""
    await cl.Message(content="🚀 Initializing AutoGen MagenticOne Team with Playwright MCP...").send()
    
    try:
        # Initialize the website agent
        global website_agent
        website_agent = WebsiteExploreAgent()
        success = await website_agent.initialize()
        
        if not success:
            await cl.Message(content="❌ Failed to initialize AutoGen team").send()
            return
        
        # Load existing history
        history = await get_history()
        if history:
            await cl.Message(content=f"📜 Loaded {len(history)} previous messages").send()
        
        await cl.Message(
            content="✅ **AutoGen MagenticOne Team** is ready!\n\n"
            "I'm your autonomous test exploration agent powered by AutoGen's MagenticOne orchestration.\n\n"
            "**What I can help you with:**\n"
            "- 🎯 Execute comprehensive website testing scenarios\n"
            "- 🔧 Use live browser automation through Playwright MCP\n"
            "- 📋 Generate detailed test plans with evidence\n"
            "- 🚀 Explore websites autonomously and report progress\n"
            "- 💡 Handle complex multi-step testing workflows\n\n"
            "**Example test scenarios:**\n"
            "```\n"
            "I want to test my website login functionality:\n"
            "1. go to playwright.microsoft.com\n"
            "2. Click on sign in button\n"
            "3. Enter email and password\n"
            "4. if login successful report success\n"
            "```\n\n"
            "**Available MCP Tools:**\n"
            "- Browser navigation and automation\n"
            "- Element interaction (click, type, select)\n"
            "- Screenshot and snapshot capture\n"
            "- JavaScript execution\n"
            "- Network monitoring\n\n"
            "Just describe your testing scenario and I'll execute it step by step!"
        ).send()
        
    except Exception as e:
        await cl.Message(content=f"❌ Error during initialization: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and process them through AutoGen team"""
    user_message = message.content.strip()
    
    # Queue for user input requests from the agent
    user_input_queue = asyncio.Queue()
    current_user_input_request = None
    
    # User input function for the team
    async def user_input_func(prompt: str, cancellation_token: CancellationToken | None) -> str:
        nonlocal current_user_input_request
        
        try:
            logger.info(f"🤔 Agent requesting user input: {prompt}")
            
            # Send user input request to Chainlit
            current_user_input_request = await cl.AskUserMessage(
                content=f"🤔 **Agent needs input:**\n\n{prompt}",
                timeout=300  # 5 minutes timeout
            ).send()
            
            logger.info(f"📝 User input response type: {type(current_user_input_request)}")
            logger.info(f"📝 User input response: {current_user_input_request}")
            
            # Handle different response types
            if current_user_input_request is None:
                return "No response received (None)"
            
            # If it's a dictionary, get content from 'output' field (Chainlit's response format)
            if isinstance(current_user_input_request, dict):
                # Try 'output' field first (this is where Chainlit puts the actual user input)
                content = current_user_input_request.get('output', '')
                if content:
                    return content
                
                # Fallback to 'content' field
                content = current_user_input_request.get('content', '')
                if content:
                    return content
                
                # If neither field has content, log the full structure for debugging
                logger.warning(f"No content found in response fields. Full response: {current_user_input_request}")
                return "No content in response"
            
            # If it has a content attribute
            if hasattr(current_user_input_request, 'content'):
                return current_user_input_request.content
            
            # Convert to string as fallback
            return str(current_user_input_request)
                
        except Exception as e:
            error_msg = f"Error getting user input: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return error_msg
    
    try:
        # Show processing message
        progress_msg = await cl.Message(content="🤖 Processing your request with AutoGen MagenticOne...").send()
        
        # Get the team
        team = await get_team(user_input_func)
        
        # Create TextMessage for team processing
        text_message = TextMessage(content=user_message, source="user")
        
        # Get chat history
        history = await get_history()
        
        # Buffer for streaming content
        message_buffer = {}
        response_messages = []
        
        # Process message through AutoGen team
        stream = team.run_stream(task=text_message)
        
        # Update progress message
        progress_msg.content = "🔄 AutoGen team is working..."
        await progress_msg.update()
        
        async for msg in stream:
            try:
                if isinstance(msg, TaskResult):
                    continue
                
                # Safely serialize message
                message_data = safe_model_dump(msg)
                message_type = message_data.get('type', '')
                source = message_data.get('source', '')
                content = message_data.get('content', '')
                
                # Calculate content length safely
                content_length = 0
                if isinstance(content, str):
                    content_length = len(content)
                elif isinstance(content, list):
                    content_length = len(str(content))
                elif content:
                    content_length = len(str(content))
                
                logger.info(f"📨 Processing: type='{message_type}', source='{source}', content_length={content_length}, content_type={type(content).__name__}")
                
                # Handle different message types
                if isinstance(msg, UserInputRequestedEvent):
                    # This should be handled by the user_input_func, but we can log it
                    logger.info("📤 UserInputRequestedEvent detected")
                    continue
                    
                elif message_type == 'ModelClientStreamingChunkEvent' and source and content:
                    # Buffer streaming chunks (only if content is a string)
                    if isinstance(content, str):
                        if source not in message_buffer:
                            message_buffer[source] = {
                                'type': 'TextMessage',
                                'source': source,
                                'content': '',
                                'models_usage': message_data.get('models_usage'),
                                'metadata': message_data.get('metadata', {}),
                                'created_at': message_data.get('created_at'),
                                'request_id': message_data.get('request_id')
                            }
                        # Append content to buffer
                        message_buffer[source]['content'] += content
                        
                        # Update progress with current agent activity
                        agent_name = source.replace('_', ' ').title()
                        progress_msg.content = f"🤖 {agent_name} is working..."
                        await progress_msg.update()
                    
                elif message_type == 'TextMessage' and source and content:
                    # Complete message from agent
                    agent_name = source.replace('_', ' ').title()
                    
                    # Convert content to string if it's not already
                    content_str = content if isinstance(content, str) else str(content)
                    
                    # Send the complete message
                    await cl.Message(
                        content=f"**{agent_name}:**\n\n{content_str}",
                        author=agent_name
                    ).send()
                    
                    response_messages.append(message_data)
                    history.append(message_data)
                    
                else:
                    # Other message types
                    # Handle content that might be a list or string
                    content_str = ""
                    if isinstance(content, str):
                        content_str = content.strip()
                    elif isinstance(content, list):
                        # Convert list content to string
                        content_str = str(content).strip()
                    elif content:
                        content_str = str(content).strip()
                    
                    if content_str:
                        response_messages.append(message_data)
                        history.append(message_data)
                        
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Send any remaining buffered messages
        for source, buffered_message in message_buffer.items():
            buffered_content = buffered_message.get('content', '')
            # Handle content that might be a list or string
            if isinstance(buffered_content, str):
                content_to_check = buffered_content.strip()
            elif isinstance(buffered_content, list):
                content_to_check = str(buffered_content).strip()
            else:
                content_to_check = str(buffered_content).strip() if buffered_content else ""
            
            if content_to_check:
                agent_name = source.replace('_', ' ').title()
                
                await cl.Message(
                    content=f"**{agent_name}:**\n\n{content_to_check}",
                    author=agent_name
                ).send()
                
                response_messages.append(buffered_message)
                history.append(buffered_message)
        
        # Update progress to completion
        progress_msg.content = "✅ AutoGen team completed the task!"
        await progress_msg.update()
        
        # Save team state and history
        await save_team_state(team)
        
        # Add user message to history
        user_msg_data = {
            "type": "TextMessage",
            "source": "user", 
            "content": user_message,
            "created_at": datetime.now().isoformat()
        }
        history.append(user_msg_data)
        
        await save_history(history)
        
    except Exception as e:
        logger.error(f"Error in main message handler: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        await cl.Message(
            content=f"❌ **Error processing request:**\n\n{str(e)}\n\n"
            "Please try again or check the logs for more details."
        ).send()

@cl.on_stop
async def on_stop():
    """Clean up resources when chat stops"""
    global website_agent
    if website_agent:
        await website_agent.cleanup()
        website_agent = None
    logger.info("🛑 AutoGen Chainlit app stopped and resources cleaned up")

if __name__ == "__main__":
    print("🚀 Starting AutoGen MagenticOne Team with Chainlit...")
    print("🌐 Chainlit app will be available at: http://localhost:8000")
    print("📋 Features:")
    print("  - AutoGen MagenticOne orchestration")
    print("  - Playwright MCP integration for live browser automation")
    print("  - Interactive chat interface with Chainlit")
    print("  - Real-time agent progress updates")
    print("  - Autonomous test execution with evidence capture")
    print()
    
    # Run chainlit
    import subprocess
    subprocess.run(["chainlit", "run", __file__, "--port", "8000"])
