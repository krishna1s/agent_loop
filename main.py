#!/usr/bin/env python3
"""
TestingAgent Chainlit entrypoint using the ProgressOnly Magentic-One orchestrator.

This app initializes an AssistantAgent with MCP Playwright tools and runs a
ProgressOnlyMagenticOneGroupChat so the UI receives concise, structured
"PROGRESS UPDATE" messages emitted by the orchestrator only.
"""

import json
import io
import contextlib
import tempfile
import shutil
import textwrap
import base64
import logging
import os
import asyncio
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from pathlib import Path
from autogen_ext.agents.azure._azure_ai_agent import AzureAIAgent
from azure.identity.aio import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.ai.agents.models import (
    McpTool,
    FunctionTool,
    ToolSet,
    MCPToolResource
)

import time  # noqa: F401

# Chainlit imports
import chainlit as cl

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams
from playwright.async_api import async_playwright  # for CDP execution API
from azure.identity import DefaultAzureCredential  # noqa: F401

# Import our custom orchestrator
from TestingAgent import CustomMagneticOneGroupChat

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import aiofiles  # noqa: F401
import yaml  # noqa: F401
from pydantic import BaseModel  # noqa: F401

# Setup logging
logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.INFO)

logger = logging.getLogger("AutoGenChainlitFiltered")

# Get MCP server configuration from environment variables
mcp_server_url = "https://playwrightmcp.eastus.azurecontainer.io"
mcp_server_label = os.environ.get("MCP_SERVER_LABEL", "mcp_playwright")

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
        self.agent = None            # Active agent (AzureAIAgent if available, else AssistantAgent)
        self.azure_agent = None      # Specific handle if AzureAIAgent initialized
        self.team = None
        self.workbench = None
        self.runtime = None
        
        # MCP server configuration (runs inside the same container by default)
        # Override with PLAYWRIGHT_MCP_SSE_URL if you host it elsewhere
        mcp_sse_url = os.getenv(
            "PLAYWRIGHT_MCP_SSE_URL",
            "http://127.0.0.1:8931/sse",
        )
        self.playwright_server_params = SseServerParams(
            url=mcp_sse_url,
        )
    
    async def initialize(self):
        """Initialize the AutoGen agent with MCP Playwright capabilities"""
        
        try:
            # Check environment variables
            required_vars = [
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_DEPLOYMENT_NAME",
                "AZURE_OPENAI_API_VERSION",
            ]
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(
                    f"Missing environment variables: {missing_vars}"
                )
            
            # Create Azure OpenAI client (for legacy AssistantAgent path)
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            # Important: model must be a valid OpenAI model name, not the Azure deployment name
            model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
            
            if api_key:
                # API Key authentication
                self.model_client = AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment,
                    model=model_name,
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
                    model=model_name,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider
                )
            
            # Initialize MCP workbench
            self.workbench = McpWorkbench(self.playwright_server_params)
            await self.workbench.start()
            
            # print("✅ MCP Playwright server connected")
            
            # Get tools from MCP workbench (await the coroutine)
            mcp_tools = await self.workbench.list_tools()
            print(f"🔧 Retrieved {len(mcp_tools)} MCP tools")
            
            # Create the AssistantAgent with MCP workbench
            self.agent = AssistantAgent(
                name="TestPlanGenerator",
                model_client=self.model_client,
                system_message=self.get_system_message(),
                workbench=self.workbench,
                description=(
                    "Specialized agent for systematic website exploration and "
                    "test plan generation using MCP Playwright tools"
                ),
                # NOTE:
                # We disable reflection-on-tool-use and streaming to avoid a
                # known issue where an assistant message containing tool_calls
                # is not immediately followed by the corresponding tool result
                # messages, which triggers a 400 from the API. The workbench
                # will still enable tool usage; this only disables the extra
                # reflection pass and streaming mode.
                reflect_on_tool_use=False,
                model_client_stream=False
            )
            
            # Example: project client wiring if needed in the future
            # project_client = AIProjectClient(  # type: ignore
            #     credential=DefaultAzureCredential(),
            #     endpoint=(
            #         "https://foundyuedn.services.ai.azure.com/api/projects/"
            #         "projectuedn"
            #     ),
            # )

            # Attempt AzureAIAgent initialization if Azure AI Project env vars present
            # azure_project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")

            # if azure_project_endpoint:
            #     try:

            #         mcp_server_url = config.mcp_server_url
            #         # agent_name = config.az_assistant_name
            #         # agent_id = config.az_assistant_id
            #         # bing_connection_name = config.bing_connection_name

            #         # Fetch tool schemas
            #         async def fetch_tools():
            #             conn = ServerConnection(mcp_server_url)
            #             await conn.connect()
            #             tools = await conn.list_tools()
            #             await conn.cleanup()
            #             return tools

            #         tools = asyncio.run(fetch_tools())

            #         # Build a function for each tool
            #         def make_tool_func(tool_name):
            #             def tool_func(**kwargs):
            #                 async def call_tool():
            #                     conn = ServerConnection(mcp_server_url)
            #                     await conn.connect()
            #                     result = await conn.execute_tool(tool_name, kwargs)
            #                     await conn.cleanup()
            #                     return result

            #                 return asyncio.run(call_tool())

            #             tool_func.__name__ = tool_name
            #             return tool_func

            #         functions_dict = {tool["name"]: make_tool_func(tool["name"]) for tool in tools}

            #         mcp_function_tool = FunctionTool(functions=list(functions_dict.values()))

            #         toolset = ToolSet()
            #         toolset.add(mcp_function_tool)

            #         # # Initialize agent MCP tool
            #         mcp_tool = McpTool(
            #             server_label=mcp_server_label,
            #             server_url=mcp_server_url,
            #             allowed_tools=["*"],  # Optional: specify allowed tools,
            #         )
            #         mcp_tool.set_approval_mode("never")
            #         print(mcp_tool.definitions)

            #         project_client = AIProjectClient(
            #             credential=DefaultAzureCredential(),
            #             endpoint=azure_project_endpoint,
            #         )

            #         # agents_client = project_client.agents
            #         # agents_client.enable_auto_function_calls(toolset)

            #         try:
            #             # Preferred: pass tools if supported
            #             self.azure_agent = AzureAIAgent(
            #                 name="TestPlanGenerator",
            #                 description="Azure Foundry powered website exploration & test plan agent with MCP tools",
            #                 project_client=project_client,
            #                 deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            #                 instructions=self.get_system_message(),
            #                 tool_resources=mcp_tool.resources,
            #                 # tools=toolset.definitions,
            #                 metadata={"source": "AzureAIAgent"},
            #             )

                        
            #         except TypeError as te:
            #             print(f"⚠️ AzureAIAgent tools param unsupported ({te}); retrying without tools")
            #             self.azure_agent = AzureAIAgent(
            #                 name="TestPlanGenerator",
            #                 description="Azure Foundry powered website exploration agent",
            #                 project_client=project_client,
            #                 deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            #                 instructions=self.get_system_message(),
            #                 metadata={"source": "AzureAIAgent"},
            #             )
            #         print("✅ AzureAIAgent initialized")
            #     except Exception as ae:
            #         print(f"⚠️ Failed to init AzureAIAgent: {ae}; will fallback to AssistantAgent")
            #         self.azure_agent = None
            # else:
            #     print("ℹ️ Azure AI project env vars not set; using AssistantAgent fallback")
            # Fallback / legacy AssistantAgent (still uses workbench directly)
            if not self.azure_agent:
                self.agent = AssistantAgent(
                    name="TestPlanGenerator",
                    model_client=self.model_client,
                    system_message=self.get_system_message(),
                    workbench=self.workbench,
                    description="Specialized agent for systematic website exploration and test plan generation using MCP Playwright tools",
                    reflect_on_tool_use=True,
                    model_client_stream=True
                )
                print("✅ AssistantAgent (fallback) initialized with MCP Playwright tools")
            else:
                # Provide unified attribute for downstream code
                self.agent = self.azure_agent
                print("✅ Using AzureAIAgent as primary agent")
            
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
        return (
            'You are an autonomous Test Exploration Agent that generates '
            'detailed test cases from high-level user instructions using MCP '
            'Playwright tools.\n\n'

            'Be concise in progress updates that are emitted to the UI. Your '
            'internal reasoning can be thorough, but only concise progress '
            'summaries should be shown to the user. Do NOT emit '
            'chain-of-thought or detailed sub-agent logs to the UI.\n\n'

            'Your goals:\n'
            '1. Take simple user test steps and autonomously explore the '
            'target website using available MCP tools.\n'
            '2. Keep working in a loop until you can produce a fully '
            'enhanced, detailed test plan with Preconditions, Test Steps '
            '(step-by-step actions), and Expected Results.\n'
            '3. After every meaningful action or decision (navigation, '
            'click, type, wait, error, or discovery), provide a brief '
            'progress update such as:\n'
            '   - "Navigated to login page."\n'
            '   - "Clicked \'Sign In\' button."\n'
            '   - "Waiting for dashboard to load."\n'
            '   - "Could not find element, taking snapshot and retrying."\n'
            '4. Capture screenshots sparingly: take screenshots only when '
            'necessary for debugging, evidence, or when you need to inspect '
            'HTML attributes. Do NOT take screenshots after every minor '
            'action.\n'
            '5. Use `mcp_playwright_browser_snapshot` only when you need to '
            'inspect element attributes or when a selector fails to locate an '
            'element. Use snapshots selectively to aid element discovery.\n'
            '6. If clarification or missing test data is required, ask the '
            'user for input and then continue once the user responds.\n'
            '7. Return the final documentation in Markdown format and include '
            'Preconditions, Test Steps, and Expected Results.\n\n'

            '### 🔹 Available MCP Tools\n\n'
            'Use the following MCP tools for browser exploration:\n'
            '- mcp_playwright_browser_navigate: Navigate to URLs\n'
            '- mcp_playwright_browser_snapshot: Capture page snapshots for '
            'element discovery\n'
            '- mcp_playwright_browser_click: Click on elements\n'
            '- mcp_playwright_browser_type: Type text into input fields\n'
            '- mcp_playwright_browser_take_screenshot: Take screenshots\n'
            '- mcp_playwright_browser_hover: Hover over elements\n'
            '- mcp_playwright_browser_select_option: Select dropdown options\n'
            '- mcp_playwright_browser_press_key: Press keyboard keys\n'
            '- mcp_playwright_browser_evaluate: Execute JavaScript\n'
            '- mcp_playwright_browser_wait_for: Wait for conditions\n'
            '- mcp_playwright_browser_resize: Resize browser window\n'
            '- mcp_playwright_browser_navigate_back/forward: Navigate '
            'browser history\n'
            '- mcp_playwright_browser_tab_*: Tab management tools\n\n'
            '### 🔹 Execution Strategy\n\n'
            '- Start with mcp_playwright_browser_navigate to open the '
            'portal.\n'
            '- At each step:\n'
            '  - Perform the action using appropriate MCP tools.\n'
            '  - If element not found → use mcp_playwright_browser_snapshot, '
            'retry with updated selector, and report progress.\n'
            '  - If blocked → ask user for futher input\n\n'
            'IMPORTANT: Keep your progress updates concise and focused on '
            'actions taken. The orchestrator will handle the formatting for '
            'the UI.'
        )

# Global variables for team management
website_agent = None
progress_manager = None
state_path = "team_state.json"
history_path = "team_history.json"


# No progress checklist or aggregation: show orchestrator output directly

async def get_filtered_team(
    user_input_func: Callable[
        [str, Optional[CancellationToken]], Awaitable[str]
    ]
) -> Any:
    """Create and return a ProgressOnly MagenticOne team with a user input function."""
    global website_agent, progress_manager
    try:
        # Initialize the WebsiteExploreAgent if not already done
        if website_agent is None:
            website_agent = WebsiteExploreAgent()
            await website_agent.initialize()

        # Check if initialization was successful
        if website_agent.agent is None:
            raise RuntimeError(
                "Failed to initialize WebsiteExploreAgent - agent is None"
            )

        agent = website_agent.agent

        user_proxy = UserProxyAgent(
            name="user",
            input_func=user_input_func,  # Use the user input function.
        )

        # Create the ProgressOnly MagenticOneGroupChat directly
        team = CustomMagneticOneGroupChat(
            [agent, user_proxy],
            model_client=website_agent.model_client,
            max_turns=20,
            emit_team_events=False,
        )

    # Expose the active team to the progress callback for possible persistence
        progress_manager = team

        logger.debug(
            "Created progress-only team with participants: agent=%s, user_proxy=%s",
            agent,
            user_proxy,
        )

        return team
    except Exception as e:
        logger.error(f"Error creating filtered team: {e}")
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

async def save_team_state(team: Any):
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
    """Initialize the filtered AutoGen team when chat starts"""
    
    # await cl.Message(content="🚀 Initializing AutoGen MagenticOne Team with Prompt-Based Progress Filtering...").send()
    
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
        # if history:
        #     await cl.Message(content=f"📜 Loaded {len(history)} previous messages").send()
        
        await cl.Message(
            content="How can I help you today !!",
            author="Testing Agent"
        ).send()
        
    except Exception as e:
        await cl.Message(content=f"❌ Error during initialization: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and process them through filtered AutoGen team"""
    global current_progress_msg
    user_message = message.content.strip()
    
    # Queue for user input requests from the agent
    user_input_queue = asyncio.Queue()
    current_user_input_request = None
    
    # User input function for the team
    async def user_input_func(prompt: str, cancellation_token: CancellationToken | None) -> str:
        nonlocal current_user_input_request
        
        try:
            # Send user input request to Chainlit
            current_user_input_request = await cl.AskUserMessage(
                content=f"🤔 **Agent needs input:**\n\n{prompt}",
                timeout=300  # 5 minutes timeout
            ).send()
            
            if current_user_input_request:
                # Extract the actual user response
                if isinstance(current_user_input_request, dict):
                    # Check different possible keys for the response
                    response = current_user_input_request.get('output') or \
                              current_user_input_request.get('content') or \
                              current_user_input_request.get('response') or \
                              str(current_user_input_request)
                    
                    logger.info(f"📝 User input extracted: {response}")
                    return response
                else:
                    return str(current_user_input_request)
            else:
                return "No response received"
                
        except Exception as e:
            logger.error(f"Error getting user input: {e}")
            return f"Error getting user input: {e}"
    
    try:
        # Disable user input while the agent is processing
        try:
            await cl.context.emitter.task_start()
        except Exception as e:
            logger.debug(f"Could not send task_start: {e}")

        # Show initial processing message
        await cl.Message(content="🤖 AI Thinking...").send()

        # Get the filtered team
        team = await get_filtered_team(user_input_func)

        # Create TextMessage for team processing
        text_message = TextMessage(content=user_message, source="user")

        # Get chat history
        history = await get_history()

        # Process message through filtered AutoGen team
        response_messages = []

        # Use the orchestrator-only stream (drop participant-origin events)
        stream = _orchestrator_only_run_stream(team, task=text_message)

        async for msg in stream:
            logger.debug(f"Received orchestrator stream event: type={type(msg).__name__} source={getattr(msg,'source',None)} response_present={hasattr(msg,'response')}")
            try:
                if isinstance(msg, TaskResult):
                    # Task completed
                    await cl.Message(
                        content="✅ **Task Completed Successfully!**\n\n"
                        "The agent has finished executing your test scenario."
                    ).send()
                    continue

                if isinstance(msg, UserInputRequestedEvent):
                    # This should be handled by the user_input_func
                    logger.info("📤 UserInputRequestedEvent detected in main stream")
                    continue

                # Only show filtered messages that made it through the orchestrator
                # Handle GroupChatAgentResponse-like events that wrap a chat_message
                response_obj = getattr(msg, 'response', None)
                if response_obj is not None and getattr(response_obj, 'chat_message', None) is not None:
                    inner_msg = response_obj.chat_message
                    message_data = safe_model_dump(inner_msg)
                else:
                    message_data = safe_model_dump(msg)

                message_type = message_data.get('type', '')
                source = message_data.get('source', '')
                content = message_data.get('content', '')


                manager_name = getattr(team, '_group_chat_manager_name', None)
                # Always use 'Testing Agent' as author for manager-origin messages
                if source == manager_name and isinstance(content, str):
                    await cl.Message(content=content, author='Testing Agent').send()
                elif len(str(content).strip()) > 10:
                    agent_name = source.replace('_', ' ').title()
                    await cl.Message(content=f"**{agent_name}:**\n\n{content}", author=agent_name).send()

                response_messages.append(message_data)
                history.append(message_data)
            except Exception as e:
                logger.error(f"Error processing filtered message: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

        # Show final progress as a new message
        await cl.Message(content="✅ **Processing Complete!**\n\nTask execution finished.").send()

        # Re-enable user input when the run finishes
        try:
            await cl.context.emitter.task_end()
        except Exception as e:
            logger.debug(f"Could not send task_end: {e}")

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

        # Ensure UI input is re-enabled on error
        try:
            await cl.context.emitter.task_end()
        except Exception:
            pass

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

async def _orchestrator_only_run_stream(team, task: str):
    """Wrapper around team.run_stream that only yields events originating from the
    orchestrator (the group chat manager). This provides a strong guarantee that
    participant-origin messages never reach the UI.

    The wrapper inspects event types and the `source` fields. It yields:
    - TextMessage or other chat messages whose `.source` == team's group chat manager name
    - GroupChatAgentResponse events that contain a chat_message whose source == manager name
    - UserInputRequestedEvent when issued by the manager
    """
    manager_name = getattr(team, "_group_chat_manager_name", None)
    async for evt in team.run_stream(task=task):
        try:
            # GroupChatAgentResponse-like objects have a `.response.chat_message`
            response = getattr(evt, "response", None)
            if response is not None and getattr(response, "chat_message", None) is not None:
                msg = response.chat_message
                if getattr(msg, "source", None) == manager_name:
                    yield evt
                else:
                    logger.debug(f"Dropping GroupChatAgentResponse from participant: type={type(evt).__name__} source={getattr(msg,'source',None)}")
                    continue

            # Direct chat messages (TextMessage, StopMessage, etc.)
            if getattr(evt, "source", None) == manager_name:
                yield evt
                continue

            # User input requests from manager
            if isinstance(evt, UserInputRequestedEvent) and getattr(evt, "source", None) == manager_name:
                yield evt
                continue

            # Otherwise drop the event (participant-origin)
            logger.debug(f"Dropping event from participant: type={type(evt).__name__} source={getattr(evt,'source',None)}")
            continue
        except Exception:
            # Be conservative: on errors skip the event
            logger.exception("Error while filtering orchestrator stream event")
            continue


# --- FastAPI API for programmatic access ---
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel

# Only create FastAPI app if not running under Chainlit
import sys
if "chainlit" not in sys.argv[0]:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Explicit catch-all OPTIONS handler for preflight across all API paths
    @app.options("/{rest_of_path:path}")
    async def preflight_handler(request: Request, rest_of_path: str) -> Response:
        requested_headers = request.headers.get("Access-Control-Request-Headers", "")
        allow_headers = requested_headers or (
            "Authorization, Content-Type, X-Requested-With, Accept, Origin, User-Agent, DNT, Cache-Control, X-Mx-ReqToken"
        )
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": allow_headers,
            "Access-Control-Max-Age": "86400",
        }
        return Response(status_code=204, headers=headers)

    class TaskRequest(PydanticBaseModel):
        message: str

    class InitRequest(PydanticBaseModel):
        endpoint: str
        deployment_name: str
        model: Optional[str] = None  # Valid OpenAI model name (e.g., gpt-4o)
        api_version: Optional[str] = None
        api_key: Optional[str] = None

    @app.post("/init")
    async def initialize_config(req: InitRequest):
        """Initialize/update Azure OpenAI configuration for subsequent tasks.

        Sets process environment variables so future agent initializations use
        them:
        - AZURE_OPENAI_ENDPOINT
        - AZURE_OPENAI_DEPLOYMENT_NAME
        - AZURE_OPENAI_API_VERSION
        - AZURE_OPENAI_MODEL
        """
        # Basic validation
        endpoint = (req.endpoint or "").strip()
        deployment = (req.deployment_name or "").strip()
        api_version = (req.api_version or "2024-06-01").strip()
        model = (req.model or os.getenv("AZURE_OPENAI_MODEL") or "gpt-4o").strip()
        api_key = (req.api_key or "").strip()
        if not endpoint or not deployment:
            raise HTTPException(
                status_code=400,
                detail="'endpoint' and 'deployment_name' are required",
            )

        # Set environment variables for subsequent use in this process
        os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = deployment
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version
        if api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
        os.environ["AZURE_OPENAI_MODEL"] = model

        # Reset any existing agent so the next task uses the new config
        global website_agent
        if website_agent is not None:
            try:
                await website_agent.cleanup()
            except Exception:
                # Non-fatal; ensure we drop the old instance
                logger.exception(
                    "Error cleaning up existing WebsiteExploreAgent during /init"
                )
            website_agent = None

        return {
            "status": "ok",
            "endpoint": endpoint,
            "deployment_name": deployment,
            "api_version": api_version,
            "model": model,
            "api_key_set": bool(api_key),
        }

    # --- Execute Playwright script against existing CDP browser ---
    class ExecuteScriptRequest(PydanticBaseModel):
        script: str  # Python or JavaScript Playwright code. See docstring for contract.
        timeout_ms: Optional[int] = 60000
        navigate_url: Optional[str] = None  # Optional initial navigation
        language: Optional[str] = None  # 'python' or 'javascript'; auto-detected if not provided

    @app.post("/execute")
    async def execute_playwright_script(req: ExecuteScriptRequest):
        """
        Execute a provided Python Playwright script against an existing browser over CDP.

        Contract:
    - Inputs: script (str), optional timeout_ms, navigate_url
        - The script can either:
            1) Define `async def run(page, context, browser, playwright, **kwargs): ...`
               and return a value; or
            2) Be a script body which will be wrapped into an async function with access to
               `page`, `context`, `browser`, `playwright`, `asyncio`, `time`.
        - Output: JSON with status, logs (stdout), and result or error.
        """
        # Use CDP endpoint from env, defaulting to local 9222
        cdp = (os.getenv("CDP_ENDPOINT") or "http://127.0.0.1:9222").strip()
        timeout_s = max(1, int((req.timeout_ms or 60000) / 1000))

        # Detect language
        lang = (req.language or "").strip().lower()
        script_text = req.script or ""
        if not lang:
            js_indicators = ["import ", "export ", "require(", "getByRole(", "module.exports", "async ({ page }) =>"]
            lang = "javascript" if any(tok in script_text for tok in js_indicators) else "python"

        # JavaScript execution path
        if lang == "javascript":
            # Ensure node exists
            if not shutil.which("node"):
                raise HTTPException(status_code=500, detail="Node.js is not available on PATH; cannot run JavaScript.")

            # Build JS runner that connects over CDP and executes user code
            user_code = script_text

            # If this looks like a Playwright Test spec, run it with the test runner
            if ("@playwright/test" in user_code) or ("require('@playwright/test')" in user_code):
                workdir = tempfile.mkdtemp(prefix='pwtest-')
                fixtures_path = os.path.join(workdir, 'fixtures.mjs')
                spec_path = os.path.join(workdir, 'spec.spec.mjs')
                pkg_path = os.path.join(workdir, 'package.json')
                config_path = os.path.join(workdir, 'playwright.config.mjs')

                fixtures_src = textwrap.dedent(
                                        """
                                        import { chromium, test as base, expect } from '@playwright/test';
                                        const CDP = process.env.CDP_ENDPOINT || 'http://127.0.0.1:9222';
                                        export const test = base.extend({
                                            browser: async ({}, use) => {
                                                const browser = await chromium.connectOverCDP(CDP);
                                                await use(browser);
                                                // shared browser is managed externally; do not close
                                            },
                                            context: async ({}, use) => {
                                                const browser = await chromium.connectOverCDP(CDP);
                                                let context = browser.contexts()[0];
                                                if (!context) context = await browser.newContext();
                                                await use(context);
                                            },
                                            page: async ({ context }, use) => {
                                                const page = await context.newPage();
                                                // Ensure a desktop-like viewport so responsive sites expose desktop nav
                                                try { await page.setViewportSize({ width: 1366, height: 900 }); } catch {}
                                                // Make navigation/assertions a bit more forgiving
                                                try { page.setDefaultTimeout(15000); page.setDefaultNavigationTimeout(30000); } catch {}
                                                const url = process.env.NAVIGATE_URL;
                                                if (url) {
                                                    console.log('[exec] Navigating to ' + url + '...');
                                                    await page.goto(url, { waitUntil: 'domcontentloaded' });
                                                    try { await page.waitForLoadState('domcontentloaded'); } catch {}
                                                }
                                                await use(page);
                                                try { await page.close(); } catch {}
                                            }
                                        });
                                        export { expect };
                                        """
                                )

                # Replace import to use our fixtures module so test() comes from fixtures
                transformed = user_code.replace("@playwright/test", "./fixtures.mjs")

                pkg_json = {
                    "name": "tmp-playwright-exec",
                    "private": True,
                    "type": "module"
                }
                config_src = textwrap.dedent(
                    """
                    /** Minimal config; runner connects over CDP via fixtures */
                    export default { timeout: 60000, expect: { timeout: 10000 } };
                    """
                )

                try:
                    with open(fixtures_path, 'w', encoding='utf-8') as f:
                        f.write(fixtures_src)
                    with open(spec_path, 'w', encoding='utf-8') as f:
                        f.write(transformed)
                    with open(pkg_path, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(pkg_json))
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(config_src)

                    proc_env = os.environ.copy()
                    proc_env['CDP_ENDPOINT'] = cdp
                    if req.navigate_url:
                        proc_env['NAVIGATE_URL'] = req.navigate_url
                    # Help Node find globally installed modules if present
                    proc_env['NODE_PATH'] = (proc_env.get('NODE_PATH', '') + ':/usr/lib/node_modules:/usr/local/lib/node_modules').strip(':')

                    # Ensure @playwright/test is available in temp project
                    create = asyncio.create_subprocess_exec
                    install_proc = await create('npm', 'i', '--no-audit', '--no-fund', '--silent', '@playwright/test@latest',
                                               cwd=workdir,
                                               stdout=asyncio.subprocess.PIPE,
                                               stderr=asyncio.subprocess.PIPE,
                                               env=proc_env)
                    try:
                        _outs, _errs = await asyncio.wait_for(install_proc.communicate(), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        try: install_proc.kill()
                        except Exception: pass
                        raise HTTPException(status_code=504, detail='Installing @playwright/test timed out')
                    if install_proc.returncode != 0:
                        raise HTTPException(status_code=500, detail='Failed to install @playwright/test')

                    # Use npx to run playwright test for this temp project
                    proc = await create('npx', '--yes', '@playwright/test', 'test', spec_path,
                                        '--reporter=json',
                                        cwd=workdir,
                                        stdout=asyncio.subprocess.PIPE,
                                        stderr=asyncio.subprocess.PIPE,
                                        env=proc_env)
                    try:
                        outs, errs = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        try: proc.kill()
                        except Exception: pass
                        raise HTTPException(status_code=504, detail='Playwright Test execution timed out')

                    stdout = outs.decode('utf-8', errors='replace') if outs else ''
                    stderr = errs.decode('utf-8', errors='replace') if errs else ''

                    # Try to parse JSON reporter output
                    report = None
                    try:
                        report = json.loads(stdout)
                    except Exception:
                        for line in stdout.splitlines()[::-1]:
                            line = line.strip()
                            if line.startswith('{') and line.endswith('}'):
                                try:
                                    report = json.loads(line)
                                    break
                                except Exception:
                                    continue

                    return {
                        'status': 'ok',
                        'exit_code': proc.returncode,
                        'logs': stdout + (('\n' + stderr) if stderr else ''),
                        'result': report,
                    }
                finally:
                    try:
                        shutil.rmtree(workdir, ignore_errors=True)
                    except Exception:
                        pass
            # If user provided ESM 'import' lines, it's likely to fail under CJS. Warn early.
            if "import " in user_code and "from" in user_code:
                # We still attempt execution, but many environments require a project/module type
                pass

            runner = textwrap.dedent(
                f"""
                (async () => {{
                  const playwright = require('playwright');
                  const cdp = process.env.CDP_ENDPOINT || 'http://127.0.0.1:9222';
                  const navigateUrl = process.env.NAVIGATE_URL || '';
                  const browser = await playwright.chromium.connectOverCDP(cdp);
                  let context = browser.contexts()[0];
                  if (!context) {{ context = await browser.newContext(); }}
                  const page = await context.newPage();
                  if (navigateUrl) {{
                    console.log('[exec] Navigating to ' + navigateUrl + '...');
                    await page.goto(navigateUrl);
                  }}
                  let __result;
                  try {{
                    // Provide run() entrypoint if defined; else wrap body
                    {''}
                    // Begin user code injection
                    {user_code}
                    // End user code injection

                    if (typeof run === 'function') {{
                      __result = await run({{ page, context, browser, playwright }});
                    }} else {{
                      __result = await (async () => {{
                        // If user provided top-level statements, they ran already.
                        // Allow them to set globalThis.__result to pass a value.
                        return globalThis.__result;
                      }})();
                    }}
                    console.log('__RESULT__' + JSON.stringify(__result));
                  }} catch (err) {{
                    console.error('__ERROR__' + (err && err.stack || err && err.message || String(err)));
                    process.exitCode = 1;
                  }} finally {{
                    try {{ await page.close(); }} catch {{}}
                  }}
                }})().catch(e => {{
                  console.error('__ERROR__' + (e && e.stack || e && e.message || String(e)));
                  process.exitCode = 1;
                }});
                """
            )

            # Write to temp file
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False)
            try:
                tmp.write(runner)
                tmp.flush()
                tmp.close()

                # Run with environment
                proc_env = os.environ.copy()
                proc_env['CDP_ENDPOINT'] = cdp
                if req.navigate_url:
                    proc_env['NAVIGATE_URL'] = req.navigate_url
                proc_env['NODE_PATH'] = (proc_env.get('NODE_PATH', '') + ':/usr/lib/node_modules:/usr/local/lib/node_modules').strip(':')

                create = asyncio.create_subprocess_exec
                proc = await create('node', tmp.name,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                    env=proc_env)
                try:
                    outs, errs = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    raise HTTPException(status_code=504, detail='JavaScript execution timed out')

                stdout = outs.decode('utf-8', errors='replace') if outs else ''
                stderr = errs.decode('utf-8', errors='replace') if errs else ''
                # Extract result
                result = None
                for line in stdout.splitlines():
                    if line.startswith('__RESULT__'):
                        try:
                            result = json.loads(line[len('__RESULT__'):])
                        except Exception:
                            result = line[len('__RESULT__'):]
                # Prefer JS result marker; else none
                if proc.returncode and not result:
                    # Try to surface error from stderr
                    err_line = None
                    for line in (stderr.splitlines() + stdout.splitlines()):
                        if line.startswith('__ERROR__'):
                            err_line = line[len('__ERROR__'):]
                            break
                    raise HTTPException(status_code=500, detail=err_line or 'JavaScript execution failed')

                return {
                    'status': 'ok',
                    'logs': stdout + (('\n' + stderr) if stderr else ''),
                    'result': result,
                }
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        # Python execution path (default)
        # Capture stdout from the user script
        stdout_buf = io.StringIO()
        page = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.connect_over_cdp(cdp)
                contexts = browser.contexts
                if not contexts:
                    # For CDP connections there is usually a persistent default context
                    raise HTTPException(status_code=500, detail="No browser contexts available over CDP.")
                context = contexts[0]
                page = await context.new_page()
                if req.navigate_url:
                    with contextlib.redirect_stdout(stdout_buf):
                        print(f"[exec] Navigating to {req.navigate_url}...")
                    await page.goto(req.navigate_url)

                # Prepare execution environment
                user_globals = {
                    "__builtins__": __builtins__,
                    "asyncio": asyncio,
                    "time": time,
                }
                user_locals = {}

                script = req.script

                # If script defines an async run(), use it; otherwise wrap
                if "async def run" in script:
                    with contextlib.redirect_stdout(stdout_buf):
                        print("[exec] Detected async run() entrypoint")
                    exec(script, user_globals, user_locals)
                    # Prefer locals, then globals
                    fn = user_locals.get("run") or user_globals.get("run")
                    if not callable(fn):
                        raise HTTPException(status_code=400, detail="run is not callable")
                    run_coro = fn(page=page, context=context, browser=browser, playwright=p)
                else:
                    # Indent the body into an async function wrapper
                    body = "\n".join([("    " + line) if line.strip() else line for line in script.splitlines()])
                    wrapped = (
                        "async def __user_fn__(page, context, browser, playwright, asyncio, time):\n" + body + "\n"
                    )
                    exec(wrapped, user_globals, user_locals)
                    fn = user_locals.get("__user_fn__")
                    run_coro = fn(page, context, browser, p, asyncio, time)

                # Execute with timeout and capture stdout
                with contextlib.redirect_stdout(stdout_buf):
                    result = await asyncio.wait_for(run_coro, timeout=timeout_s)

                # Try to JSON-encode the result, fallback to str
                try:
                    _ = json.dumps(result)
                    safe_result = result
                except Exception:
                    safe_result = str(result)

                return {
                    "status": "ok",
                    "logs": stdout_buf.getvalue(),
                    "result": safe_result,
                }
        except HTTPException:
            # Re-raise structured errors
            raise
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "status": "error",
                "error": str(e),
                "logs": stdout_buf.getvalue(),
            })
        finally:
            # Best-effort page cleanup; avoid closing the shared CDP browser
            try:
                if page is not None:
                    await page.close()
            except Exception:
                pass


    # In-memory task store: {task_id: {"status": str, "progress": [messages],
    # "created_at": datetime, ...}}
    import uuid
    from typing import Dict, List
    from typing_extensions import TypedDict

    class TaskEntry(TypedDict, total=False):
        status: str
        progress: List[dict]
        created_at: str
        error: Optional[str]
        input_prompt: Optional[str]
        input_response: Optional[str]
        _resume_event: asyncio.Event

    task_store: Dict[str, TaskEntry] = {}


    class TaskStatus(str):
        PENDING = "pending"
        RUNNING = "running"
        WAITING_FOR_INPUT = "waiting_for_input"
        COMPLETED = "completed"
        ERROR = "error"


    @app.post("/task")
    async def create_task(req: TaskRequest):
        """Create a new orchestrator task. Returns a task_id."""
        task_id = str(uuid.uuid4())
        task_store[task_id] = {
            "status": TaskStatus.PENDING,
            "progress": [],
            "created_at": datetime.now().isoformat(),
            "error": None,
            "input_prompt": None,
            "input_response": None,
            "_resume_event": asyncio.Event(),
        }

        async def user_input_func(
            prompt: str, cancellation_token: CancellationToken | None
        ) -> str:
            # Set status to waiting, store prompt, and wait for resume
            task_store[task_id]["status"] = TaskStatus.WAITING_FOR_INPUT
            task_store[task_id]["input_prompt"] = prompt
            task_store[task_id]["_resume_event"].clear()
            await task_store[task_id]["_resume_event"].wait()
            # After resume, return the provided input
            return task_store[task_id]["input_response"] or ""

        async def run_orchestrator_task(task_id, user_message):
            try:
                task_store[task_id]["status"] = TaskStatus.RUNNING
                team = await get_filtered_team(user_input_func)
                text_message = TextMessage(content=user_message, source="user")
                stream = _orchestrator_only_run_stream(team, task=text_message)
                async for msg in stream:
                    response_obj = getattr(msg, 'response', None)
                    if response_obj is not None and getattr(
                        response_obj, 'chat_message', None
                    ) is not None:
                        inner_msg = response_obj.chat_message
                        message_data = safe_model_dump(inner_msg)
                    else:
                        message_data = safe_model_dump(msg)
                    # If this is a UserInputRequestedEvent, status will be set
                    # by user_input_func
                    if isinstance(msg, UserInputRequestedEvent):
                        # Progress and prompt handled in user_input_func
                        pass
                    else:
                        task_store[task_id]["progress"].append(message_data)
                # Only set to completed if not waiting for input
                if task_store[task_id]["status"] != TaskStatus.WAITING_FOR_INPUT:
                    task_store[task_id]["status"] = TaskStatus.COMPLETED
            except Exception as e:
                logger.error(f"Error in task {task_id}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                task_store[task_id]["status"] = TaskStatus.ERROR
                task_store[task_id]["error"] = str(e)

        # Start the orchestrator task in the background
        asyncio.create_task(run_orchestrator_task(task_id, req.message))
        return {"task_id": task_id, "status": TaskStatus.PENDING}

    class InputRequest(PydanticBaseModel):
        input: str

    @app.post("/task/{task_id}/input")
    async def provide_input(task_id: str, req: InputRequest):
        """Provide input to a waiting task and resume it.

        Also logs the user input in progress timeline.
        """
        task = task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task["status"] != TaskStatus.WAITING_FOR_INPUT:
            raise HTTPException(
                status_code=400,
                detail="Task is not waiting for input",
            )
        task["input_response"] = req.input
        # Log the user input as a progress event
        task["progress"].append({
            "type": "UserInputResponse",
            "source": "user",
            "content": req.input,
            "created_at": datetime.now().isoformat()
        })
        task["_resume_event"].set()
        # Set status back to running
        task["status"] = TaskStatus.RUNNING
        return {"task_id": task_id, "status": TaskStatus.RUNNING}

    @app.get("/task/{task_id}")
    async def get_task_status(task_id: str):
        """Get the status and progress of a task by task_id."""
        task = task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "created_at": task["created_at"],
            "error": task.get("error"),
            "input_prompt": task.get("input_prompt"),
        }

    # --- HTTP: fetch latest screenshot ---
    @app.get("/screenshot")
    async def get_screenshot(format: str = "binary"):
        """Return the latest screenshot.
        Query param 'format':
          - binary (default): returns image/png
          - base64: returns JSON { ts, image_b64 }
        """
        file_path = os.getenv("SCREENSHOT_FILE", "/app/screenshots/latest.png")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Screenshot not found")
        stat = os.stat(file_path)
        if format.lower() == "base64":
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("ascii")
                return JSONResponse({
                    "ts": int(stat.st_mtime * 1000),
                    "image_b64": "data:image/png;base64," + b64
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read screenshot: {e}")
        return FileResponse(file_path, media_type="image/png")

    @app.get("/screenshots")
    async def list_screenshots(limit: int = 50, include: str = "meta"):
        """List historical screenshots captured by the loop.
        Query params:
          - limit: max number of recent entries (default 50)
          - include:
              meta (default): return filenames + timestamps
              base64: include base64 content (be cautious: large)
        """
        directory = os.getenv("SCREENSHOT_DIR", "/app/screenshots")
        if not os.path.isdir(directory):
            return {"screenshots": []}
        files = [f for f in os.listdir(directory) if f.startswith("shot_") and f.endswith('.png')]
        # Sort by timestamp embedded in filename
        def ts_from_name(name: str) -> int:
            try:
                return int(name.split('_')[1].split('.')[0])
            except Exception:
                return 0
        files.sort(key=ts_from_name, reverse=True)
        selected = files[:limit]
        results = []
        for fname in selected:
            path = os.path.join(directory, fname)
            try:
                stat = os.stat(path)
                entry = {
                    "file": fname,
                    "ts": ts_from_name(fname),
                    "size": stat.st_size,
                }
                if include == "base64":
                    try:
                        with open(path, 'rb') as f:
                            b64 = base64.b64encode(f.read()).decode('ascii')
                        entry["image_b64"] = "data:image/png;base64," + b64
                    except Exception as e:
                        entry["error"] = f"read_failed: {e}"
                results.append(entry)
            except FileNotFoundError:
                continue
        return {"count": len(results), "screenshots": results}

    if __name__ == "__main__":
        import uvicorn
        print("🚀 Starting FastAPI app for MagenticOne orchestrator APIs...")
        uvicorn.run(app, host="0.0.0.0", port=8002)
else:
    if __name__ == "__main__":
        print("🚀 Starting AutoGen MagenticOne Team with Prompt-Based Progress Filtering + Chainlit...")
        print("🌐 Chainlit app will be available at: http://localhost:8000")
        print("📋 Features:")
        print("  - Prompt-enhanced Magnetic One orchestrator with AI-generated progress summaries")
        print("  - Clean, structured progress updates (Just Completed / Currently Doing / Next Step)")
        print("  - Orchestrator-only message display (sub-agent messages filtered out)")
        print("  - Playwright MCP integration for live browser automation")
        print("  - Interactive chat interface with Chainlit")
        print("  - Contextual AI-driven progress updates instead of hardcoded keywords")
        print("  - Smart orchestration with user-friendly communication")
        print()
        # Run chainlit from virtual environment
        import subprocess
        subprocess.run(["/Users/krishnachandak/Documents/repos/testing-autogen/env/bin/chainlit", "run", __file__, "--port", "8000"])
