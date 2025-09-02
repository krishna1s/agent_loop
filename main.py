#!/usr/bin/env python3
"""
TestingAgent Chainlit entrypoint using the ProgressOnly Magentic-One orchestrator.

This app initializes an AssistantAgent with MCP Playwright tools and runs a
ProgressOnlyMagenticOneGroupChat so the UI receives concise, structured
"PROGRESS UPDATE" messages emitted by the orchestrator only.
"""

import json
import logging
import os
import asyncio
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from pathlib import Path
from autogen_ext.agents.azure._azure_ai_agent import AzureAIAgent
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential

import time

# Chainlit imports
import chainlit as cl

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent, ModelClientStreamingChunkEvent
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams
from azure.identity import DefaultAzureCredential

# Import our custom orchestrator
from TestingAgent import CustomMagneticOneGroupChat

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import aiofiles
import yaml
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.INFO)

logger = logging.getLogger("AutoGenChainlitFiltered")

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
            url="http://playwrightmcp.eastus.azurecontainer.io:4000/sse",
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
            
            project_client = AIProjectClient(  # type: ignore
                credential=DefaultAzureCredential(), endpoint="https://foundyuedn.services.ai.azure.com/api/projects/projectuedn"
            )

            # self.azure_Agent  = AzureAIAgent(
            #     name="bing_agent",
            #     description="An AI assistant with Bing grounding",
            #     project_client=project_client,
            #     deployment_name="gpt-4o",
            #     instructions="You are a helpful assistant.",
            #     workbench=self.workbench,
            #     metadata={"source": "AzureAIAgent"},
            # )
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

Be concise in progress updates that are emitted to the UI. Your internal reasoning can be thorough, but only concise progress summaries should be shown to the user. Do NOT emit chain-of-thought or detailed sub-agent logs to the UI.

Your goals:
1. Take simple user test steps and autonomously explore the target website using available MCP tools. 
2. Keep working in a loop until you can produce a fully enhanced, detailed test plan with Preconditions, Test Steps (step-by-step actions), and Expected Results.
3. After every meaningful action or decision (navigation, click, type, wait, error, or discovery), provide a brief progress update such as:
   - "Navigated to login page."
   - "Clicked 'Sign In' button."
   - "Waiting for dashboard to load."
   - "Could not find element, taking snapshot and retrying."
4. Capture screenshots sparingly: take screenshots only when necessary for debugging, evidence, or when you need to inspect HTML attributes. Do NOT take screenshots after every minor action.
5. Use `mcp_playwright_browser_snapshot` only when you need to inspect element attributes or when a selector fails to locate an element. Use snapshots selectively to aid element discovery.
6. If clarification or missing test data is required, ask the user for input and then continue once the user responds.
7. Return the final documentation in Markdown format and include Preconditions, Test Steps, and Expected Results.

### 🔹 Available MCP Tools

Use the following MCP tools for browser exploration:
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


IMPORTANT: Keep your progress updates concise and focused on actions taken. The orchestrator will handle the formatting for the UI.'''

# Global variables for team management
website_agent = None
progress_manager = None
state_path = "team_state.json"
history_path = "team_history.json"


# No progress checklist or aggregation: show orchestrator output directly

async def get_filtered_team(user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]]) -> Any:
    """Create and return a ProgressOnly MagenticOne team with user input function"""
    global website_agent, progress_manager
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

        # Create the ProgressOnly MagenticOneGroupChat directly
        team = CustomMagneticOneGroupChat(
            [agent, user_proxy],
            model_client=website_agent.model_client,
            max_turns=20,
            emit_team_events=False,
        )

        # Expose the active team to the progress callback so we can persist updates
        progress_manager = team

        logger.debug(f"Created progress-only team with participants: agent={agent}, user_proxy={user_proxy}")

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


    # In-memory task store: {task_id: {"status": str, "progress": [messages], "created_at": datetime, ...}}
    import uuid
    from typing import Dict
    task_store: Dict[str, dict] = {}


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

        async def user_input_func(prompt: str, cancellation_token: CancellationToken | None) -> str:
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
                    if response_obj is not None and getattr(response_obj, 'chat_message', None) is not None:
                        inner_msg = response_obj.chat_message
                        message_data = safe_model_dump(inner_msg)
                    else:
                        message_data = safe_model_dump(msg)
                    # If this is a UserInputRequestedEvent, status will be set by user_input_func
                    if isinstance(msg, UserInputRequestedEvent):
                        # Progress and prompt already handled in user_input_func
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
        """Provide input to a waiting task and resume it. Also log the input in progress."""
        task = task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task["status"] != TaskStatus.WAITING_FOR_INPUT:
            raise HTTPException(status_code=400, detail="Task is not waiting for input")
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
