import json
import logging
import os
import asyncio
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from pathlib import Path

# ChainLit imports (keeping for potential future use)
import chainlit as cl
from chainlit import Message, AskUserMessage, AskFileMessage, Image

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent
from autogen_agentchat.base import Response
from autogen_core import CancellationToken, AgentId, SingleThreadedAgentRuntime
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams
from azure.identity import DefaultAzureCredential

import aiofiles
import yaml
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

import os 
# load env
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
import traceback

logger = logging.getLogger("Logger")

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
            
            # Initialize MCP workbench
            self.workbench = McpWorkbench(self.playwright_server_params)
            await self.workbench.start()
            
            print("✅ MCP Playwright server connected")
            
            # Get tools from MCP workbench (await the coroutine)
            mcp_tools = await self.workbench.list_tools()
            print(f"🔧 Retrieved {len(mcp_tools)} MCP tools")
            print(f"🔧 First tool type: {type(mcp_tools[0]) if mcp_tools else 'No tools'}")
            print(f"🔧 First tool content: {mcp_tools[0] if mcp_tools else 'No tools'}")
            
            # Create simple wrapper functions for key MCP tools
            async def browser_navigate(url: str) -> str:
                """Navigate to a URL"""
                result = await self.workbench.call_tool('browser_navigate', {'url': url})
                return str(result)
            
            async def browser_snapshot() -> str:
                """Capture accessibility snapshot of the current page"""
                result = await self.workbench.call_tool('browser_snapshot', {})
                return str(result)
            
            async def browser_click(element: str, ref: str) -> str:
                """Click on an element"""
                result = await self.workbench.call_tool('browser_click', {'element': element, 'ref': ref})
                return str(result)
            
            async def browser_type(element: str, ref: str, text: str) -> str:
                """Type text into an element"""
                result = await self.workbench.call_tool('browser_type', {'element': element, 'ref': ref, 'text': text})
                return str(result)
            
            async def browser_take_screenshot() -> str:
                """Take a screenshot of the current page"""
                result = await self.workbench.call_tool('browser_take_screenshot', {})
                return str(result)
            
            # Create the AssistantAgent with simple wrapper tools
            self.agent = AssistantAgent(
                name="TestPlanGenerator",
                model_client=self.model_client,
                tools=[browser_navigate, browser_snapshot, browser_click, browser_type, browser_take_screenshot],
                system_message=self.get_system_message(),
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
7. If clarification or missing test data is required, call:
    request_user_input(message="What do you want me to do here?")
    Then continue once input is received.
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

You are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.


- Continue until the test flow is completed end-to-end.
- Save a structured enhanced test case (Markdown).'''

class ExtendedTextMessage(BaseModel):
    """Extended TextMessage with additional fields for WebSocket coordination."""
    content: str
    source: str = "user"
    is_user_input_response: bool = False

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model_config_path = "model_config.yaml"
state_path = "team_state.json"
history_path = "team_history.json"

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    """Serve the chat interface HTML file."""
    return FileResponse("app_team.html")


async def get_team(
    user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]],
) -> RoundRobinGroupChat:
    try:
        # Get model client from config.
        # async with aiofiles.open(model_config_path, "r") as file:
        #     model_config = yaml.safe_load(await file.read())
        # model_client = ChatCompletionClient.load_component(model_config)

        # Initialize the WebsiteExploreAgent properly
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
        
        team = RoundRobinGroupChat(
            [agent, user_proxy],
        )
        # Load state from file.
        # if not os.path.exists(state_path):
        #     return team
        # async with aiofiles.open(state_path, "r") as file:
        #     state = json.loads(await file.read())
        # await team.load_state(state)
        return team
    except Exception as e:
        logger.error(f"Error creating team: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


async def get_history() -> list[dict[str, Any]]:
    """Get chat history from file."""
    if not os.path.exists(history_path):
        return []
    async with aiofiles.open(history_path, "r") as file:
        return json.loads(await file.read())


@app.get("/history")
async def history() -> list[dict[str, Any]]:
    try:
        return await get_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    
    # Create a message queue to coordinate between main loop and user input
    message_queue = asyncio.Queue()
    user_input_queue = asyncio.Queue()
    websocket_closed = asyncio.Event()
    
    # Background task to handle incoming WebSocket messages
    async def message_receiver():
        try:
            while not websocket_closed.is_set():
                data = await websocket.receive_json()
                message = ExtendedTextMessage.model_validate(data)
                # Check if this is a response to a user input request
                if message.is_user_input_response:
                    await user_input_queue.put(message.content)
                else:
                    # Convert to regular TextMessage for team processing
                    text_message = TextMessage(content=message.content, source=message.source)
                    await message_queue.put(text_message)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            websocket_closed.set()
        except Exception as e:
            logger.error(f"Error in message receiver: {e}")
            websocket_closed.set()

    # Start the message receiver task
    receiver_task = asyncio.create_task(message_receiver())

    # User input function used by the team.
    async def _user_input(prompt: str, cancellation_token: CancellationToken | None) -> str:
        try:
            # Send prompt to client
            await websocket.send_json({
                "type": "user_input_request",
                "content": prompt
            })
            
            # Wait for user response with timeout
            timeout = 300  # 5 minutes timeout
            try:
                user_response = await asyncio.wait_for(
                    user_input_queue.get(), 
                    timeout=timeout
                )
                return user_response
            except asyncio.TimeoutError:
                return "No response received within timeout"
                
        except WebSocketDisconnect:
            logger.info("Client disconnected while waiting for user input")
            raise
        except Exception as e:
            logger.error(f"Error getting user input: {e}")
            return f"Error getting user input: {e}"

    try:
        while not websocket_closed.is_set():
            try:
                # Get user message from queue with timeout
                request = await asyncio.wait_for(message_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # Check if websocket is still open

            try:
                # Get the team and respond to the message.
                team = await get_team(_user_input)
                history = await get_history()
                stream = team.run_stream(task=request)
                
                # Buffer for collecting streaming content
                message_buffer = {}
                
                async for message in stream:
                    
                    if isinstance(message, TaskResult):
                        continue
                    
                    # Safely serialize message for WebSocket
                    try:
                        logger.debug(f"Processing message: type={type(message)}, message={message}")
                        message_data = safe_model_dump(message)
                        
                        # Check if this is a streaming content message that should be buffered
                        message_type = message_data.get('type', '')
                        source = message_data.get('source', '')
                        content = message_data.get('content', '')
                        
                        # Add detailed debug logging
                        logger.info(f"📨 Message details: type='{message_type}', source='{source}', content_length={len(content) if content else 0}")
                        logger.info(f"📨 Is UserInputRequestedEvent: {isinstance(message, UserInputRequestedEvent)}")
                        
                        # Handle different message types appropriately
                        if isinstance(message, UserInputRequestedEvent):
                            # Send user input requests immediately
                            logger.info("📤 Sending UserInputRequestedEvent immediately")
                            await websocket.send_json(message_data)
                        elif message_type == 'ModelClientStreamingChunkEvent' and source and content:
                            # Buffer streaming chunks to prevent token-by-token display
                            logger.info(f"📦 Buffering streaming chunk from {source}")
                            if source not in message_buffer:
                                message_buffer[source] = {
                                    'type': 'TextMessage',  # Convert to TextMessage type
                                    'source': source,
                                    'content': '',
                                    'models_usage': message_data.get('models_usage'),
                                    'metadata': message_data.get('metadata', {}),
                                    'created_at': message_data.get('created_at'),
                                    'request_id': message_data.get('request_id')
                                }
                            # Append content to buffer
                            message_buffer[source]['content'] += content
                        elif message_type == 'TextMessage' and source and content:
                            # Send complete TextMessage immediately (this is the final message)
                            logger.info(f"📤 Sending complete TextMessage from {source}")
                            await websocket.send_json(message_data)
                            history.append(message_data)
                        else:
                            # Send other messages immediately (system messages, non-content messages, etc.)
                            logger.info(f"📤 Sending other message immediately: type='{message_type}', source='{source}'")
                            await websocket.send_json(message_data)
                            history.append(message_data)
                                
                    except Exception as send_error:
                        logger.error(f"Error sending message via WebSocket: {send_error}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        logger.error(f"Message that caused error: type={type(message) if 'message' in locals() else 'undefined'}, message={message if 'message' in locals() else 'undefined'}")
                        # Send a simple error message instead
                        error_msg = {
                            "type": "error",
                            "content": f"Failed to send message: {str(send_error)}",
                            "source": "system"
                        }
                        await websocket.send_json(error_msg)
                
                # Send buffered complete messages
                for source, buffered_message in message_buffer.items():
                    if buffered_message['content'].strip():  # Only send if there's actual content
                        await websocket.send_json(buffered_message)
                        history.append(buffered_message)

                # Save team state to file.
                async with aiofiles.open(state_path, "w") as file:
                    state = await team.save_state()
                    await file.write(safe_json_dumps(state))

                # Save chat history to file.
                async with aiofiles.open(history_path, "w") as file:
                    await file.write(safe_json_dumps(history))
                    
            except WebSocketDisconnect:
                # Client disconnected during message processing - exit gracefully
                logger.info("Client disconnected during message processing")
                break
            except Exception as e:
                # Log detailed error information
                logger.error(f"Error in message processing: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Send error message to client
                error_message = {
                    "type": "error",
                    "content": f"Error: {str(e)}",
                    "source": "system"
                }
                try:
                    await websocket.send_json(error_message)
                    # Re-enable input after error
                    await websocket.send_json({
                        "type": "UserInputRequestedEvent",
                        "content": "An error occurred. Please try again.",
                        "source": "system"
                    })
                except WebSocketDisconnect:
                    # Client disconnected while sending error - exit gracefully
                    logger.info("Client disconnected while sending error message")
                    break
                except Exception as send_error:
                    logger.error(f"Failed to send error message: {str(send_error)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Unexpected error: {str(e)}",
                "source": "system"
            })
        except WebSocketDisconnect:
            # Client already disconnected - no need to send
            logger.info("Client disconnected before error could be sent")
        except Exception:
            # Failed to send error message - connection likely broken
            logger.error("Failed to send error message to client")
            pass
    finally:
        # Clean up tasks
        websocket_closed.set()
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass


# Example usage
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)