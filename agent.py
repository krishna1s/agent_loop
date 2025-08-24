import json
import logging
import os
import asyncio
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from pathlib import Path

# ChainLit imports (keeping for potential future use)
# import chainlit as cl
# from chainlit import Message, AskUserMessage, FileMessage, Image

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
from autogen_agentchat.agents import AssistantAgent
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
            
            # Initialize MCP workbench with Playwright server
            try:
                self.workbench = McpWorkbench(self.playwright_server_params)
                await self.workbench.start()
                print("✅ MCP Playwright server connected")
                
                # Get available MCP tools
                mcp_tools_raw = await self.workbench.list_tools()
                print(f"✅ Available MCP tools: {[tool['name'] for tool in mcp_tools_raw]}")
                print(f"✅ Total MCP tools available: {len(mcp_tools_raw)}")
                
            except Exception as mcp_error:
                logger.error(f"Failed to connect to MCP server: {mcp_error}")
                print(f"❌ Failed to connect to MCP server: {mcp_error}")
                print("Make sure the MCP Playwright server is running on http://localhost:8931/sse")
                # Continue without MCP tools if server is not available
                self.workbench = None
            
            # Create the AssistantAgent with workbench (MCP tools will be available through workbench)
            if self.workbench:
                # Create agent that can use MCP tools through the workbench
                from autogen_agentchat.agents import AssistantAgent
                
                # Create custom tool functions that call MCP tools
                async def navigate_to_url(url: str) -> str:
                    """Navigate to a specific URL"""
                    try:
                        result = await self.workbench.call_tool("browser_navigate", {"url": url})
                        return str(result)
                    except Exception as e:
                        return f"Error navigating to {url}: {str(e)}"
                
                async def take_screenshot() -> str:
                    """Take a screenshot of the current page"""
                    try:
                        result = await self.workbench.call_tool("browser_take_screenshot", {})
                        return str(result)
                    except Exception as e:
                        return f"Error taking screenshot: {str(e)}"
                
                async def get_page_snapshot() -> str:
                    """Get accessibility snapshot of the current page"""
                    try:
                        result = await self.workbench.call_tool("browser_snapshot", {})
                        return str(result)
                    except Exception as e:
                        return f"Error getting page snapshot: {str(e)}"
                
                async def click_element(element: str, ref: str) -> str:
                    """Click on an element"""
                    try:
                        result = await self.workbench.call_tool("browser_click", {"element": element, "ref": ref})
                        return str(result)
                    except Exception as e:
                        return f"Error clicking element: {str(e)}"
                
                async def type_text(element: str, ref: str, text: str) -> str:
                    """Type text into an element"""
                    try:
                        result = await self.workbench.call_tool("browser_type", {"element": element, "ref": ref, "text": text})
                        return str(result)
                    except Exception as e:
                        return f"Error typing text: {str(e)}"
                
                # List of MCP tool functions
                mcp_tools = [navigate_to_url, take_screenshot, get_page_snapshot, click_element, type_text]
            else:
                mcp_tools = []
            
            # Create the AssistantAgent with MCP tools
            self.agent = AssistantAgent(
                name="TestPlanGenerator",
                model_client=self.model_client,
                tools=mcp_tools,  # Use MCP tools from workbench
                system_message=self.get_system_message(),
                description="Specialized agent for systematic website exploration and test plan generation using MCP Playwright tools",
                reflect_on_tool_use=False,  # Don't ask for confirmation before using tools
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
        return '''You are an AUTONOMOUS Test Exploration Agent that helps users test websites using MCP Playwright tools.

🎯 **YOUR ROLE:**
- Execute website testing tasks using browser automation
- Work directly with user requests and test scenarios  
- Be autonomous in execution once you understand the task

🚀 **EXECUTION BEHAVIOR:**
1. **When user provides a task/URL/scenario - START IMMEDIATELY** 
2. **When user says "CONTINUE" without specifics - Use a demo site for testing**
3. **Don't ask for more information repeatedly - take action instead**
4. **Execute steps systematically and report progress**
5. **Only ask for credentials when you encounter login forms**
6. **Continue until task is complete or you hit a real blocker**

🎯 **WORKFLOW APPROACH:**
1. If user provides specific URL/task: execute it immediately
2. If user says "CONTINUE" or gives vague input: navigate to https://httpbin.org for demonstration
3. Take snapshots to understand page structure before interacting
4. Use element references from snapshots for reliable interactions
5. Report results and completion
6. **IMPORTANT**: If no specific task is given, demonstrate your capabilities on httpbin.org

� **AVAILABLE MCP TOOLS:**
- mcp_playwright_browser_navigate: Navigate to URLs
- mcp_playwright_browser_snapshot: Get page accessibility snapshot with element refs
- mcp_playwright_browser_click: Click elements (use element description + ref from snapshot)
- mcp_playwright_browser_type: Type text into input fields
- mcp_playwright_browser_take_screenshot: Capture screenshots for verification

⚡ **EXECUTION STYLE:**
- Be conversational and helpful
- When given a clear task, execute it autonomously without asking for approval on each step
- Use clear element descriptions from snapshots for reliable interactions
- Handle errors gracefully and inform the user
- Provide clear progress updates and final results

**Remember:** You are here to help users test their websites efficiently. Be autonomous in execution but communicative about what you're doing!

🏁 **TASK COMPLETION:**
When you have successfully completed the requested testing task (e.g., navigated to pages, tested functionality, gathered results), end your response with "TASK_COMPLETE" to indicate you're done.'''


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
    return FileResponse("index.html")


async def get_team(
    user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]],
) -> AssistantAgent:
    try:
        # Initialize the WebsiteExploreAgent properly
        website_agent = WebsiteExploreAgent()
        await website_agent.initialize()
        
        # Check if initialization was successful
        if website_agent.agent is None:
            raise RuntimeError("Failed to initialize WebsiteExploreAgent - agent is None")
        
        agent = website_agent.agent

        # For autonomous execution, we'll directly use the main agent without group chat
        # This eliminates the "CONTINUE" loop issue
        
        # Log the agent details
        logger.debug(f"Using single agent for autonomous execution: {getattr(agent, 'name', 'NO_NAME')}")
        
        # Return the agent directly instead of creating a team
        # We'll handle the conversation flow in the chat function
        return agent
        # Load state from file.
        if not os.path.exists(state_path):
            return team
        async with aiofiles.open(state_path, "r") as file:
            state = json.loads(await file.read())
        await team.load_state(state)
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
                async for message in stream:
                    if isinstance(message, TaskResult):
                        continue
                    
                    # Skip streaming chunks to avoid duplication - only send final messages
                    if isinstance(message, ModelClientStreamingChunkEvent):
                        continue
                    
                    # Filter out tool result messages to reduce UI noise
                    # Only show meaningful conversational messages
                    message_type = getattr(message, 'type', '')
                    if 'ToolCall' in message_type or 'ToolResult' in message_type:
                        logger.debug(f"Filtering out tool message: {message_type}")
                        continue
                    
                    # Safely serialize message for WebSocket
                    try:
                        logger.debug(f"Processing message: type={type(message)}, message={message}")
                        message_data = safe_model_dump(message)
                        await websocket.send_json(message_data)
                        
                        if not isinstance(message, UserInputRequestedEvent):
                            # Don't save user input events to history.
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