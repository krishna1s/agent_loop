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
from TestingAgent import ProgressOnlyMagenticOneGroupChat

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import aiofiles
import yaml
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
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

# Global progress tracking
current_progress_msg = None
# Aggregated checklist state for progress items
progress_checklist: list[dict] = []

async def progress_callback(phase: str, message: str):
    """Aggregate and display orchestrator progress as a single updatable checklist in Chainlit.

    Additionally persist a compact progress message into the team's message thread so
    ProgressOnlyMagenticOneGroupChat.get_progress_summary() can count progress updates.
    """
    global current_progress_msg, progress_checklist, progress_manager

    try:
        # Sanitize incoming text
        content_str = ''.join(ch for ch in str(message) if ord(ch) >= 32).strip()
        if not content_str:
            return

        # Helper: extract a section block (e.g., "Just Completed", "Currently Doing", "Next Step")
        def extract_section(text: str, start_headers: list[str], end_headers: list[str]) -> str:
            text_up = text.upper()
            start_idx = -1
            for h in start_headers:
                idx = text_up.find(h.upper())
                if idx != -1:
                    start_idx = idx
                    break
            if start_idx == -1:
                return ""
            end_idx = len(text)
            for eh in end_headers:
                eidx = text_up.find(eh.upper(), start_idx + 1)
                if eidx != -1 and eidx < end_idx:
                    end_idx = eidx
            return text[start_idx:end_idx]

        headers = {
            'completed': ['JUST COMPLETED', 'COMPLETED'],
            'current': ['CURRENTLY DOING', 'CURRENTLY'],
            'next': ['NEXT STEP', 'NEXT']
        }
        end_headers = ['JUST COMPLETED', 'CURRENTLY DOING', 'NEXT STEP', 'PROGRESS UPDATE']

        # Parse sections and update the persistent checklist state
        for section, starts in headers.items():
            sec_text = extract_section(content_str, starts, end_headers)
            if not sec_text:
                continue

            # Remove header label (everything up to the first colon if present)
            colon_pos = sec_text.find(':')
            if colon_pos != -1:
                body = sec_text[colon_pos + 1 :]
            else:
                body = sec_text
                for s in starts:
                    idx = body.upper().find(s.upper())
                    if idx != -1:
                        body = body[idx + len(s) :]
                        break

            # Extract candidate items from bullets or lines; fallback to sentence split if needed
            parts: list[str] = []
            for ln in body.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                if ln.startswith('- ') or ln.startswith('• ') or ln.startswith('* '):
                    parts.append(ln[2:].strip())
                else:
                    parts.append(ln)

            if not parts:
                parts = [p.strip() for p in body.split('.') if p.strip()]

            for p in parts:
                item = ' '.join(p.split())
                if not item:
                    continue

                # Normalize for deduplication: lower-case, strip trailing punctuation and collapse whitespace
                def _normalize(s: str) -> str:
                    s = s.strip()
                    # remove trailing punctuation common in our outputs
                    while s and s[-1] in '.-–—:':
                        s = s[:-1].strip()
                    return ' '.join(s.lower().split())

                key = _normalize(item)
                existing = next((it for it in progress_checklist if it.get('key') == key), None)
                if section == 'completed':
                    if not existing:
                        progress_checklist.append({'text': item, 'status': 'done', 'key': key})
                    else:
                        existing['status'] = 'done'
                elif section == 'current':
                    if not existing:
                        progress_checklist.append({'text': item, 'status': 'doing', 'key': key})
                    else:
                        existing['status'] = 'doing'
                else:  # next
                    if not existing:
                        progress_checklist.append({'text': item, 'status': 'pending', 'key': key})

        # Build a compact markdown checklist to display
        md_lines: list[str] = []
        for it in progress_checklist:
            status = it.get('status', 'pending')
            prefix = '[X]' if status == 'done' else '[~]' if status == 'doing' else '[ ]'
            md_lines.append(f"- {prefix} {it['text']}")

        content_md = f"🔄 **{phase}**\n\n" + "\n".join(md_lines)

        if current_progress_msg:
            current_progress_msg.content = content_md
            await current_progress_msg.update()
        else:
            current_progress_msg = await cl.Message(content=content_md).send()

        # Persist the progress update into the team's message thread if available.
        # This ensures ProgressOnlyMagenticOneGroupChat.get_progress_summary() can read
        # the saved state and count progress updates.
        try:
            if progress_manager is not None:
                # Use the group's manager name as the message source if possible
                mgr_name = getattr(progress_manager, '_group_chat_manager_name', None) or 'Testing Agent'
                ledger_msg = TextMessage(content=content_md, source=mgr_name)
                # Write the message into the team's internal message thread
                await progress_manager.update_message_thread([ledger_msg])
                logger.debug('Persisted progress update to team message thread')
        except Exception as e:
            logger.exception(f"Failed to persist progress update into team state: {e}")

    except Exception as e:
        logger.error(f"Error in progress callback: {e}")

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
        team = ProgressOnlyMagenticOneGroupChat(
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
        # Reset progress message for new conversation
        current_progress_msg = None
        
        # Disable user input while the agent is processing
        try:
            await cl.context.emitter.task_start()
        except Exception as e:
            logger.debug(f"Could not send task_start: {e}")
        
        # Show initial processing message
        current_progress_msg = await cl.Message(content="🤖 AI Thinking...").send()
        # Trigger a streaming/typing indicator for the assistant so the UI shows "agent is thinking"
        try:
            current_progress_msg.streaming = True
            step_dict = current_progress_msg.to_dict()
            await cl.context.emitter.stream_start(step_dict)
        except Exception:
            # Non-fatal: continue without the typing indicator if the emitter doesn't support it
            pass
        
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

                # If the manager produced a final Markdown Enhanced Test Plan, surface it as final output
                manager_name = getattr(team, '_group_chat_manager_name', None)
                # Derive a friendly display name for the source
                if source == manager_name:
                    agent_name = 'Testing Agent'
                else:
                    agent_name = source.replace('_', ' ').title()
                
                if message_type == 'TextMessage' and source == manager_name and isinstance(content, str):
                    content_str = ''.join(ch for ch in content if ord(ch) >= 32).strip()
                    # Detect likely final plan by common headings
                    if content_str.startswith('#') or 'Enhanced Test Plan' in content_str or '### Test Steps' in content_str:
                        try:
                            await cl.context.emitter.task_end()
                        except Exception:
                            pass
                        await cl.Message(content=content_str, author='Testing Agent').send()
                        response_messages.append(message_data)
                        history.append(message_data)
                        # Do not also route this into the checklist; continue processing
                        continue

                # For non-final messages fall back to previous handling
                # (progress updates will be routed to the checklist below)
                
                # Convert content to string if needed
                content_str = content if isinstance(content, str) else str(content)
                # Remove control / non-printable characters that can appear in model output
                content_str = ''.join(ch for ch in content_str if ord(ch) >= 32)
                # Remove contiguous digits immediately preceding 'PROGRESS UPDATE' (e.g., '504 PROGRESS UPDATE')
                pu_idx = content_str.upper().find('PROGRESS UPDATE')
                if pu_idx != -1:
                    j = pu_idx - 1
                    while j >= 0 and content_str[j].isdigit():
                        j -= 1
                    if j < pu_idx - 1:
                        content_str = content_str[: j + 1] + content_str[pu_idx:]
                
                # Only show if it has meaningful content
                if len(content_str.strip()) > 10:  # Avoid very short messages
                    if source == manager_name:
                        # Route manager-origin progress updates into the single persistent checklist message
                        await progress_callback("PROGRESS UPDATE", content_str)
                    else:
                        await cl.Message(
                            content=f"**{agent_name}:**\n\n{content_str}",
                            author=agent_name
                        ).send()
                
                response_messages.append(message_data)
                history.append(message_data)
                    
            except Exception as e:
                logger.error(f"Error processing filtered message: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Update final progress
        if current_progress_msg:
            current_progress_msg.content = "✅ **Processing Complete!**\n\nTask execution finished."
            await current_progress_msg.update()

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

        # Show progress summary
        progress_summary = await team.get_progress_summary()
        await cl.Message(
            content=f"📊 **Progress Summary:**\n\n"
            f"- **Progress Updates:** {progress_summary['total_progress_updates']}\n"
            f"- **Task:** {progress_summary['task_description'][:100]}...\n"
            f"- **Current Phase:** {progress_summary['current_phase']}"
        ).send()
        
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
