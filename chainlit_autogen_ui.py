#!/usr/bin/env python3
"""
Chainlit Chat UI with AutoGen Test Generation Agent
Provides a web-based chat interface for the AutoGen Test Generation Agent
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import chainlit as cl
from chainlit import Message, AskUserMessage, AskFileMessage

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.auth.azure import AzureTokenProvider
from azure.identity import DefaultAzureCredential

# Import our tool functions
try:
    from test_execution_tools import (
        setup_npm_project_direct, run_playwright_test_direct, read_test_results_direct,
        read_test_script_direct, save_test_script_direct, list_test_files_direct
    )
except ImportError:
    print("⚠️ Test execution tools not available - using mock functions")
    
    def setup_npm_project_direct(*args, **kwargs):
        return {"success": True, "message": "Mock npm project setup"}
    
    def run_playwright_test_direct(*args, **kwargs):
        return {"success": True, "message": "Mock test execution"}
    
    def read_test_results_direct(*args, **kwargs):
        return {"success": True, "message": "Mock test results"}
    
    def read_test_script_direct(*args, **kwargs):
        return {"success": True, "message": "Mock script reading"}
    
    def save_test_script_direct(*args, **kwargs):
        return {"success": True, "message": "Mock script saving"}
    
    def list_test_files_direct(*args, **kwargs):
        return {"success": True, "message": "Mock file listing"}

class ChainlitAutoGenAgent:
    """AutoGen Test Generation Agent integrated with Chainlit"""
    
    def __init__(self):
        self.agent = None
        self.model_client = None
        self.team = None
        self.session_data = {}
    
    async def initialize(self):
        """Initialize the AutoGen agent and team"""
        
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
            
            # Create tool functions
            tools = [
                self.read_test_script,
                self.save_test_script,
                self.setup_npm_project,
                self.list_test_files,
                self.run_playwright_test,
                self.read_test_results
            ]
            
            # Create system message
            system_message = self.create_system_message()
            
            # Create the AssistantAgent
            self.agent = AssistantAgent(
                name="test_generation_agent",
                model_client=self.model_client,
                tools=tools,
                system_message=system_message,
                description="Specialized agent for converting recorded browser scripts to comprehensive Python Playwright tests",
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
            print(f"❌ Failed to initialize AutoGen agent: {e}")
            return False
    
    def create_system_message(self) -> str:
        """Create comprehensive system message for the test generation agent"""
        
        return """
You are a specialized Test Generation Agent focused on converting recorded browser scripts to comprehensive Python Playwright tests.

**YOUR MISSION:**
Convert TypeScript/JavaScript recording files into high-quality Python Playwright test suites with multiple test scenarios.

**CORE RESPONSIBILITIES:**
1. **Read and Analyze Recording Scripts** - Use read_test_script to analyze input files
2. **Generate Multiple Test Scenarios** - Create comprehensive test cases covering different aspects
3. **Apply Testing Best Practices** - Follow industry standards and guidelines
4. **Create Well-Structured Files** - Generate clean, maintainable Python test code
5. **Use Proper File Operations** - ALWAYS use provided functions for file operations

**WORKFLOW:**
1. Read and analyze the recording file using read_test_script
2. Set up the npm project structure using setup_npm_project
3. Generate multiple focused test files using save_test_script:
   - main_user_journey.py (core flow from recording)
   - navigation_tests.py (navigation and menu testing)
   - form_interaction_tests.py (form filling and validation)
   - ui_validation_tests.py (UI elements and responsive testing)
4. List all generated files using list_test_files for verification

**PYTHON PLAYWRIGHT TEMPLATE:**
Use headed mode (headless=False), comprehensive error handling, semantic selectors, and proper assertions.

Focus solely on generating high-quality Python Playwright test code. Provide clear status updates and summaries.

When you complete a task, provide a clear summary of what was accomplished.
"""
    
    # Tool Functions
    async def read_test_script(self, test_file: str, test_directory: str = "generated_tests") -> str:
        """Read a test script file and return its content for analysis"""
        try:
            result = read_test_script_direct(test_file, test_directory)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "message": f"Error reading test script: {str(e)}"})
    
    async def save_test_script(self, test_file: str, content: str, test_directory: str = "generated_tests") -> str:
        """Save or update a test script file with Python Playwright code"""
        try:
            result = save_test_script_direct(test_file, content, test_directory)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "message": f"Error saving test script: {str(e)}"})
    
    async def setup_npm_project(self, test_directory: str = "generated_tests") -> str:
        """Set up npm project with Playwright dependencies"""
        try:
            result = setup_npm_project_direct(test_directory)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "message": f"Error setting up npm project: {str(e)}"})
    
    async def list_test_files(self, test_directory: str = "generated_tests") -> str:
        """List all test files in the test directory"""
        try:
            result = list_test_files_direct(test_directory)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "message": f"Error listing test files: {str(e)}"})
    
    async def run_playwright_test(self, test_file: str = None, test_directory: str = "generated_tests", 
                                 browser: str = "chromium", headed: bool = True) -> str:
        """Execute Playwright tests"""
        try:
            result = run_playwright_test_direct(test_file, test_directory, browser, headed)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "message": f"Error running Playwright test: {str(e)}"})
    
    async def read_test_results(self, results_file: str = None, test_directory: str = "generated_tests") -> str:
        """Read and parse Playwright test results"""
        try:
            result = read_test_results_direct(results_file, test_directory)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "message": f"Error reading test results: {str(e)}"})
    
    async def process_task(self, task: str) -> str:
        """Process a task using the AutoGen agent"""
        
        if not self.team:
            return "❌ Agent not initialized. Please restart the session."
        
        try:
            # Create conversation history with user message
            conversation_history = [
                TextMessage(
                    source="user",
                    content=task
                )
            ]
            
            # Process with agent - get complete response
            result = await self.agent.on_messages(conversation_history, CancellationToken())
            
            # Extract response content
            if result.messages:
                responses = []
                for msg in result.messages:
                    if hasattr(msg, 'content') and msg.content:
                        responses.append(str(msg.content))
                
                if responses:
                    return "\n\n".join(responses)
                else:
                    return "✅ Task completed successfully."
            else:
                return "✅ Task completed successfully."
            
        except Exception as e:
            return f"❌ Error processing task: {str(e)}"
    
    async def close(self):
        """Clean up resources"""
        if self.model_client:
            await self.model_client.close()

# Global agent instance
autogen_agent = ChainlitAutoGenAgent()

@cl.on_chat_start
async def start_chat():
    """Initialize the chat session"""
    
    await cl.Message(
        content="🎯 **AutoGen Test Generation Agent**\n\nInitializing the agent...",
        author="System"
    ).send()
    
    # Initialize the agent
    success = await autogen_agent.initialize()
    
    if success:
        welcome_message = """
🎉 **AutoGen Test Generation Agent Ready!**

I can help you convert recorded browser scripts to comprehensive Python Playwright tests.

**What I can do:**
- 📝 Convert TypeScript/JavaScript recordings to Python Playwright tests
- 🏗️ Set up npm project structures
- 📁 Generate multiple test scenarios (navigation, forms, UI validation, etc.)
- 🔍 Analyze existing test scripts
- ▶️ Execute tests and read results

**To get started:**
1. Upload a recording file or provide a file path
2. Specify your target directory (optional)
3. I'll generate comprehensive test suites for you!

**Example commands:**
- "Convert recording-2.ts to Python tests"
- "Set up a new test project in my-tests directory"
- "List all test files in generated_tests"
- "Run tests in browser headed mode"

How can I help you today?
"""
        
        await cl.Message(
            content=welcome_message,
            author="AutoGen Agent"
        ).send()
        
        # Store agent in session
        cl.user_session.set("autogen_agent", autogen_agent)
        
    else:
        error_message = """
❌ **Failed to Initialize Agent**

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
async def handle_message(message: cl.Message):
    """Handle incoming chat messages"""
    
    # Get the agent from session
    agent = cl.user_session.get("autogen_agent")
    
    if not agent or not agent.team:
        await cl.Message(
            content="❌ Agent not available. Please restart the session.",
            author="System"
        ).send()
        return
    
    # Show typing indicator
    async with cl.Step(name="Processing", type="run") as step:
        step.input = message.content
        
        # Check if user is asking for file upload
        if any(keyword in message.content.lower() for keyword in ["upload", "file", "recording"]):
            files = await cl.AskFileMessage(
                content="Please upload your recording file (TypeScript/JavaScript):",
                accept={"text/plain": [".ts", ".js", ".py", ".txt"]},
                max_size_mb=10
            ).send()
            
            if files:
                file = files[0]
                # Save uploaded file
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                
                file_path = upload_dir / file.name
                
                # Read the file content from Chainlit file object
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        # Chainlit file objects have content as bytes, need to decode
                        if hasattr(file, 'content') and file.content:
                            content = file.content
                            if isinstance(content, bytes):
                                content = content.decode('utf-8')
                            f.write(content)
                        elif hasattr(file, 'path') and file.path:
                            # Read from the temporary file path
                            with open(file.path, 'r', encoding='utf-8') as temp_file:
                                f.write(temp_file.read())
                        else:
                            raise ValueError("Unable to access file content")
                except Exception as e:
                    await cl.Message(
                        content=f"❌ Error saving uploaded file: {str(e)}",
                        author="System"
                    ).send()
                    return
                
                task = f"""
**TASK: Convert Recording to Python Playwright Tests**

Please analyze and convert the uploaded recording file to comprehensive Python Playwright tests.

**INPUT FILE:**
Recording file path: {file_path}
Target directory: generated_tests/chainlit_session/test_cases

**PROCESS:**
1. Read and analyze the recording file using read_test_script
2. Set up the npm project structure using setup_npm_project
3. Generate multiple focused test files using save_test_script
4. List all generated files using list_test_files for verification

Begin the conversion process now.
"""
                
            else:
                await cl.Message(
                    content="❌ No file uploaded. Please try again.",
                    author="System"
                ).send()
                return
        else:
            # Process regular chat message
            task = message.content
        
        # Process the task with the agent
        await cl.Message(
            content="🤖 Processing your request with AutoGen agent...",
            author="System"
        ).send()
        
        response = await agent.process_task(task)
        step.output = response
    
    # Send the agent's response
    await cl.Message(
        content=response,
        author="AutoGen Agent"
    ).send()

@cl.on_chat_end
async def end_chat():
    """Clean up when chat ends"""
    
    agent = cl.user_session.get("autogen_agent")
    if agent:
        await agent.close()
    
    await cl.Message(
        content="👋 Thanks for using AutoGen Test Generation Agent!",
        author="System"
    ).send()

if __name__ == "__main__":
    # Check if running directly
    print("🎯 AutoGen + Chainlit Chat UI")
    print("=" * 40)
    print("🚀 Starting Chainlit server...")
    print("📝 Make sure to set your Azure OpenAI environment variables")
    print("🌐 The chat UI will open in your browser")
    print("=" * 40)
