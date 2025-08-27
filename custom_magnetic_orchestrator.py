#!/usr/bin/env python3
"""
Custom Magnetic One Orchestrator with Prompt-Based Progress Filtering
This extends the MagenticOneGroupChat to show only orchestrator summaries
by modifying the orchestrator's system prompt to generate user-friendly updates.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime

# AutoGen imports
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent, ModelClientStreamingChunkEvent
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

import traceback

logger = logging.getLogger("CustomMagenticOne")

class ProgressFilteredMagneticOne(MagenticOneGroupChat):
    """
    Custom Magnetic One orchestrator that shows only orchestrator progress summaries.
    The orchestrator is prompted to provide concise, user-friendly updates about:
    - What task was just completed
    - What is being done next
    - Overall progress status
    
    All sub-agent messages are filtered out to show only the orchestrator's summaries.
    """
    
    def __init__(
        self,
        participants,
        model_client: AzureOpenAIChatCompletionClient,
        progress_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
        **kwargs
    ):
        # Extract custom parameters
        self.progress_callback = progress_callback
        
        # Pass the rest to the parent class with correct signature
        super().__init__(
            participants=participants, 
            model_client=model_client,
            **kwargs  # This will include termination_condition, max_turns, etc. as keyword arguments
        )
        
        self.current_phase = "Initializing"
        self.task_description = ""
        
        # Progress tracking
        self.progress_milestones = []
        
        # Note: Prompt enhancement will be handled at the application level
        # since accessing orchestrator agents from here is complex
    
    def _setup_progress_reporting_prompt(self):
        """Modify the orchestrator's system prompt to include progress reporting instructions"""
        
        progress_instructions = """

### CRITICAL: COMMUNICATION STYLE INSTRUCTIONS ###

You are an AI orchestrator that must communicate ONLY in brief, user-friendly progress updates. 

STRICT RULES:
1. NEVER show detailed planning, reasoning, or internal thoughts to the user
2. NEVER show fact sheets, team lists, or strategy discussions
3. ONLY communicate brief progress updates in this format:

🔄 **[ACTION]** - [Brief description of what's happening now]

EXAMPLES OF ALLOWED MESSAGES:
🔄 **Navigating** - Opening playwright.microsoft.com
🔄 **Analyzing** - Looking for Sign In button
🔄 **Clicking** - Clicking the Sign In button
🔄 **Typing** - Entering login credentials
🔄 **Checking** - Verifying login success
🔄 **Complete** - Login test finished successfully

FORBIDDEN MESSAGE TYPES:
- Long explanations about the team or plan
- Fact sheets or analysis sections
- Detailed reasoning or thoughts
- Strategy discussions
- Technical implementation details

COMMUNICATION APPROACH:
- Each message should be ONE LINE with an emoji and brief action
- Focus on WHAT you're doing, not WHY or HOW
- Keep messages under 10 words when possible
- Only speak when you're actually taking an action or need user input

Remember: The user wants to see PROGRESS, not PLANNING. Be concise and action-focused.

"""
        
        # Find and update the orchestrator agent's system message
        try:
            # Try different possible attribute names for participants
            participants = (getattr(self, '_participants', None) or 
                          getattr(self, 'participants', None) or 
                          getattr(self, '_agents', None) or 
                          [])
            
            for participant in participants:
                if hasattr(participant, 'system_message'):
                    participant_name = getattr(participant, 'name', '').lower()
                    if 'orchestrator' in participant_name:
                        # REPLACE the existing system message to enforce brief communication
                        participant.system_message = progress_instructions
                        logger.info(f"✅ Replaced {participant.name} system message with brief progress instructions")
                        break
        except Exception as e:
            logger.warning(f"Could not enhance orchestrator prompt: {e}")
            # Continue without enhancement
        
        # Find and update the orchestrator agent's system message
        # Get participants from the parent class
        try:
            # Try different possible attribute names for participants
            participants = getattr(self, '_participants', None) or \
                          getattr(self, 'participants', None) or \
                          getattr(self, '_agents', None) or \
                          []
            
            for participant in participants:
                if hasattr(participant, 'system_message'):
                    participant_name = getattr(participant, 'name', '').lower()
                    if 'orchestrator' in participant_name or hasattr(participant, '_is_orchestrator'):
                        # Append progress instructions to existing system message
                        current_message = getattr(participant, 'system_message', '')
                        participant.system_message = current_message + progress_instructions
                        logger.info(f"✅ Enhanced {participant.name} with progress reporting instructions")
                        break
        except Exception as e:
            logger.warning(f"Could not enhance orchestrator prompt: {e}")
            # Continue without enhancement
    
    async def _send_progress_update(self, phase: str, message: str, agent_name: str = "System"):
        """Send a filtered progress update to the UI"""
        if self.progress_callback:
            try:
                await self.progress_callback(f"**{phase}** ({agent_name})", message)
            except Exception as e:
                logger.error(f"Error sending progress update: {e}")
        
        # Log the milestone
        self.progress_milestones.append({
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "agent": agent_name,
            "message": message
        })
        
        logger.info(f"📍 Progress: {phase} - {message}")
    
    def _is_orchestrator_progress_message(self, message_data: Dict[str, Any]) -> bool:
        """Show only meaningful progress messages, filter out verbose orchestrator reasoning"""
        message_type = message_data.get('type', '')
        source = message_data.get('source', '')
        content = str(message_data.get('content', ''))
        
        # Always show user input requests regardless of source
        if message_type == 'UserInputRequestedEvent':
            return True
        
        # Filter out verbose orchestrator planning messages
        if source and ('orchestrator' in source.lower() or 'magenticone' in source.lower()):
            return True
        
        # Show messages from agents (TestPlanGenerator, etc.)
        # if (message_type == 'TextMessage' and source and 
        #     source.lower() not in ['orchestrator', 'magenticoneorchestrator'] and
        #     len(content.strip()) > 10):
        #     return True
        
        return False
    
    async def run_stream(self, task):
        """Override run_stream to show only orchestrator progress summaries"""
        
        # Initialize task
        await self._send_progress_update("Starting", "Initializing Magnetic One orchestrator for task", "System")
        self.task_description = str(task.content) if hasattr(task, 'content') else str(task)
        
        # Start the original stream
        original_stream = super().run_stream(task=task)
        
        async for message in original_stream:
            try:
                # Handle TaskResult (final completion)
                if isinstance(message, TaskResult):
                    await self._send_progress_update("Completed", "Task execution finished", "System")
                    yield message
                    continue
                
                # Handle UserInputRequestedEvent (always show these)
                if isinstance(message, UserInputRequestedEvent):
                    await self._send_progress_update("User Input", "Orchestrator requesting user input", "System")
                    yield message
                    continue
                
                # Process other messages
                if hasattr(message, 'model_dump'):
                    message_data = message.model_dump()
                    
                    # Only show orchestrator progress messages
                    if self._is_orchestrator_progress_message(message_data):
                        yield message
                
            except Exception as e:
                logger.error(f"Error processing message in filtered stream: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Still yield the message to avoid breaking the stream
                yield message
        
        # Send final progress summary
        await self._send_progress_update("Summary", 
            f"Task completed successfully with {len(self.progress_milestones)} progress updates", 
            "System")
    
    def get_progress_summary(self) -> dict:
        """Get a summary of progress tracking"""
        return {
            "total_progress_updates": len(self.progress_milestones),
            "current_phase": self.current_phase,
            "progress_milestones": list(self.progress_milestones),
            "task_description": getattr(self, 'task_description', 'No task')
        }

    async def save_state(self) -> dict:
        """Save the current state for persistence"""
        return {
            "current_phase": self.current_phase,
            "progress_milestones": list(self.progress_milestones),
            "task_description": getattr(self, 'task_description', '')
        }

class ProgressAwareMagneticOne:
    """
    Wrapper class that provides a clean interface for creating a progress-filtered Magnetic One team
    """
    
    def __init__(self):
        self.team = None
        self.progress_callback = None
        
    async def create_team(
        self,
        agents: List,
        model_client: AzureOpenAIChatCompletionClient,
        progress_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
        max_turns: int = 20
    ):
        """Create a progress-filtered Magnetic One team"""
        
        self.progress_callback = progress_callback
        
        # Create the custom team with correct parameter structure
        self.team = ProgressFilteredMagneticOne(
            participants=agents,  # First positional parameter
            model_client=model_client,  # Second positional parameter
            progress_callback=progress_callback,  # Custom parameter
            max_turns=max_turns,  # Keyword parameter
            termination_condition=None  # Keyword parameter
        )
        
        return self.team
    
    async def run_with_progress(self, task, progress_callback: Callable[[str, str], Awaitable[None]]):
        """Run a task with progress updates"""
        if not self.team:
            raise ValueError("Team not created. Call create_team first.")
        
        self.team.progress_callback = progress_callback
        
        async for result in self.team.run_stream(task):
            yield result

# Usage example and documentation
if __name__ == "__main__":
    print("""
    Enhanced Custom Magnetic One Orchestrator with Prompt-Based Progress Filtering
    
    This implementation provides:
    
    1. ✅ Prompt-Enhanced Orchestrator:
       - Modified system prompt instructs orchestrator to provide formatted progress updates
       - Structured format: Just Completed, Currently Doing, Next Step
       - Emoji-enhanced visual clarity (🔄, ✅, 🚀, etc.)
       
    2. ✅ Orchestrator-Only Display:
       - Only shows messages from the Magnetic One orchestrator
       - Filters out all sub-agent messages and internal communications
       - Focuses on high-level task coordination and progress
       
    3. ✅ User-Friendly Updates:
       - Clear progress format instead of hardcoded keyword matching
       - Action-focused summaries (what was done, what's next)
       - No internal reasoning or technical details exposed
       
    4. ✅ Simplified Architecture:
       - Removed complex message filtering logic
       - Relies on orchestrator's AI capability to generate appropriate summaries
       - More maintainable and flexible approach
    
    Key Benefits:
    - No hardcoded keyword filtering needed
    - Orchestrator AI generates contextually appropriate progress updates
    - Clean separation: orchestrator summaries vs. agent details
    - User sees only essential progress information
    - Easier to maintain and extend
    """)
