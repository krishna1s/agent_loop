import json
import logging
from typing import Callable, List, Any
from pathlib import Path

# Import the upstream MagenticOne orchestrator and group chat classes
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import (
    MagenticOneOrchestrator,
)
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_group_chat import (
    MagenticOneGroupChat,
)
from autogen_core import DefaultTopicId, AgentId
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams._group_chat._events import GroupChatAgentResponse, GroupChatReset
from autogen_agentchat.base import Response

logger = logging.getLogger(__name__)


class CustomMagneticOrchestrator(MagenticOneOrchestrator):
    """Subclass of MagenticOneOrchestrator that instructs the model to return
    a single compact JSON progress ledger and to put the user-visible text in a
    strict "PROGRESS UPDATE" format.

    The internal orchestration loop and state handling are unchanged; we only
    override the prompt construction so the LLM's progress ledger is concise
    and UI-friendly.
    """

    def _get_progress_ledger_prompt(self, task: str, team: str, names: List[str]) -> str:
        # Example ledger to show the precise shape we expect.
        example = {
            "is_request_satisfied": {"answer": False, "reason": "Not all steps completed."},
            "is_progress_being_made": {"answer": True, "reason": "A short action completed."},
            "is_in_loop": {"answer": False, "reason": "Not detected."},
            "instruction_or_question": {
                "answer": (
                    "[Done] Logged in to the site.\n"
                    "[Currently Doing] : Navigating to the dashboard.\n"
                    "[Next Step] : Click the 'Export' button."
                ),
                "reason": "Concise progress summary for the UI.",
            },
            "next_speaker": {"answer": names[0] if names else "", "reason": "Choose the agent who should act next."},
        }

        prompt = f"""
You are the task orchestrator. You will be asked to produce a single JSON object (and ONLY the JSON) describing the current progress for the TASK below.

TASK:
{task}

TEAM:
{team}

PARTICIPANTS: {', '.join(names)}

INSTRUCTIONS:
1) Return EXACTLY ONE JSON object. Do NOT add explanation or free-text outside the JSON.
2) The JSON must contain these keys: is_request_satisfied, is_progress_being_made, is_in_loop, instruction_or_question, next_speaker.
   - Each key's value must itself be an object with two keys: "answer" and "reason".
3) The value for instruction_or_question.answer MUST be the ONLY text that is meant for the end user. It MUST follow this format EXACTLY:

[Done] <one-sentence summary>
[Currently Doing] : <one-sentence summary>
[Next Step]: <one-sentence actionable instruction>

IMPORTANT: "Next Step" MUST be a single, concrete, *actionable* instruction (e.g. "Click the 'Sign in' button", "Enter email into '#email' and press Enter", or "Ask user for MFA code"). Avoid vague language like "prepare" or "working on".

4) Keep "reason" fields short (one short sentence) and do not include chain-of-thought.
5) When selecting next_speaker, return one of the participant names.

Example output (replace with current state):
{json.dumps(example, indent=2)}
"""
        return prompt

    async def _reenter_outer_loop(self, cancellation_token):
        """Re-enter Outer loop but DO NOT publish the verbose task ledger to the UI output topic.

        We still update the internal message thread and broadcast to the group topic so participants
        receive the ledger, but we intentionally avoid pushing the ledger into the output topic / queue
        that feeds the external UI.
        """
        # Reset the agents
        for participant_topic_type in self._participant_name_to_topic_type.values():
            await self._runtime.send_message(
                GroupChatReset(),
                recipient=AgentId(type=participant_topic_type, key=self.id.key),
                cancellation_token=cancellation_token,
            )

        # Reset partially the group chat manager
        self._message_thread.clear()

        # Prepare the ledger (internal only)
        ledger_message = TextMessage(
            content=self._get_task_ledger_full_prompt(self._task, self._team_description, self._facts, self._plan),
            source=self._name,
        )

        # Save my copy
        await self.update_message_thread([ledger_message])

        # Broadcast to group (so participants know the ledger) BUT do NOT publish to the output topic
        await self.publish_message(
            GroupChatAgentResponse(response=Response(chat_message=ledger_message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        # Restart the inner loop
        await self._orchestrate_step(cancellation_token=cancellation_token)

    def _get_task_ledger_facts_prompt(self, task: str) -> str:
        base = super()._get_task_ledger_facts_prompt(task)
        return base + "\n\nNOTE: Keep the 'facts' answer short (1-2 sentences)."

    def _get_task_ledger_plan_prompt(self, team: str) -> str:
        base = super()._get_task_ledger_plan_prompt(team)
        return base + "\n\nNOTE: Return a concise plan with short, numbered actionable steps including right page selectors where relevant."

    def _get_task_ledger_full_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        base = super()._get_task_ledger_full_prompt(task, team, facts, plan)
        return base + "\n\nNOTE: Future progress checks will return a compact JSON progress ledger as specified."

    def _get_task_ledger_facts_update_prompt(self, task: str, facts: str) -> str:
        base = super()._get_task_ledger_facts_update_prompt(task, facts)
        return base + "\n\nNOTE: Keep updates short."

    def _get_task_ledger_plan_update_prompt(self, team: str) -> str:
        base = super()._get_task_ledger_plan_update_prompt(team)
        return base + "\n\nNOTE: Keep plan updates concise."

    def _get_final_answer_prompt(self, task: str) -> str:
        base = super()._get_final_answer_prompt(task)
        return (
            "You will now produce the FINAL ENHANCED TEST PLAN in Markdown. Output MUST be valid Markdown and include the following sections.\n"
            "Use proper Markdown syntax, with a blank line after each heading and between sections. Do NOT use triple backticks or code blocks.\n"
            "Example:\n"
            "## Enhanced Test Plan\n\n"
            f"**Task:** {task}\n\n"
            "### Preconditions\n\n"
            "- List any environment, credentials, or setup steps required.\n\n"
            "### Test Steps\n\n"
            "Provide a numbered list of step-by-step actions the agent performed or should perform. For each step include the selector based on page snapshot, which can be used while generating playwright script or tool used when relevant.\n\n"
            "### Expected Results\n\n"
            "For each step above, provide the expected result.\n\n"
            "### Recommendations\n\n"
            "Optional: Suggested next steps or mitigations.\n\n"
            "ONLY output the Markdown document. Do NOT include chain-of-thought or extra commentary.\n\n"
            + base
        )


class CustomMagneticOneGroupChat(MagenticOneGroupChat):
    """Group chat class that wires the ProgressOnly orchestrator.

    Use this class wherever you previously used MagenticOneGroupChat to get the same
    orchestration semantics but with progress-focused prompts.
    """

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: Any,
        termination_condition: Any,
        max_turns: int | None,
        message_factory: Any,
    ) -> Callable[[], CustomMagneticOrchestrator]:
        """Return a factory that constructs our ProgressOnly orchestrator with the
        same constructor shape as the original group chat.
        """

        return lambda: CustomMagneticOrchestrator(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            max_turns,
            message_factory,
            self._model_client,
            self._max_stalls,
            self._final_answer_prompt,
            output_message_queue,
            termination_condition,
            self._emit_team_events,
        )

    async def get_progress_summary(self) -> dict:
        """Return a compact progress summary extracted from the team's saved state.

        Keys returned:
        - total_progress_updates: int
        - task_description: str
        - current_phase: str
        - last_progress: Optional[str]

        This method is async because it relies on the manager's async save_state()
        to obtain the latest message thread.
        """
        try:
            state = await self.save_state()
        except Exception as e:
            logger.error(f"Failed to obtain team state for progress summary: {e}")
            return {
                "total_progress_updates": 0,
                "task_description": getattr(self, "description", "") or "",
                "current_phase": "",
                "last_progress": None,
            }

        # Try to find the manager's message thread inside the saved state. Different
        # autogen versions serialize the manager state in different shapes.
        message_thread = []
        try:
            # Newer save_state() returns agent_states keyed by agent name
            if isinstance(state, dict):
                agent_states = state.get("agent_states") or state.get("agent_states", {})
                if isinstance(agent_states, dict) and self._group_chat_manager_name in agent_states:
                    mgr_state = agent_states[self._group_chat_manager_name]
                    if isinstance(mgr_state, dict):
                        message_thread = mgr_state.get("message_thread", []) or []
                # Fallback to top-level message_thread
                if not message_thread:
                    message_thread = state.get("message_thread", []) or []
        except Exception:
            message_thread = []

        # Merge in any external progress overrides written by the Chainlit callback.
        try:
            overrides_path = Path.cwd() / "progress_overrides.json"
            if overrides_path.exists():
                raw = overrides_path.read_text()
                if raw:
                    overrides = json.loads(raw)
                    for ent in overrides:
                        if ent.get("team_id") == getattr(self, "_team_id", None):
                            # Append a minimal dict resembling a dumped chat message so the
                            # existing counting logic works unchanged.
                            message_thread.append({"content": ent.get("content", ""), "source": ent.get("manager_name")})
        except Exception:
            # Non-fatal; ignore overrides on parse errors
            pass

        total_progress_updates = 0
        last_progress = None
        current_phase = ""

        for msg in message_thread:
            try:
                # msg is the dumped representation of a BaseChatMessage
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                else:
                    content = str(msg)

                if not content:
                    continue

                if "PROGRESS UPDATE" in content or "🔄 PROGRESS UPDATE" in content:
                    total_progress_updates += 1
                    last_progress = content

                    # Try to extract the "Currently Doing" line
                    for line in content.splitlines():
                        if "Currently Doing:" in line:
                            current_phase = line.split("Currently Doing:", 1)[1].strip()
                            break
            except Exception:
                # Be tolerant of unexpected message dump shapes
                continue

        task_description = state.get("task") or ""

        return {
            "total_progress_updates": total_progress_updates,
            "task_description": task_description,
            "current_phase": current_phase,
            "last_progress": last_progress,
        }
