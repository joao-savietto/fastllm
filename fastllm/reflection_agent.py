"""
Reflection Agent Module

This module implements a reflection-based agentic workflow system
that enables iterative
problem-solving and content generation through structured phases of action,
reflection, and
refinement.

The ReflectionAgent follows a 5-step process:
1. Initial Action Generation - Generate initial plan and solution
2. Reflection Phase - Critically evaluate the output for improvements
3. Refinement Process - Create improved version based on reflections
4. Iterative Improvement - Cycle through reflection/refinement
until quality standards met
5. Reply to User - Provide final response in user's original language

The agent uses a workflow-based architecture with Node and BooleanNode
components
to structure the execution flow and make decisions based on quality metrics.
"""

from fastllm.agent import Agent
from fastllm.workflow import Node, BooleanNode
from typing import Generator, Dict, Any


REFLECTION_SYSTEM = """# Reflection Agentic Workflow System Prompt

You are an AI assistant operating within a Reflection Agentic
workflow designed for iterative problem-solving and content generation.

## Workflow Structure:

### Step 1. Initial Action Generation
* Step 1.1:
- Think step by step, break the task down
- Receive a prompt or task from the user
* Step 1.2:
- Generate a solution or action based on your understanding of the request
- Utilize available tools to accomplish the task if needed
- Provide a first attempt at solving the problem or completing the task

### Step 2. Reflection Phase
- Critically evaluate your own output using self-reflection techniques
- Assess accuracy, completeness, efficiency, and quality of your
  previous response
- Consider potential improvements, missing elements, or areas of weakness
- Evaluate how effectively you used available tools in your attempt
- If needed, consult a secondary "critic" agent to provide
  additional evaluation
- If you believe you completed the task and nothing else needs to be done,
  say "Task Completed"   to skip the "Step 3" and "Step 4" entirely.
- If you still need to refine the answer, say "Needs refinements" to
  go to Step 3 and 4.
- If you say the words "Task Completed" in this step, the task will be
  interrupted no matter the context, because it will trigger the finish
  status.

### Step 3. Refinement Process
- Based on reflection insights, generate an improved version of your response
- Address identified issues while maintaining core functionality and intent
- Enhance quality through better structure, accuracy, or completeness
- Optimize tool usage in the refined approach if applicable

### Step 4. Iterative Improvement
- Continue cycling between Reflection and Refinement phases until:
  - The output meets acceptable quality standards
  - No further meaningful improvements can be made
  - User satisfaction is achieved
- Each iteration should demonstrate measurable improvement over
  previous versions
- Consider tool utilization effectiveness in each iteration

## Step 5. Reply the user
- Now that the task is completed, you'll write a reply to the user task given
  to you in Step 1
- This is the only part of the workflow the user will be able to read
- You can either repeat the final and refined answer, or write a summary
  of what you done
- Try to explain how and why you came to the conclusion

## Key Instructions:
- Always think step by step
- Always begin with Initial Action Generation regardless of complexity
- Provide explicit reasoning for each reflection step and refinement decision
- Maintain awareness that you're in a multi-step process,
  not just responding to a single query
- Use iterative improvements as your primary method of achieving quality output
- When appropriate, indicate when you've reached an acceptable final result
- Consider tool availability and usage effectiveness throughout the process
- **VERY IMPORTANT**: On Step 5, Always reply in the same language
  as the task given by the user.
- On step 2, you should only say the words "Task Completed" if the task is
  actually completed.

Follow this structured approach while maintaining flexibility to adapt
based on the specific task requirements and user feedback."""


def is_complete(node: BooleanNode, session_id: str, last_message: dict) -> bool:
    """
    Determine if refinement iteration should continue based on quality metrics.

    This function checks whether the agent has completed its task by looking
    for the phrase "task completed" in the last message content.
    If found, it indicates that no further refinements are needed.

    Args:
        node (BooleanNode): The boolean node being evaluated
        session_id (str): Identifier for the current conversation session
        last_message (dict): Dictionary containing the last message
        from the agent

    Returns:
        bool: True if task is complete (contains "task completed"),
        False otherwise
    """
    try:
        content = str(last_message.get("content", ""))

        return "task completed" in content.lower()

    except Exception:
        return False


class ReflectionAgent:
    """
    A reflection-based agentic workflow system for iterative problem-solving.

    This agent implements a structured 5-step process that includes
    initial action
    generation, reflection on outputs, refinement of solutions,
    and iterative improvement
    until acceptable quality standards are met. The agent maintains
    conversation state
    through session IDs and can utilize various tools during execution.

    Attributes:
        agent (Agent): Underlying agent instance that handles
        the core functionality

    Example:
        >>> from fastllm.reflection_agent import ReflectionAgent
        >>> agent = ReflectionAgent()
        >>> response = agent.generate("Solve a math problem")
        >>> for chunk in response:
        ...     print(chunk)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ReflectionAgent.

        Sets up the underlying agent with reflection system
        prompt and initializes
        the workflow components needed for the iterative process.

        Args:
            *args: Argument list passed to the Agent constructor
            **kwargs: Arbitrary keyword args passed to the Agent constructor
        """
        self.agent = Agent(*args, **kwargs)
        self.agent.system_prompt = REFLECTION_SYSTEM

    def generate(
        self,
        message: str = "",
        image: bytes = None,
        session_id: str = "default",
        stream: bool = True,
        after_generation: callable = None,
        before_generation: callable = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a response through the reflection-based workflow process.

        Executes the full 5-step reflection workflow to
        produce high-quality output:
        1. Initial Action Generation
        2. Reflection Phase
        3. Refinement Process
        4. Iterative Improvement
        5. Reply to User

        The method uses a workflow of connected nodes that process
        through different steps,
        with decision points based on the is_complete function to
        determine when refinement
        should stop.

        Args:
            message (str, optional): The task or prompt for the agent
            to work on. Defaults to "".
            image (bytes, optional): Image data to be processed by the agent.
            session_id (str, optional): Identifier for conversation state.
                Defaults to "default".
            stream (bool, optional): Whether to return results as a
                generator stream. Defaults to True.
            after_generation (callable, optional): Callback function
                to execute after generation completes. Defaults to None.
            before_generation (callable, optional): Callback function to
                execute before generation begins. Defaults to None.

        Yields:
            Dict[str, Any]: Generator that yields response chunks
                from the agent workflow

        Returns:
            Generator[Dict[str, Any], None, None]: Stream of responses
                from the reflection process
        """
        common_params = {
            "agent": self.agent,
            "before_generation": before_generation,
            "after_generation": after_generation,
        }

        prompt_plan = (
            f"We are now at Step 1,1: Initial Action Generation\nFirst,"
            f" step by step, generate a plan to solve the task given by the "
            " user. Don't try to solve the task before Step 1.2 begins.:\n\n "
            f'Task:\n{message}"'
        )
        step1_plan = Node(instruction=prompt_plan, **common_params)

        prompt_action = (
            "We are at Step 1.2: Generate your solution. Use the available "
            "tools, if any, to accomplish the task."
        )
        step1_act = Node(instruction=prompt_action, **common_params)

        step2_prompt = (
            "We are now at Step 2: Reflection Phase. Review the results"
            " you got so far and evaluate possible enhancements and fixes. "
            "Write `Task Completed` if you determine that there are no more "
            "improvements to be done. Or else, Say 'Needs refinements' to go "
            "to step 3. Don't try to refine before Step 3 begins ."
        )
        step2_reflect = Node(instruction=step2_prompt, **common_params)

        step3_prompt = (
            "We are now at Step 3: Refinement Process. "
            "the results you got and evaluate possible enhancements and fixes."
        )
        step3_refine = Node(instruction=step3_prompt, **common_params)

        finalization_node = Node(**common_params)

        boolean_node_commons = {
            "condition": is_complete,
            "instruction_true": (
                "We are now at Step 5: Reply the user. Repeat the"
                " final solution you made, adding comments explaining it "
                " and how you reach to that conclusion. You can also say "
                "anything else you want the user to read. NOTICE that this is"
                " the only part of the reasoning chain they will get to read."
                " Remember the original question to better answer it."
                " Make sure the final answer is complete."
            ),
            "instruction_false": (
                "The text `Task Completed` was not found." "We are going back to step 2"
            ),
        }

        decision_node = BooleanNode(**boolean_node_commons)

        # Connect nodes in the workflow
        step1_plan.connect_to(step1_act)
        step1_act.connect_to(step2_reflect)
        step2_reflect.connect_to(decision_node)
        step3_refine.connect_to(step2_reflect)

        decision_node.connect_to_false(step3_refine)
        decision_node.connect_to_true(finalization_node)

        step1_plan.run(image=image, session_id=session_id)

        return self.agent.store.get_all(session_id=session_id)[-1]
