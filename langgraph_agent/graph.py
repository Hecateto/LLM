import json
import logging
import re
from typing import Any, Dict, List

from json_repair import repair_json  # 新增：更健壮的JSON修复库
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    ToolCall
)
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import ValidationError

from prompts import (
    PLAN_SYSTEM_PROMPT,
    PLAN_CREATE_PROMPT,
    UPDATE_PLAN_PROMPT,
    EXECUTE_SYSTEM_PROMPT,
    EXECUTION_PROMPT,
    REPORT_SYSTEM_PROMPT
)
from state import State, Plan, Step
from tools import create_file, str_replace, shell_exec, send_message

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    temperature=0,
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-tvzlvyogdvfnojvinjimgqtuhohdpegqitvkybhezajtremm"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def extract_json(text):
    text = text.strip()

    # 优先匹配 ```json ... ``` 代码块
    match_json_block = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match_json_block:
        return match_json_block.group(1).strip()

    # 其次匹配 ``` ... ``` 代码块
    match_block = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if match_block:
        return match_block.group(1).strip()

    # 最后尝试查找第一个 '{' 和最后一个 '}'
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1].strip()

    # 如果都找不到，返回原始文本，让 repair_json 尝试修复
    return text


def parse_plan_json(json_str: str) -> Plan:
    logger.info(f"Parsing plan JSON: {json_str}")
    try:
        repaired = repair_json(json_str, return_objects=False)
        plan_dict = json.loads(repaired)
        steps = []
        for step_dict in plan_dict.get('steps', []):
            try:
                steps.append(Step(**step_dict))
            except ValidationError as ve:
                logger.error(f"Step validation error: {ve}")
                logger.info(f"Step's status is {step_dict.get('status', 'N/A')}")
                steps.append(Step(
                    title=step_dict.get('title', 'Untitled Step'),
                    description=step_dict.get('description', 'No description provided.'),
                    status='pending'
                ))
        return Plan(
            goal=plan_dict.get('goal', ''),
            thought=plan_dict.get('thought', ''),
            steps=steps
        )
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Error parsing plan JSON: {e}, Raw JSON: {json_str}")
        raise


def create_planner_node(state: State):
    '''
    State: user_message, plan, observations, final_report
    '''

    logger.info("Creating Planner Node...")

    try:
        messages_list: List[BaseMessage] = state.get('messages', [])
        user_message = state.get('user_message', '')
        if not user_message:
            user_message = state['user_message']

        messages: List[BaseMessage] = [
            SystemMessage(content=PLAN_SYSTEM_PROMPT),
            HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message=user_message))
        ]
        response = llm.invoke(messages)
        raw_content = response.content or ""
        json_str = extract_json(raw_content)
        plan = parse_plan_json(json_str)
        new_messages = messages_list + [AIMessage(
            content=json.dumps(plan.dict(), ensure_ascii=False, indent=2)
        )]

        return {"messages": new_messages, "plan": plan, "error": None}
    except Exception as e:
        error_msg = f"Error in create_planner_node: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def update_planner_node(state: State):
    logger.info("Updating Planner Node...")
    try:
        plan = state.get('plan') or Plan()

        if not plan or not plan.steps:
            error_msg = "No existing plan found to update."
            return {"error": error_msg}

        cur_plan_dict = plan.dict()
        update_prompt = UPDATE_PLAN_PROMPT.format(
            plan=json.dumps(cur_plan_dict, ensure_ascii=False),
            goal=plan.goal
        )

        messages_list = state.get('messages', [])
        update_messages = messages_list + [HumanMessage(content=update_prompt)]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.invoke(update_messages)
                content = response.content or ""

                json_str = extract_json(content)
                updated_plan = parse_plan_json(json_str)
                updated_plan.goal = plan.goal

                new_messages = messages_list + [
                    HumanMessage(content=update_prompt),
                    AIMessage(content=json.dumps(updated_plan.dict(), ensure_ascii=False, indent=2))
                ]

                return {"messages": new_messages, "plan": updated_plan, "error": None}

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed to update plan: {str(e)}"
                logging.warning(error_msg)

        final_error = f"Failed to update plan after {max_retries} attempts."
        return Command(goto="error", update={"error": final_error})
    except Exception as e:
        error_msg = f"Error in update_planner_node: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def execute_node(state: State):
    '''
    State: user_message, plan, observations, final_report
    Plan: goal, thought, steps
    Steps: title, description, status
    '''

    logger.info("Executing Node...")

    try:
        plan = state.get('plan') or Plan()
        cur_step, cur_step_idx = None, -1

        for idx, step in enumerate(plan.steps):
            if step.status == 'pending':
                cur_step = step
                cur_step_idx = idx
                break

        if not cur_step:
            logger.info("All steps completed. Execution node finished.")
            # 返回空字典或包含 error=None，让条件边决定下一步
            return {"error": None}

        logger.info(f"Current Step: {cur_step.title} - {cur_step.description}")

        observations = state.get('observations', [])
        messages_list = state.get('messages', [])
        user_message = state.get('user_message', '')

        exec_messages: List[BaseMessage] = observations + [
            SystemMessage(content=EXECUTE_SYSTEM_PROMPT),
            HumanMessage(content=EXECUTION_PROMPT.format(
                user_message=user_message,
                step=cur_step
            ))
        ]

        # logger.info(f"Execution Messages: {exec_messages}")

        tools = {
            "create_file": create_file,
            "str_replace": str_replace,
            "shell_exec": shell_exec,
            "send_message": send_message
        }

        tool_executed = False
        max_loops = 3
        loop_count = 0
        final_tool_result: Dict[str, Any] = {"success": False, "error": "Not executed any tool."}

        logger.info("Starting execution loop...")

        while loop_count < max_loops:
            loop_count += 1

            response = llm.bind_tools(list(tools.values())).invoke(exec_messages)
            if not response.tool_calls:
                logger.info("No tool calls in response, finishing execution loop.")
                break

            # logger.info(f"Tool call found: {response.tool_calls[0]}")

            valid_tool_calls: List[ToolCall] = []
            parsed_args_for_call = {}  # 存储解析后的参数

            try:
                for tool_call in response.tool_calls:
                    # 复制 tool_call 字典
                    valid_call = tool_call.copy()

                    if isinstance(valid_call.get('args'), str):
                        logger.warning(f"Detected string args, parsing: {valid_call['args']}")
                        # 解析字符串为字典
                        parsed_args = json.loads(valid_call['args'])
                        valid_call['args'] = parsed_args
                    else:
                        parsed_args = valid_call.get('args', {})

                    # 存储解析后的参数，用于后续工具调用
                    if 'id' in valid_call:
                        parsed_args_for_call[valid_call['id']] = parsed_args

                    valid_tool_calls.append(valid_call)

            except json.JSONDecodeError as e:
                # 如果 LLM 返回的 JSON 字符串无效，这是一个严重错误
                error_msg = f"Failed to parse tool call args JSON string from LLM: {e}. String was: {tool_call.get('args')}"
                logger.error(error_msg)
                # 附加原始的、无效的 response
                exec_messages.append(response)
                first_tool_call_id = response.tool_calls[0].get('id', 'unknown_id')
                # 附加一个 ToolMessage 来报告这个解析错误
                exec_messages.append(ToolMessage(content=error_msg, tool_call_id=first_tool_call_id))
                continue  # 跳到 while 循环的下一次迭代

            # 2. 创建一个新的、合法的 AIMessage
            # 我们必须复制所有相关属性，特别是 ID，以保持消息的连续性
            valid_response_message = AIMessage(
                content=response.content,
                tool_calls=valid_tool_calls,
                id=response.id,
                usage_metadata=response.usage_metadata,
                response_metadata=response.response_metadata,
            )

            # 3. 将这个【新创建的、合法的】AIMessage 添加到历史记录中
            exec_messages.append(valid_response_message)

            if not valid_tool_calls:
                logger.warning("AIMessage had tool_calls but none were valid or processed.")
                continue

            tool_call = valid_tool_calls[0]
            tool_name = tool_call['name']
            tool_id = tool_call['id']

            tool_args = parsed_args_for_call.get(tool_id, {})

            if tool_name not in tools:
                error_msg = f"Tool {tool_name} not found."
                logger.warning(error_msg)
                exec_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                continue

            try:
                tool_result = tools[tool_name].invoke(tool_args)
                final_tool_result = tool_result
                tool_executed = True
                logger.info(f"Tool {tool_name} invoked with args {tool_args}, result: {tool_result}")
            except Exception as e:
                tool_result = {"success": False, "error": f"Failed to invoke tool: {str(e)}"}
                final_tool_result = tool_result
                logger.error(f"Error invoking tool {tool_name}: {str(e)}", exc_info=True)

            exec_messages.append(ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=tool_id
            ))
            logger.info(f"Final tool result after execution loop: {final_tool_result}")

        updated_plan = plan.copy(deep=True)

        if tool_executed:
            if final_tool_result.get("success", False):
                updated_plan.steps[cur_step_idx].status = 'completed'
                execution_summary = f"Step '{cur_step.title}' executed successfully: {final_tool_result}"
            else:
                execution_summary = f"Step '{cur_step.title}' execution failed: {final_tool_result}"
                logger.warning(execution_summary)
        else:
            execution_summary = f"Step '{cur_step.title}' completed without tool execution. LLM response: {response.content}"
            logger.warning(
                f"No tool executed for step {cur_step_idx}, marking as complete to avoid loop. LLM said: {response.content}")
            updated_plan.steps[cur_step_idx].status = 'completed'

        new_messages = messages_list + [AIMessage(content=execution_summary)]
        new_observations = observations + [AIMessage(content=execution_summary)]

        return {
            "messages": new_messages,
            "observations": new_observations,
            "plan": updated_plan,
            "error": None
        }

    except Exception as e:
        error_msg = f"Error in execute_node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def report_node(state: State):
    logger.info("Generating Report Node...")
    try:
        observations = state.get('observations', [])
        messages_list = state.get('messages', [])
        report_messages: List[BaseMessage] = observations + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]

        tools = {
            "create_file": create_file,
            "str_replace": str_replace,
            "shell_exec": shell_exec,
            "send_message": send_message
        }

        max_loops = 3
        loop_count = 0
        while loop_count < max_loops:
            loop_count += 1
            response = llm.bind_tools(list(tools.values())).invoke(report_messages)

            if not response.tool_calls:
                logger.info("No tool calls in response, finishing report generation.")
                break

            valid_tool_calls: List[ToolCall] = []
            parsed_args_for_call = {}

            try:
                for tool_call in response.tool_calls:
                    valid_call = tool_call.copy()
                    if isinstance(valid_call.get('args'), str):
                        logger.warning(f"Detected string args, parsing: {valid_call['args']}")
                        parsed_args = json.loads(valid_call['args'])
                        valid_call['args'] = parsed_args
                    else:
                        parsed_args = valid_call.get('args', {})

                    if 'id' in valid_call:
                        parsed_args_for_call[valid_call['id']] = parsed_args

                    valid_tool_calls.append(valid_call)

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse tool call args JSON string from LLM: {e}. String was: {tool_call.get('args')}"
                logger.error(error_msg)
                report_messages.append(response)
                first_tool_call_id = response.tool_calls[0].get('id', 'unknown_id')
                report_messages.append(ToolMessage(content=error_msg, tool_call_id=first_tool_call_id))
                continue

            valid_response_message = AIMessage(
                content=response.content,
                tool_calls=valid_tool_calls,
                id=response.id,
                usage_metadata=response.usage_metadata,
                response_metadata=response.response_metadata,
            )

            report_messages.append(valid_response_message)

            if not valid_tool_calls:
                logger.warning("AIMessage had tool_calls but none were valid or processed.")
                continue

            tool_call = valid_tool_calls[0]
            tool_name = tool_call['name']
            tool_id = tool_call['id']
            tool_args = parsed_args_for_call.get(tool_id, {})

            if tool_name not in tools:
                error_msg = f"Tool {tool_name} not found."
                report_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
                continue

            try:
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"Tool {tool_name} invoked with args {tool_args}, result: {tool_result}")
            except Exception as e:
                tool_result_dict = {"success": False, "error": f"Error invoking tool {tool_name}: {str(e)}"}
                tool_result = json.dumps(tool_result_dict, ensure_ascii=False)
                logger.error(tool_result, exc_info=True)

            if not isinstance(tool_result, str):
                tool_result_str = json.dumps(tool_result, ensure_ascii=False)
            else:
                tool_result_str = tool_result

            report_messages.append(ToolMessage(
                content=tool_result_str,
                tool_call_id=tool_id
            ))

        final_report = response.content if hasattr(response, 'content') else "No report generated."
        new_messages = messages_list + [AIMessage(content=final_report)]
        return {
            "messages": new_messages,
            "final_report": final_report,
            "error": None
        }
    except Exception as e:
        error_msg = f"Error in report_node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"final_report": f"Failed to generate report due to an error: {error_msg}", "error": error_msg}


def error_node(state: State):
    error_message = state.get('error') or state.get('error_message', 'Unknown error occurred in error_node.')
    logger = logging.getLogger(__name__)
    logger.error(f"Error handled by error_node: {error_message}")

    messages = state.get('messages', [])
    messages.append(AIMessage(content=f"An error occurred: {error_message}"))

    return {
        "messages": messages,
        "final_report": f"Task failed due to an error: {error_message}"
    }