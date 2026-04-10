WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your available actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which available action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an available action for current step and present it within <action> </action> tags.
""".strip()


WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your available actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which available action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an available action for current step and present it within <action> </action> tags.
""".strip()


def format_action_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(none)"
    return "\n".join(
        f"Observation: {obs}\nAction: {action}" for obs, action in history
    )


def build_webshop_prompt(
    *,
    task_description: str,
    current_observation: str,
    available_actions: list[str],
    history: list[tuple[str, str]],
    history_length: int,
    step_count: int,
) -> str:
    hist = history[-history_length:]
    rendered_actions = "\n".join(f"'{action}'," for action in available_actions)
    if hist:
        return WEBSHOP_TEMPLATE.format(
            task_description=task_description,
            step_count=step_count,
            history_length=len(hist),
            action_history=format_action_history(hist),
            current_step=step_count + 1,
            current_observation=current_observation,
            available_actions=rendered_actions,
        )
    return WEBSHOP_TEMPLATE_NO_HIS.format(
        task_description=task_description,
        current_observation=current_observation,
        available_actions=rendered_actions,
    )
