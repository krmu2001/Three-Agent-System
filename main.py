import asyncio
import json
from typing import Any, Dict
from groq import Groq

client = Groq()

def llm(system: str, user: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Call the LLM Chat Completion API.

    Args:
        system (str): The system prompt.
        user (str): The user prompt.
        model (str): The model to use.

    Returns:
        The message content as a string. Expected to be JSON formatted.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return (response.choices[0].message.content or "").strip()


async def llm_async(system: str, user: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Async Wrapper for LLM using asyncio.to_thread to avoid blocking the event loop.
    """
    return await asyncio.to_thread(llm, system, user, model)



async def agent_1() -> Dict[str, Any]:
    """
    Generate a dilemma with two opposing viewpoints and return it as JSON.
    """
    system = (
        "You are Agent 1: Dilemma Generator. "
        "Your task is to create a dilemma with exactly two opposing viewpoints. "
    )
    user = (
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "topic": string,\n'
        '  "dilemma": string,\n'
        '  "viewpoint_a": {"title": string, "position": string},\n'
        '  "viewpoint_b": {"title": string, "position": string}\n'
        "}\n"
        "Constraints:\n"
        "\\- The two viewpoints must clearly clash.\n"
        "\\- Keep it understandable in under 180 words total.\n"
    )
    text = await llm_async(system, user)
    return json.loads(text)


async def agent_advocate(agent_name: str, dilemma: Dict[str, Any], viewpoint: str) -> Dict[str, Any]:
    """
    Agent 2 and Agent 3 are Advocates for Viewpoint A and Viewpoint B respectively.
    They each take the dilemma JSON and defend their assigned viewpoint.
    """
    system = f"You are {agent_name}: an Advocate. Defend viewpoint {viewpoint}."
    user = (
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "agent": string,\n'
        '  "defended_viewpoint": "A" | "B",\n'
        '  "argument": string,\n'
        '  "key_points": [string, string, string]\n'
        "}\n\n"
        f"Dilemma JSON:\n{json.dumps(dilemma, ensure_ascii=False)}\n"
    )
    text = await llm_async(system, user)
    return json.loads(text)


async def main() -> None:
    """
    Run the dilemma generator and both advocate agents concurrently, then print results.
    """
    dilemma = await agent_1()
    a2_task = asyncio.create_task(agent_advocate("Agent 2", dilemma, "A"))
    a3_task = asyncio.create_task(agent_advocate("Agent 3", dilemma, "B"))
    a2, a3 = await asyncio.gather(a2_task, a3_task)

    print(json.dumps(dilemma, indent=2, ensure_ascii=False))
    print(json.dumps(a2, indent=2, ensure_ascii=False))
    print(json.dumps(a3, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
