
import os
from openai import OpenAI

import json


# Connect to NVIDIA API
client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"), # get NVIDIA API key from the environment
    base_url="https://integrate.api.nvidia.com/v1"
)

# Few-shot prompting: giving the model instructions and examples of the expected behavior
SYSTEM_PROMPT="""
    You're an expert AI assistant in resolving user queries using chain of thought.
    You work on START, PLAN, and OUTPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think enough PLAN has been done, finally you can give an OUTPUT.

    Rules:
    -Strictly follow the given JSON output format
    -Only run one step at a time.
    -The sequence of steps is START (where user gives an input), PLAN (That can be multiple times) and finally OUTPUT(which is going to the displayed to the user).

    Output JSON Format:
    {"step": "START" | "PLAN" | "OUTPUT", "content": "string"}

    Example:
    START: Hey, Can you solve 2 + 3 * 5 / 10
    PLAN: {"step": "PLAN", "content": "Looks like user is interested in math problem"} 
    PLAN: {"step": "PLAN", "content": "Looking at the problem, we should solve this using BODMAS method"}
    PLAN: {"step": "PLAN", "content": "Yes, The BODMAS is correct thing to be done here"}
    PLAN: {"step": "PLAN", "content": "first we must multiply 3 * 5 which is 15"}
    PLAN: {"step": "PLAN", "content": "Now the new equation is 2 + 15 / 10"}
    PLAN: {"step": "PLAN", "content": "We must perform division that is 15 / 10 = 1.5"}
    PLAN: {"step": "PLAN", "content": "Now the new equation is 2 + 1.5"}
    PLAN: {"step": "PLAN", "content": "Now finally lets perform the add 3.5"}
    PLAN: {"step": "PLAN", "content": "Great, we have solved and finally left with 3.5 as ans"}
    OUTPUT: {"step": "OUTPUT", "content": "3.5"}

"""
print("\n\n\n")

# Automating the processs
message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_query = input("👉")
message_history.append({"role": "user", "content": user_query})

while True:

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        temperature=0.2,
        max_tokens=1024,
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        messages=message_history
    )

    raw_result = response.choices[0].message.content

    message_history.append({
        "role": "assistant",
        "content": raw_result
    })

    parsed_result = json.loads(raw_result)

    if parsed_result.get("step") == "START":
        print("🔥", parsed_result.get("content"))

    elif parsed_result.get("step") == "PLAN":
        print("🧠", parsed_result.get("content"))

    elif parsed_result.get("step") == "OUTPUT":
        print("🤖", parsed_result.get("content"))
        break

    message_history.append({
        "role": "user",
        "content": "Continue to the next step."
    })


print("\n\n\n")


# OUTPUT :
# 👉hey can you solve 2 + 3 / 10 * 2 * 4 * 3 / 20
# 🧠 The user wants me to solve the math problem: 2 + 3 / 10 * 2 * 4 * 3 / 20. I need to apply the order of operations (BODMAS/BIDMAS) which means I should handle multiplication and division from left to right before addition.
# 🧠 First, I'll handle the division and multiplication from left to right. The expression is 2 + 3 / 10 * 2 * 4 * 3 / 20. Let's break it down step by step.
# 🧠 First, compute 3 / 10 = 0.3. The expression becomes 2 + 0.3 * 2 * 4 * 3 / 20.
# 🧠 Next, compute 0.3 * 2 = 0.6. The expression becomes 2 + 0.6 * 4 * 3 / 20.
# 🧠 Compute 0.6 * 4 = 2.4. The expression becomes 2 + 2.4 * 3 / 20.
# 🧠 Compute 2.4 * 3 = 7.2. The expression becomes 2 + 7.2 / 20.
# 🧠 Compute 7.2 / 20 = 0.36. The expression becomes 2 + 0.36.
# 🧠 Compute 2 + 0.36 = 2.36. The final answer is 2.36.
# 🤖 2.36
