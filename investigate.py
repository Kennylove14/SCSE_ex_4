## Import the necessary modules
import json

import ollama
## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result
MODEL_NAME = "qwen2.5:7b"

## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
## You may need to use json.dumps() to convert the available_items list into a JSON string.

def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found matching assistant.\n\n"
        "Rules:\n"
        "- You must use ONLY the list of found items provided to you; never invent items.\n"
        "- Not every detail of an item must match for it to be a possible match.\n"
        "- Compare the user's lost-item description against each item's type, color, "
        "location and date, and keep every item that is a plausible match.\n"
        "- Return ONLY a single JSON object, with no explanation or extra text, "
        "in EXACTLY this structure:\n"
        '{"matches": ["ITEM_ID"], "confidence": "LOW"}\n'
        '- "matches" must contain ALL possible matching item IDs.\n'
        '- "confidence" measures how confident you are about the matches and must be '
        'exactly one of: LOW, MEDIUM, HIGH.\n'
        '- If there is no match, return an empty list, e.g. {"matches": [], "confidence": "LOW"}.'
    )

    user_prompt = (
        "A user reported losing an item. Their description is:\n"
        f'"{description}"\n\n'
        "The unclaimed items available in the lost-and-found database are:\n"
        f"{json.dumps(available_items, indent=2, ensure_ascii=False)}\n\n"
        "Based on the rules, return all possible matches as the JSON result."
    )
    return system_prompt, user_prompt
    

## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result. 
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
    


## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence", and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs .
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False

    matches = result["matches"]
    confidence = result["confidence"]

    if not isinstance(matches, list):
        return False
    if not all(isinstance(item_id, str) for item_id in matches):
        return False
    if confidence not in ("LOW", "MEDIUM", "HIGH"):
        return False

    valid_ids = {item["id"] for item in available_items}
    for item_id in matches:
        if item_id not in valid_ids:
            return False
    return True


## Logic to display the matches found by Qwen in a user-friendly format.
## It should look something like this:
""" 
CAMPUS LOST-AND-FOUND ASSISTANT
==================================================

Describe the item you lost: I lost a black bag somewhere

Searching for possible matches...

MATCH RESULT
--------------------------------------------------
Confidence: MEDIUM

Possible matches:

ID: F101
Item: backpack
Color: black
Location: Library 2nd floor
Date found: 2026-09-15

Result saved to output/match_result.json
 """
## If no matches are found, it should display a message indicating that no matches were found, along with the empty list
def display_matches(result, available_items):
    print("MATCH RESULT")
    print("-" * 50)
    confidence = result.get("confidence", "UNKNOWN")
    print(f"Confidence: {confidence}")
    print()

    matches = result.get("matches", [])
    items_by_id = {item["id"]: item for item in available_items}

    if not matches:
        print("No matches were found for your description.")
        print()
        print("Possible matches: []")
        return

    print("Possible matches:")
    print()
    for item_id in matches:
        item = items_by_id.get(item_id)
        if item is None:
            continue
        print(f"ID: {item['id']}")
        print(f"Item: {item['item']}")
        print(f"Color: {item['color']}")
        print(f"Location: {item['location']}")
        print(f"Date found: {item['date']}")
        print()
    

## Control center for the entire program.
def main():
    items = load_items("found_items.json")
    available_items = get_unclaimed_items(items)

    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50)
    print()
    description = input("Describe the item you lost: ")
    print()
    print("Searching for possible matches...")
    print()

    system_prompt, user_prompt = build_prompt(description, available_items)
    response_text = ask_qwen(system_prompt, user_prompt)
    result = parse_response(response_text)

    if not validate_result(result, available_items):
        print("Warning: model output failed validation; showing it as-is.")
        print()

    display_matches(result, available_items)

    output_file = "output/match_result.json"
    save_result(result, output_file)
    print(f"Result saved to {output_file}")


if __name__ == "__main__":
    main()