import json

# Example UMB file path (re-upload required due to environment reset)
umb_file_path = "emergence_UMB.json"
output_file_path = "umb_clean_log.txt"

# Load UMB data
with open(umb_file_path, "r", encoding="utf-8") as f:
    umb_data = json.load(f)

# Ensure data is a list of dicts with "prompt" and "reply"
if isinstance(umb_data, list):
    clean_lines = []
    for entry in umb_data:
        prompt = entry.get("input", "").strip()
        reply = entry.get("output", "").strip()
        if prompt or reply:
            if prompt:
                clean_lines.append(f"🧠 You> {prompt}")
            if reply:
                clean_lines.append(f"🖥 Phasers> {reply}")
            clean_lines.append("")  # empty line between entries

    # Save clean log to file
    with open(output_file_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(clean_lines))

output_file_path  # Return path to the generated clean log file
