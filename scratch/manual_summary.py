import json
import os
import re
import sys
from pathlib import Path

# Add brain module to path
sys.path.append("/home/aryan/Documents/projects/digital-brain")

from brain.router import Router
from brain.vault import Vault
from brain.models import MemoryNeuron, NeuronSource

def parse_overview(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = content.strip().split("\n")
    
    full_text = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if data.get("type") == "USER_INPUT" and "content" in data:
                match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", data["content"], re.DOTALL)
                if match:
                    full_text.append("USER: " + match.group(1).strip())
                else:
                    full_text.append("USER: " + data["content"].strip())
            elif data.get("type") == "PLANNER_RESPONSE" and "content" in data:
                full_text.append("AI: " + data["content"].strip())
        except Exception:
            pass
    return "\n\n".join(full_text)

def main():
    target_dir = Path("/home/aryan/.gemini/antigravity/brain/77825ecb-a70f-44d7-be62-5c414c261ae9")
    overview_path = target_dir / ".system_generated" / "logs" / "overview.txt"
    
    print("Parsing conversation...")
    text = parse_overview(overview_path)
    if not text:
        print("Could not parse text.")
        return
        
    router = Router()
    vault = Vault()
    vault.sync_id_counters()
    
    print("Generating rich summary via local LLM...")
    prompt = f"""You are a Second Brain summarizer.
Read the following IDE conversation between a USER and an AI.
Write a dense, highly informative 2-paragraph summary of exactly what technical work was accomplished.
Identify any distinct projects (like "Digital Brain", "Uni", "Musico", "PocketBase") and wrap them in double brackets like [[Digital Brain]].
Identify any core technologies used (like "Python", "ChromaDB", "Qwen") and wrap them in double brackets like [[ChromaDB]].

Conversation:
{text[:4000]}  # Trim to fit context

Return ONLY the summary markdown.
"""
    
    summary = router.ask_local(prompt)
    print("\n--- GENERATED SUMMARY ---")
    print(summary)
    print("-------------------------\n")
    
    print("Saving to Vault...")
    memory = MemoryNeuron(
        title="Digital Brain 100x Optimization Session",
        source=NeuronSource.ANTIGRAVITY_IDE.value,
        raw_log_path=str(overview_path),
        body=summary.strip()
    )
    
    path = vault.write_neuron(memory)
    print(f"Saved to {path}")

if __name__ == "__main__":
    main()
