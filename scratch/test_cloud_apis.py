import sys
from pathlib import Path

# Add brain module to path
sys.path.append("/home/aryan/Documents/projects/digital-brain")

from brain.router import Router
from brain.ingestion.ide_connector import IDEConversationParser

def main():
    router = Router()
    parser = IDEConversationParser()
    
    # Pick the 'Optimizing Uni App Experience' conversation
    target_dir = Path("/home/aryan/.gemini/antigravity/brain/09cf4a7e-f208-4b39-92d1-52283470f44f")
    overview_path = target_dir / ".system_generated" / "logs" / "overview.txt"
    
    if not overview_path.exists():
        print("Log not found.")
        return
        
    print(f"Parsing conversation from {overview_path.parent.parent.name}...")
    parsed = parser.parse_overview(overview_path)
    text = parsed.get("raw_content", "")
    
    prompt = f"""CREATE: Summarize this content in 7-8 sentences.
Focus on: what happened, what was decided, what was learned.

Text: {text[:4000]}

Key themes (as comma-separated list):"""
    
    print("\nSending to Cloud APIs (xAI -> Gemini)...")
    
    # Using route() instead of ask_local() will hit the cloud APIs because 
    # we configured them with priority in router.py!
    result = router.route(prompt)
    
    print("\n--- CLOUD API RESULT ---")
    print(f"Provider Used: {result['provider']}")
    print(f"Detected Intent: {result['mode']}")
    print("-" * 24)
    print(result['response'])
    print("------------------------\n")

if __name__ == "__main__":
    main()
