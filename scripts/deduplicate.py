import os
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY")
GITHUB_EVENT_PATH = os.environ.get("GITHUB_EVENT_PATH")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://models.inference.ai.azure.com")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not GITHUB_TOKEN or not GITHUB_REPOSITORY or not GITHUB_EVENT_PATH:
    print("Error: Missing required environment variables.")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_issue_from_event():
    with open(GITHUB_EVENT_PATH, 'r') as f:
        event_data = json.load(f)
    return event_data.get('issue')

def fetch_open_issues(current_issue_number):
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/issues?state=open&per_page=100"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Error fetching issues: {response.status_code}")
        return []
    
    issues = response.json()
    # Filter out proper pull requests and the current issue itself
    return [
        i for i in issues 
        if "pull_request" not in i 
        and i['number'] != current_issue_number
    ]

def get_issue_text(issue):
    title = issue.get('title', '')
    body = issue.get('body', '') or ''
    return f"{title}\n{body}"

def semantic_search(new_issue_text, existing_issues):
    if not existing_issues:
        return []
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding issues...")
    new_embedding = model.encode(new_issue_text, convert_to_tensor=True)
    existing_texts = [get_issue_text(i) for i in existing_issues]
    existing_embeddings = model.encode(existing_texts, convert_to_tensor=True)
    
    print("Calculating similarity...")
    # Cosine similarity
    cosine_scores = util.cos_sim(new_embedding, existing_embeddings)[0]
    
    # Filter matches with similarity > 0.5 (Lowered from 0.6)
    matches = []
    for idx, score in enumerate(cosine_scores):
        print(f"DEBUG: Comparing with #{existing_issues[idx]['number']} - Score: {score:.4f}")
        if score > 0.5:
            matches.append({
                "issue": existing_issues[idx],
                "score": float(score)
            })
    
    # Sort by score descending
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:5] # Top 5

def check_duplicate_with_llm(new_issue, candidate_issue):
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )

    prompt = f"""
You are an expert GitHub issue triager. Determine if "Issue A" is a duplicate of "Issue B".

Issue A (New):
Title: {new_issue['title']}
Body: {new_issue['body']}

Issue B (Existing #{candidate_issue['number']}):
Title: {candidate_issue['title']}
Body: {candidate_issue['body']}

Is Issue A a duplicate of Issue B? 
Respond with ONLY "YES" or "NO" followed by a very brief one-sentence explanation.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        print(f"LLM Comparison with #{candidate_issue['number']}: {content}")
        return content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "NO Error"

def post_comment(issue_number, duplicate_number, reason):
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/issues/{issue_number}/comments"
    body = f"It looks like this issue might be a duplicate of #{duplicate_number}.\n\nReason: {reason}\n\n(Automated by Hybrid Semantic Triage)"
    requests.post(url, headers=HEADERS, json={"body": body})

def main():
    print("Starting Hybrid Deduplication...")
    
    current_issue = get_issue_from_event()
    if not current_issue:
        print("No issue found in event payload.")
        return

    print(f"Processing Issue #{current_issue['number']}: {current_issue['title']}")
    
    existing_issues = fetch_open_issues(current_issue['number'])
    print(f"Found {len(existing_issues)} existing open issues.")
    
    matches = semantic_search(get_issue_text(current_issue), existing_issues)
    print(f"Found {len(matches)} potential semantic matches.")
    
    for match in matches:
        candidate = match['issue']
        print(f"Checking candidate #{candidate['number']} (Score: {match['score']:.2f})...")
        
        llm_decision = check_duplicate_with_llm(current_issue, candidate)
        
        if llm_decision.upper().startswith("YES"):
            print(f"Confirmed duplicate: #{candidate['number']}")
            post_comment(current_issue['number'], candidate['number'], llm_decision)
            break # Stop after finding the first confirmed duplicate to avoid spam
        else:
            print(f"Not a duplicate of #{candidate['number']}")

if __name__ == "__main__":
    main()
