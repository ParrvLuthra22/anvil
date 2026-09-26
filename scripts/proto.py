import httpx
import os
import sys

def get_base_ref(owner, repo, number):
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    client = httpx.Client(headers=headers)

    # 1. Fetch issue to check if it's open or if it is a PR
    issue_resp = client.get(f"https://api.github.com/repos/{owner}/{repo}/issues/{number}")
    if issue_resp.status_code != 200:
        return None, f"HTTP {issue_resp.status_code}", False
    
    issue = issue_resp.json()
    if issue.get("state") == "open":
        return None, "Issue is open", False
    
    # 2. Fetch timeline
    timeline_resp = client.get(f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/timeline")
    if timeline_resp.status_code != 200:
        return None, f"HTTP {timeline_resp.status_code} fetching timeline", False
    
    timeline = timeline_resp.json()
    
    fix_commit = None
    fix_pr = None

    for event in timeline:
        if event.get("event") == "closed" and event.get("commit_id"):
            fix_commit = event["commit_id"]
            break
        elif event.get("event") == "cross-referenced":
            source = event.get("source", {})
            issue_source = source.get("issue", {})
            if issue_source.get("pull_request") and issue_source.get("state") == "closed":
                # Check if it was merged, not just closed
                if issue_source.get("pull_request", {}).get("merged_at"):
                    fix_pr = issue_source["number"]
        elif event.get("event") == "connected":
            subject = event.get("subject", {})
            if subject.get("type") == "pull_request" and subject.get("state") == "merged":
                fix_pr = subject["number"]

    if fix_commit:
        return f"{fix_commit}^", f"Closed by commit {fix_commit}", True
    
    if fix_pr:
        # Fetch PR
        pr_resp = client.get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{fix_pr}")
        if pr_resp.status_code == 200:
            pr = pr_resp.json()
            if pr.get("merged"):
                merge_base = pr.get("base", {}).get("sha")
                return merge_base, f"Closed by merged PR #{fix_pr}", True

    return None, "Closed but no fix commit or merged PR found", True

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, get_base_ref("pallets", "click", arg))
