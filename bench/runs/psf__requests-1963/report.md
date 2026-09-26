# ANVIL report

## Issue

- Issue: `Session.resolve_redirects` copies the original request for all subsequent requests, can cause incor (psf/requests)
- URL: https://github.com/psf/requests

The issue is about `Session.resolve_redirects` incorrectly copying the original request for each redirect iteration, causing wrong HTTP method selection. In a redirect chain like POST -> 303 (becomes GET) -> 307 (should preserve GET), the code incorrectly uses the original POST method instead of the current GET method. Need to find the `resolve_redirects` method in the requests codebase.

## Outcome

- Confidence: **low** (0.30)
- Reproduced before patching: yes (`python3 .anvil/repro.py`)
- Verified after patching: no
- Review: not run
- Patch sanity: passed

## Files changed

- `requests/sessions.py` (+5 / -2)
- `requests/utils.py` (+8 / -2)

## Tests run

- FAIL: repro `python3 .anvil/repro.py` (re-run by the harness)

## Summary

**Suspected file:line locations (ranked by likelihood):**
1. `requests/sessions.py:91` - `new_request = req.copy()` copies the original request on each iteration
2. `requests/sessions.py:102` - `method = req.method` gets method from original request instead of current request
3. `requests/sessions.py:88-120` - The entire `resolve_redirects` method loop logic

**Root cause hypothesis:**
In `Session.resolve_redirects`, the loop starts each iteration by copying the *original* request object (`req.copy()`) rather than the *current* request from the previous redirect. This means when a 303 redirect changes POST to GET, and then a subsequent 307 redirect should preserve that GET, the code incorrectly reverts to the original POST method because it reads `req.method` (the original request) instead of the method from the last request in the redirect chain.

**Relevant test files:**
- `test_requests.py` - Contains redirect-related tests (e.g., `test_cookie_sent_on_redirect`, `test_pyopenssl_redirect`, `test_uppercase_scheme_redirect`)
- No specific test found for the 303→307 redirect chain method preservation issue

## Budget

- LLM calls: 54
- Tokens: 302266
- Wall clock: 202.5s
- Patch attempts: 3, rollbacks: 0

## Tokens by phase

| Phase | Calls | Prompt tokens | Completion tokens | Prompt tokens per call |
|---|---:|---:|---:|---:|
| understand | 1 | 1,920 | 250 | 1,920 |
| localize | 9 | 35,107 | 1,291 | 3,901 |
| reproduce | 11 | 54,747 | 2,527 | 4,977 |
| patch | 32 | 201,559 | 4,865 | 6,299 |
| **total** | 53 | 293,333 | 8,933 | 5,535 |

## Known limitations

- LOCALIZE used its 8-call cap and was closed by the harness with what it had (done).
- REPRODUCE used its 10-call cap and was closed by the harness with what it had (done).
- PATCH used its 15-call cap and was closed by the harness with what it had (done).
- Stopped early: LLM call failed: LLM request failed with HTTP 429: {"error":{"message":"Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day","code":429,"metadata":{"headers":{"X-RateLimit-Limit":"50","X-RateLimit-Remaining":"0","X-RateLimit-Reset":"1790467200000"},"limit_source":"openrouter_free_tier_daily","remedy_hint":"Wait for the daily reset (see X-RateLimit-Reset), or purchase credits to raise your free-model daily limit.","provider_name":null}},"user_id":"[redacted]"} The provider's rate limit or quota was still exhausted after 5 attempts: wait and re-run, or use another model.
