# ANVIL faculty demo (5 minutes)

## Before the demo

- Run `make setup` once and confirm `make run` opens the TUI.
- Set `AI_API_KEY` in the shell environment. Never paste it into the TUI or a
  command argument.
- Have a small, public GitHub issue ready that describes a concrete bug in a
  supported repository. Use an issue whose current base branch and tests are
  reachable from the demo network.
- Confirm `output/trace.jsonl` is available as the replay backup. If the live
  run cannot finish, use the saved sample trace and say that it is a replay.

## Live presentation

| Time | Action and narration |
|---|---|
| 0:00–0:40 | Start with `make run`. Explain that ANVIL is an agent harness: it tracks phases, tool calls, token use, and a patch rather than presenting a chat answer. |
| 0:40–1:00 | Paste the prepared GitHub issue URL into the issue field and select **Start**. Point out the issue text and repository being processed. |
| 1:00–3:20 | Follow the phase panel from INGEST through PATCH and VERIFY. Explain that the model sees the issue and repository tools; verification runs against the checked-out repository. If a provider or network limit stops the run, switch to replay rather than claiming a verified fix. |
| 3:20–4:20 | Show `output/patch.diff`, `output/report.md`, and `output/trace.jsonl`. Describe the result using the report's actual verification status. |
| 4:20–5:00 | Summarize the achieved phases, step/token counts, and whether verification passed. Leave time for questions. |

## Backup: replay a recorded run

If the live provider is unavailable or the run takes too long, open a terminal
in the repository and run the recorded trace in the TUI:

```sh
python -m anvil replay output/trace.jsonl
```

Replay makes no API calls. The screen shows the recorded events and timing;
press `space` to pause or resume, `→` to step forward, `←` to step backward,
`1` for normal speed, `4` for 4× speed, and `0` for instant playback. Narrate
the artifact as a replay, not as a new live run. To use the checked-in sample
instead of a fresh output trace, run `python -m anvil replay docs/sample_trace.jsonl`.
