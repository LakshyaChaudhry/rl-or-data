---
name: handoff
description: Write a self-contained brief so a fresh agent session can take over the current work. Use when the user says the context is full, asks for "a prompt for another agent", wants to hand work to an agent on a remote node or GPU box, or wants to continue in a new chat.
---

# Handoff

Produce one brief that a new agent can act on with no access to this conversation. The reader has the repo and nothing else.

## Before writing

1. Check the real state rather than recalling it: `git status`, current branch, last few commits, whether the branch is pushed, and any jobs still running (local background tasks, `squeue`, tmux sessions).
2. Ask the user which case this is if it is not clear from the request:
   - **Continue locally**: a new session on this machine picks up the same work.
   - **Remote agent**: an agent on a node or cloud box does one job there. It sees only what is pushed to git and what is on that machine's disk.
   - **Parallel task**: a second agent does one narrow deliverable while this session carries on.

## What the brief contains

Write it in this order. Keep it under about 60 lines.

1. **Goal**: one or two sentences on what the receiving agent must deliver, and how it will know it is done.
2. **Read first**: exact file paths, in order (CLAUDE.md, SPEC or HANDOFF, the task file, the one results file that matters).
3. **Current state**: branch, what is committed and pushed, what is uncommitted, what is running and where its logs are.
4. **Decisions already made**: the choices the user settled in this session, each with its one-line reason, so they are not reopened.
5. **Do not**: files and directories that must not be edited or deleted, runs that must not be interrupted, paths another agent is writing to.
6. **Steps**: the concrete next actions, with exact commands where they are known.
7. **Open questions**: anything undecided, and who decides it.
8. **Report back**: what to return (numbers with n, seeds and truncation rate; files changed; what was not verified) and where to put it (branch name to push, file to write).

## Rules

- Use exact paths, branch names, run ids and hosts. No "the script from earlier".
- For a remote agent, push the branch first and name it in the brief. Tell the agent to push to its own branch, never to main, and to leave data and results from other runs untouched.
- If two agents will share a filesystem, name the output directory each one owns.
- State plainly what has not been verified. Do not pass on a claim as fact because an earlier agent reported it.
- Save the brief to `handoffs/YYYY-MM-DD-<slug>.md` in the repo (create the directory if needed) and also print it in a fenced block so the user can paste it.
- If the repo has a status file (STATUS.md or HANDOFF.md), update its state section in the same pass.
