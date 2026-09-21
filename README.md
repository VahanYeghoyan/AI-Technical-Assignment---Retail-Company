# Retail Data Analysis Chat Assistant

A CLI chat agent that lets non-technical executives ask questions about retail
sales data in plain language, discuss the answers, and save reports with action
items. It queries the public `bigquery-public-data.thelook_ecommerce` dataset.

The design brief is in [`docs/architecture.md`](docs/architecture.md) — that
document is the main deliverable and covers the requirements this prototype
implements in design only.

---

## What actually runs

| Requirement | State |
|---|---|
| Safety & PII masking | Implemented, 74 tests (`tests/test_safety.py`) |
| High-stakes oversight (destructive ops) | Implemented, 21 tests (`tests/test_reports.py`) |
| Resilience & error handling | Implemented, 34 tests (`tests/test_resilience.py`) |
| Observability | Implemented, replayable traces, 7 tests |
| Hybrid intelligence (Golden Bucket) | Design only — [architecture](docs/architecture.md#1-hybrid-intelligence) |
| Learning loop | Design only |
| Quality assurance | Design only |
| Agility (persona) | Implemented (hot-reload) + design |

**182 tests pass with no credentials and no LLM quota** — 174 of them without the
optional web UI installed, whose tests skip when streamlit is absent. That is
deliberate: the safety and oversight guarantees are the ones that must be
verifiable in CI on any machine, without reaching a third party.

Separately, the full path has been verified live against Vertex AI Gemini 3.6
Flash and BigQuery — including entitlement rewriting on the real warehouse
(a Women's-scoped user asking explicitly for Men's revenue gets an empty result,
not an error and not data).

---

## Setup

### 1. Python

Python **3.13+** recommended (verified on 3.14.0). 3.11 is the hard floor.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. BigQuery credentials

The Python client needs Application Default Credentials — note this is a
*different* credential from the one `gcloud auth login` writes, and `bq`
working on the command line does **not** mean the client library will work.

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Any project with the BigQuery API enabled works; the free sandbox is sufficient,
and queries against the public dataset are billed to your project (~1 TB/month
free). Every query here is dry-run and capped well below that.

### 3. Model access

```bash
cp .env.example .env
```

Two options. **Vertex AI is recommended and needs no API key** — it reuses the
Application Default Credentials you just set up, so anything that can read the
warehouse can also reach the model, and access is IAM rather than a bearer
secret that has to be minted, stored and rotated:

```bash
gcloud services enable aiplatform.googleapis.com
# .env: LLM_PROVIDER=vertex
```

Or an [AI Studio](https://aistudio.google.com/apikey) key
(`LLM_PROVIDER=gemini`, `GOOGLE_API_KEY=…`). Be aware its free tier is small and
returns `429 Your prepayment credits are depleted` on *every* model once spent —
which is what happened during this project's development, and is why the offline
stub below exists.

### 4. Run

```bash
python -m retail_agent.cli --user maya
python -m retail_agent.cli --list-users    # see the available personas
```

### 5. Optional: the web UI

The same agent in a browser, for anyone who would rather not read a terminal.
Streamlit is deliberately **not** in `requirements.txt` — the CLI is the
interface this prototype is built around and installing it should not pull in a
web stack:

```bash
pip install -r requirements-ui.txt
streamlit run streamlit_app.py
```

It is a second *renderer*, not a second agent. Both front ends call
`retail_agent/dispatch.py`, which decides what a message means — a live deletion
confirmation, a slash command, or a question for the model — so the confirmation
state machine, the model-bypassing slash commands and the entitlement scoping
behave identically in both. The sidebar switches executive persona, shows the
live data scope and persona version, and starts a new conversation.

Deliberately, there are no Confirm / Cancel buttons: above three reports the
oversight flow requires the user to type the count (`delete 7`), and a button
would turn that back into the reflexive single click the rule exists to prevent.

Every answer carries the same turn telemetry the traces record — status, model
calls, SQL attempts, self-corrections, tokens, bytes billed and the trace id —
plus the scope-rewritten SQL that actually reached BigQuery.

### Running with no credentials at all

```bash
LLM_PROVIDER=stub python -m retail_agent.cli --user maya
```

The stub provider returns scripted responses. It exists because an agent that
can only be demonstrated when a third party is up fails its own resilience
requirement — and because this project's API key hit
`429 Your prepayment credits are depleted` during development.

---

## Example session

Real output, signed in as `maya` (VP, Women's Division) against live Vertex
Gemini 3.6 Flash and live BigQuery:

```
› What were my top 3 brands by revenue this year?

Jones New York generated the highest net revenue in the Women's Division
year-to-date, delivering $21,167.40, followed by True Religion ($18,517.00)
and Calvin Klein ($17,189.20).

Year-to-Date Brand Performance (Women's Division)
Date range: January 1, 2026 through September 18, 2026
Metric: Net Revenue = SUM(sale_price) excluding Cancelled and Returned orders

| Brand           | Net Revenue | Gross Margin | Units Sold | Avg Price/Unit |
| :-------------- | ----------: | -----------: | ---------: | -------------: |
| Jones New York  |  $21,167.40 |   $11,311.70 |        217 |         $97.55 |
| True Religion   |  $18,517.00 |    $8,732.45 |         89 |        $208.06 |
| Calvin Klein    |  $17,189.20 |    $9,035.74 |        279 |         $61.61 |

Key Observations
• Volume vs. Price Realization: True Religion achieved its revenue rank on high
  price points ($208.06 average unit price) with only 89 units sold. Calvin
  Klein drove revenue through volume (279 units) at $61.61 per unit.
• Margin Efficiency: Jones New York delivered the highest absolute gross margin
  ($11,311.70, a 53.4% margin rate), outperforming True Religion (47.2%).

[status=ok llm=2 sql=1 corrections=0 tok=7088->687 bytes=10,055,935]
```

Note it volunteered the metric definition and the date range unprompted — those
come from `config/metrics.yaml` and `config/persona.yaml`, not from the question.

A PII request, same session:

```
› Who is the customer with email alice@example.com?

Customer email addresses and personal details are restricted for privacy
protection; this assistant analyzes aggregate sales, product performance, and
customer cohort metrics.

I can, however, show you customer spending distribution by tier or churn rates
across acquisition channels for the Women's Division.
```

That refusal came from the model, but it is not what enforces the rule: had it
tried, the SQL guard would have rejected any query touching `email` before it
reached BigQuery.

Maya is scoped to the Women's department, so the SQL that actually reached
BigQuery was not the SQL the model wrote:

```sql
SELECT p.brand, SUM(oi.sale_price) AS revenue
FROM (
  SELECT scope_oi.*
  FROM `bigquery-public-data.thelook_ecommerce.order_items` AS scope_oi
  JOIN `bigquery-public-data.thelook_ecommerce.products` AS scope_p
    ON scope_p.id = scope_oi.product_id
  WHERE scope_p.department IN ('Women')      -- injected, not model-written
) AS oi
JOIN ( ... same treatment for products ... ) AS p
  ON p.id = oi.product_id
GROUP BY p.brand
```

### Commands (identical in the CLI and the web UI)

| Command | Does |
|---|---|
| `/reports` | list your saved reports |
| `/undo` | restore the reports deleted most recently |
| `/trace [id]` | recent turns, or the full event stream for one turn |
| `/whoami` | identity, data scope, persona version |
| `/persona` | show the live persona version |
| `/quit` | exit |

Slash commands deliberately bypass the model, so they keep working during an
LLM outage.

---

## How the four prototype requirements are met

### Safety & PII masking

Enforcement is in code (`retail_agent/safety/`), never delegated to the prompt.
Every model-written query is parsed with **sqlglot** and must survive:

1. exactly one statement (blocks `SELECT 1; DROP TABLE x`);
2. read-only — `SELECT`/`WITH` only;
3. a table allowlist, including the project and dataset, so a fully-qualified
   name cannot reach another project;
4. **no PII column anywhere** — not just in the output. Blocking them only in
   `SELECT` would still allow
   `SELECT COUNT(*) FROM users WHERE email = 'x@y.com'`, which leaks whether a
   named person exists. PII columns are rejected in `WHERE`, `JOIN` and
   `ORDER BY` too;
5. no `SELECT *` over `users`, since a star expands into PII columns. `COUNT(*)`
   is explicitly still allowed;
6. **no whole-row references.** In BigQuery a table alias used on its own *is*
   the row, so `TO_JSON_STRING(u)`, `SELECT u`, `ARRAY_AGG(u)` and
   `JSON_VALUE(TO_JSON_STRING(u), '$.first_name')` each return every column —
   including the ones rule 4 blocks — without naming any of them. A column-name
   check cannot see those, and neither can the result scrubber: it matches
   emails and addresses, not names or postcodes, and a STRUCT is not a string.
   The oracle comes back this way too
   (`WHERE STRPOS(TO_JSON_STRING(u), 'alice@example.com') > 0`), so the row
   itself is rejected.

A reference carrying a project or a dataset is never treated as a CTE, however
it is named. Matching on the bare name let a query define a CTE after any table
it wanted and then address the real one by its full path — escaping both the
allowlist (`…thelook_ecommerce.events.ip_address`, or another project entirely)
and the entitlement rewrite (`…order_item*`, the wildcard form of
`order_items`, which showed a Women's-division user the whole company's
revenue).

PII columns (verified against the live schema): `first_name`, `last_name`,
`email`, `street_address`, `postal_code`, `latitude`, `longitude`, and
`user_geom` — a `GEOGRAPHY` column that pinpoints a home and is easy to miss.

Then four more layers: result sets are scrubbed before they reach the model,
every customer id column in a result is replaced with a salted pseudonym, the
final answer is scrubbed before display, and trace files are scrubbed on the way
in — a debug log should not quietly become the one place customer emails are
retained. "Top customers" works, but customers appear as salted pseudonyms
(`CUST-A83F21`), stable within a deployment so follow-up questions still resolve.
The pseudonym is applied in code, on the way out of BigQuery: asking the model
to write `CUST-` ids it had never been shown only got them invented.

The text scrubber matches address- and phone-*shaped* text, case included. A
case-insensitive version matched most business writing — "focus on 3 levers to
drive repeat purchases" redacted as an address — and an answer mutilated by its
own safety layer is a bug every reader sees.

**Per-user entitlements.** `config/entitlements.yaml` maps each executive to a
predicate over products (department / category / brand). The guard rewrites every
table reference into a scope-filtered subquery, propagating through
`order_items`, `orders` (orders containing an in-scope item) and `users`
(customers who bought one). Aliases and CTEs are preserved, so the model's own
SQL keeps working. In production this belongs in BigQuery authorized views plus
row-level access policies; it lives in-process here because the dataset is public
and read-only to us.

### High-stakes oversight

**The model has no delete tool.** Its only option is `propose_delete_reports`,
which resolves the match set and parks it. Execution lives in
`retail_agent/confirmation.py`, behind a human confirmation, in code the model
never touches — a jailbroken model still cannot delete a row.

Strictness scales with blast radius, so the flow stays usable:

- 1–3 reports → `yes` confirms.
- 4+ reports → the user must type the count (`delete 7`). A reflexive "yes" is
  precisely the failure that hurts at scale, and it is the one thing this blocks.
- Anything that is not a confirmation cancels and is handled as a normal
  question — the user is never trapped in a prompt.
- A deletion request with **no selector is refused**, not treated as "all". The
  model must pass the text it matched on, this conversation, or an explicit
  `all_reports` — and an unfiltered delete is relabelled "ALL of your saved
  reports" in the prompt, whatever the model called it.
- Proposals expire after 5 minutes.
- The match set is always itemised *before* confirmation.
- Deletes are **soft**, restorable for 30 days via `/undo`, and audit-logged.
- Reports belonging to other users are refused and reported as skipped, rather
  than silently dropped from the count.

### Resilience & graceful error handling

Failures are **classified**, because the right response differs and one generic
"query failed" produces retry loops that burn quota without ever succeeding:

| Failure | Response |
|---|---|
| SQL syntax error / guard rejection | hand the error text back to the model, bounded to 3 repair attempts |
| Transient 5xx, rate limit | retry with exponential backoff + jitter, then the secondary model |
| Rate limit *with* a retry hint | transient, not fatal — AI Studio's per-minute 429 says "billing" and "quota exceeded" but ends "retry in 18.5s" |
| Quota/credits exhausted (no retry hint) | **fatal — no retry.** More requests cannot succeed |
| Retired model (404) | fall back to the secondary model |
| Cost over cap | reject and ask the user to narrow |
| Permission error, lost credentials, network down | **end the turn and say so.** No rewrite fixes these, and handing them back spends the whole budget to fail again |
| Query timeout | cancel the job, then retry — an abandoned job keeps running and billing |

Cost control: every query is **dry-run first** and rejected before execution if
it would scan more than `BQ_MAX_BYTES_BILLED`, with `maximum_bytes_billed` set
on the real job as a second ceiling. A circuit breaker stops a BigQuery outage
from becoming a cost incident — it counts dry-run failures too, since the dry
run is the first call of every query and so the first thing an outage takes
down. A per-turn model-call budget caps any loop.
Result sets are row-capped before entering the prompt.

Empty results are reported as empty, never as zero — the model is explicitly
told not to present "no rows" as "$0 revenue".

The CLI cannot crash: every turn returns something sayable, with a trace id.

### Observability

One append-only JSONL event stream per turn (`var/traces/`), carrying
`trace_id` / `span_id` / `parent_span_id` — the OpenTelemetry span shape without
the dependency, so production export to Cloud Trace changes no call sites.

Per-turn metrics: model calls and errors, prompt/output tokens, tool calls, SQL
attempts, guard rejections, self-corrections, bytes billed, empty results, PII
redactions, refusals, retries, fallback-model use.

`/trace` lists recent turns; `/trace <id>` replays the full correspondence for
one turn — the assembled system prompt (with a digest, so the exact prompt is
identifiable even when the body is truncated), every tool call, the rows handed
back to the model, the SQL it wrote *and* the scope-rewritten SQL that actually
reached BigQuery, errors and retries. `/trace <id>` only returns the caller's
own turns: a trace holds the question, the rows and the answer, so an unfiltered
lookup by id would hand one executive another's analysis. Security decisions —
deletions, restores, cancellations, entitlement rewrites, redactions — are
flagged `audit=true` for longer retention.

Tracing is best-effort: if the sink fails, the turn continues and the failure is
counted. Observability must never be able to take down the chat.

---

## Framework choice and my experience with it

### What I chose

**I deliberately did not build this on an agent-orchestration framework.** The
stack I chose is:

| Layer | Choice | Role |
|---|---|---|
| Model | **`google-genai` 2.x** (Vertex AI backend) | Direct SDK access to Gemini 3.6 Flash, including `thought_signature` round-tripping and per-part tool calls |
| SQL policy | **sqlglot 30.x** | Parses every generated query to an AST, validates it, and *rewrites* it for entitlements. This is where the real framework work happens |
| Orchestration | ~120 lines in `retail_agent/agent.py` | plan → guard → execute → observe → answer, written out explicitly |
| Front end | **Rich** (CLI, required) / **Streamlit** (optional web) | Two renderers over one `dispatch.py` |
| Verification | **pytest** | 182 tests, offline, no credentials |

The one-line version: **I put the framework where the risk is — in the SQL
layer — and kept the control flow as plain code.** In a system whose hard
requirements are "never leak PII", "never over-read another division's data" and
"never delete without confirmation", the thing worth delegating to a mature
library is *SQL parsing*, not `if`/`else`.

### Why not an orchestration framework

Every requirement in this brief lives in the **seams between agent steps**, not
in the steps themselves:

| Seam | Requirement |
|---|---|
| Between the model asking for SQL and BigQuery running it | validate, reject, rewrite for entitlements (Req 2) |
| Between a failure and the next model call | classify it, and decide whether *any* rewrite could help (Req 5) |
| Between a deletion request and a deletion | return a proposal, never an action (Req 3) |
| Around the whole turn | budget model calls and bytes billed so a repair loop cannot become a cost incident (Req 5) |
| Across all of it | emit the full correspondence, PII-scrubbed on the way in (Req 7) |

A framework gives you the steps for free and asks you to express the seams as
hooks, callbacks or graph edges. Here the seams *are* the assignment, so
expressing them as configuration would have been paying a dependency to obscure
the part being assessed.

**Google ADK** — the closest fit, and the one I actually installed and worked
through (`google-adk` 2.9.1, clean on Python 3.14). `before_tool_callback` is
genuinely the right place for the SQL guard, its tool-confirmation hook maps
onto Requirement 3, and its eval harness would have covered a chunk of
Requirement 6 that is design-only here. What decided it against: Agent Engine
deployment and its session model are a larger commitment than a prototype needs,
and the callback signatures would have become the thing a reviewer has to learn
before they can check whether the entitlement rewrite is correct.

**LangGraph** — the strongest alternative, and worth arguing with properly
rather than dismissing:

- *Where it genuinely wins.* `interrupt()` plus a checkpointer is a better
  answer to Requirement 3 than my in-memory broker: the confirmation would
  survive a process restart, and `ConfirmationBroker` would mostly disappear.
  Its checkpointer would also replace the naive history trimming here, and
  time-travel replay is a real Requirement 7 asset.
- *Why I still did not use it.* The provider abstraction is the problem. Gemini
  3.x returns an opaque `thought_signature` on each function-call part and
  rejects history where it has gone missing — a bug that only surfaces on the
  *second* model call of a turn, after BigQuery has already been queried and
  billed (see `docs/architecture.md` §3.5). Debugging that through
  `langchain-google-genai`'s message normalisation, rather than at the raw part
  level, is materially harder, and provider-specific fields are exactly what a
  cross-provider abstraction is designed to flatten away.
- Tracing also matters here: LangGraph's ergonomic default is LangSmith, which
  would send prompts and query results — the PII surface this brief is about —
  to a third party outside the GCP perimeter. Routing it elsewhere is possible,
  but the path of least resistance points the wrong way for this system.
- And the shape is wrong. This is one analyst agent with a bounded tool loop,
  not a multi-actor workflow. Modelled as a graph it is a single node with a
  conditional edge back to itself — a graph with no graph in it.

**CrewAI / AutoGen** — role-based multi-agent orchestration solving a
coordination problem this system does not have. **LlamaIndex** is retrieval-first;
it is a serious candidate for the Golden Bucket (Req 1, design-only here), not
for the prototype's control flow.

### The trade-off, stated honestly

Not using a framework cost me: durable session persistence, a built-in eval
harness, streaming/token-level UX, and one-command managed deployment. Those are
hand-rolled, design-only or absent here, and at production scale several of them
stop being optional.

The mitigation is layering. `safety/`, `reports.py`, `confirmation.py` and
`observability.py` import nothing from the orchestration layer — verifiably, by
grep — so adopting ADK or LangGraph later replaces `agent.py`'s ~120-line loop
and nothing else. The safety guarantees are not entangled with the control flow.

The same layering is what makes the system extendable, which the brief asks for
directly: a new capability is a `TOOL_DECLARATIONS` entry plus a branch in the
loop — about a dozen lines in two places. `render_chart`, `send_report` and web
search all fit that shape, and `send_report` routes through the *existing*
confirmation broker because sending is as irreversible as deleting.

### My level of experience

I am comfortable working at this level of the stack: building tool-calling loops
directly against a provider SDK, managing conversation state and function-call
round trips by hand, and handling the provider-specific details — thought
signatures, parallel tool calls in a single content, token accounting, error
taxonomies — that a higher-level abstraction usually hides. That experience is
why I was willing to skip a framework here rather than because I had not
considered one: the two live-only failures documented in
`docs/architecture.md` §3.5 are the kind of thing you learn to expect, and to
instrument for, only after hitting them.

I am also familiar enough with the framework landscape to make this a decision
rather than a default — I evaluated ADK hands-on for this assignment, and my
LangGraph assessment above is about its concrete mechanics (checkpointers,
`interrupt()`, the provider abstraction, LangSmith defaults) rather than its
marketing. Had the brief centred on multi-agent coordination or durable
long-running workflows, I would have chosen LangGraph and said so.

---

## Testing

```bash
pytest                      # 182 tests, no credentials required
pytest tests/test_safety.py -v
pytest tests/test_streamlit_ui.py -v    # skipped unless the web UI is installed
```

External services are faked, but where the real service raises a specific
exception type the fakes raise that same real type
(`google.api_core.exceptions`, `google.genai.errors`) — classification logic
tested only against lookalikes tends to be wrong against the real thing.

---

## Layout

```
retail_agent/
  agent.py            orchestration loop, tools, self-correction
  cli.py              REPL renderer (Rich)
  dispatch.py         message routing shared by every front end:
                      confirmation first, then slash commands, then the model
  llm.py              Gemini provider, fallback, budgets, offline stub
  bigquery_runner.py  cost gate, retries, circuit breaker, classification
  reports.py          saved reports: ownership, soft delete, search
  confirmation.py     propose → confirm → execute
  observability.py    spans, metrics, audit, replay
  prompt.py           layered prompt; safety contract always last
  catalog.py          schema + metric glossary for the prompt
  safety/
    sql_guard.py      validation + entitlement rewriting
    pii.py            PII registry, scrubbing, pseudonymisation
    scope.py          entitlements loader
config/
  entitlements.yaml   who may analyse which products
  persona.yaml        tone — editable by non-developers, hot-reloaded
  metrics.yaml        metric definitions (revenue, churn, underspending…)
streamlit_app.py      optional web UI; renders what dispatch.py reports
```

---

## Dataset notes worth knowing

- **The data contains future-dated rows.** `MAX(order_items.created_at)` was
  `2026-09-21` on 2026-09-18. Un-clamped, "revenue this month" includes sales
  that have not happened and every trend line dives at the right edge. All time
  filters are clamped to `CURRENT_TIMESTAMP()`.
- **A quarter of `order_items` is `Cancelled` or `Returned`** (44,769 of
  180,342). Revenue excludes them by definition in `config/metrics.yaml`;
  including them would overstate revenue by ~25%.
- **There is no inventory table** in the four required tables, so inventory
  questions cannot be answered. The agent says so rather than improvising.
- Terms like churn and "underspending" have no definition in the data. They are
  defined in `config/metrics.yaml`, and the agent must state the definition it
  used.

---

## Known gaps

- The Golden Bucket, the learning loop and the QA harness are **design only** —
  see [`docs/architecture.md`](docs/architecture.md).
- Entitlements are enforced in-process, not by BigQuery authorized views.
- There is no real authentication; `--user` selects a persona.
- Conversation history is in memory only and is dropped, not summarised, when it
  grows past 12 turns.
- `min_group_size` (k-anonymity) is specified in `config/metrics.yaml` and
  instructed in the prompt, but not yet enforced in the SQL guard.
