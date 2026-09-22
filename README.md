# Retail Data Analysis Chat Assistant

A CLI chat agent that lets non-technical executives ask questions about retail
sales data in plain language, discuss the answers, and save reports with action
items. It queries the public `bigquery-public-data.thelook_ecommerce` dataset.

The design brief is in [`docs/architecture.md`](docs/architecture.md) — that
document is the main deliverable and covers the requirements this prototype
implements in design only.

---

## What actually runs

| Requirement | State | Tests |
|---|---|---|
| Safety & PII masking | Implemented | 78 in `tests/test_safety.py` |
| High-stakes oversight (destructive ops) | Implemented | 22 in `tests/test_reports.py` |
| Resilience & error handling | Implemented | 45 in `tests/test_resilience.py` |
| Observability | Implemented, replayable traces | 7 in `tests/test_reports.py` |
| Hybrid intelligence (Golden Bucket) | Design only — [architecture](docs/architecture.md#31-hybrid-intelligence--the-golden-bucket) | — |
| Learning loop | Design only | — |
| Quality assurance | Design only | — |
| Agility (persona) | Implemented (hot-reload) + design | in `tests/test_agent.py` |

`tests/test_agent.py` (44) exercises all of them end to end through the agent
loop and the CLI; `tests/test_demo.py` (9) covers the offline demo and
`tests/test_streamlit_ui.py` (8) the optional web UI.

**213 tests pass with no credentials and no LLM quota** — 205 of them without the
optional web UI installed, whose tests skip when streamlit is absent. That is
deliberate: the safety and oversight guarantees are the ones that must be
verifiable in CI on any machine, without reaching a third party.

Separately, the full path has been verified live against Vertex AI Gemini 3.6
Flash and BigQuery — including entitlement rewriting on the real warehouse
(a Women's-scoped user asking explicitly for Men's revenue gets an empty result,
not an error and not data) — most recently on 2026-09-22.

---

## Setup

### 1. Python

Python **3.11 or newer** — verified from a clean install on 3.11, 3.12, 3.13 and
3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Google Cloud credentials

The Python clients need Application Default Credentials — note this is a
*different* credential from the one `gcloud auth login` writes, and `bq`
working on the command line does **not** mean the client library will work.

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

That project is where BigQuery jobs run and, with Vertex AI, where the model is
called — nothing else needs setting. To use a different project without
changing your gcloud config, set `GOOGLE_CLOUD_PROJECT` in `.env`.

Any project with the BigQuery API enabled works; the free BigQuery sandbox is
sufficient for the data, and queries against the public dataset are billed to
your project (~1 TB/month free). Every query here is dry-run and capped well
below that.

### 3. Model access

```bash
cp .env.example .env
```

Two options. **Vertex AI is recommended and needs no API key** — it reuses the
Application Default Credentials you just set up, so anything that can read the
warehouse can also reach the model, and access is IAM rather than a bearer
secret that has to be minted, stored and rotated. It needs a project with
billing enabled:

```bash
gcloud services enable aiplatform.googleapis.com
# .env: LLM_PROVIDER=vertex   (the default in .env.example)
```

Or an [AI Studio](https://aistudio.google.com/apikey) key
(`LLM_PROVIDER=gemini`, `GOOGLE_API_KEY=…`) — the option to use if your project
is a BigQuery **sandbox**, which has no billing account and so no Vertex AI. Be
aware its free tier is small and returns `429 Your prepayment credits are
depleted` on *every* model once spent — which is what happened during this
project's development, and is why the offline demo below exists.

### 4. Run

```bash
python -m retail_agent.cli --user maya
python -m retail_agent.cli --list-users    # see the available personas
```

If a first question fails, the answer names the problem and every answer
carries a trace id — `/trace <id>` shows the exact error. The usual causes:

| Symptom | Fix |
|---|---|
| "the data warehouse … credentials are missing or were refused" | Application Default Credentials are missing, or the project cannot run BigQuery jobs: `gcloud auth application-default login`, then `gcloud config set project …` |
| "The language model is misconfigured" (Vertex) | `gcloud services enable aiplatform.googleapis.com`, and check the project has billing — or switch to `LLM_PROVIDER=gemini` |
| "The language model quota … is exhausted" | The AI Studio free tier is spent: switch to Vertex, or run the offline demo below |

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

### Offline demo — no model access needed

```bash
LLM_PROVIDER=stub python -m retail_agent.cli --user maya
```

The model is replaced by a keyword script (`retail_agent/demo.py`), but
everything it sets in motion is the real system, so the four prototype
requirements can be checked on a machine with no model access at all:

| Type | What you see |
|---|---|
| `what data is available?` | the schema catalog, no database needed |
| `show me our customers' email addresses` | the SQL guard refusing the query before it reaches BigQuery |
| `top brands by revenue this year` | the scope-rewritten query: real rows with BigQuery credentials, a graceful stop without them |
| `create a report on denim`, then `/reports` | the report library |
| `delete the reports from this conversation`, then `yes`, then `/undo` | the confirmation flow and the soft delete |
| `/trace` | the event stream behind each of those turns |

It is not an analyst — every data question gets the same query. It exists
because an agent that can only be demonstrated when a third party is up fails
its own resilience requirement, and because this project's API key hit
`429 Your prepayment credits are depleted` during development.

---

## Example session

Real output from 2026-09-22, signed in as `maya` (VP, Women's Division) against
live Vertex Gemini 3.6 Flash and live BigQuery, from a clean checkout set up
exactly as above — `cp .env.example .env`, no project in `.env`:

```
› What were my top 3 brands by revenue this year?

Your top three brands by revenue in the Women's Division for 2026 year-to-date
(1 January 2026 through 22 September 2026) are Jones New York, Calvin Klein,
and 7 For All Mankind.

Top 3 Brands Performance (2026 YTD)

 Brand                 Revenue  Gross Margin  Units Sold  Avg Price / Unit
 ─────────────────────────────────────────────────────────────────────────
 Jones New York     $19,646.90    $10,736.00         201            $97.75
 Calvin Klein       $19,388.42    $10,256.54         263            $73.72
 7 For All Mankind  $15,107.60     $7,183.36          99           $152.60

Key Takeaways

 • Jones New York leads overall performance with a strong balance of volume
   and margin (54.6% margin rate).
 • Calvin Klein drove the highest volume at 263 units, but sits in second
   place due to a lower average price per unit ($73.72 vs. $97.75).
 • 7 For All Mankind achieved third place on less than half the volume of
   Calvin Klein, leveraging premium price points ($152.60 average per unit).

Metric definition: Revenue reflects SUM(sale_price) across non-cancelled and
non-returned order items for the Women's Division. Data through 22 September
2026.

[status=ok llm=2 sql=1 corrections=0 tok=5424->605 bytes=8,674,709 trace=e8994d14bda540d1]
```

Note it volunteered the metric definition and the date range unprompted — those
come from `config/metrics.yaml` and `config/persona.yaml`, not from the question.
Two model calls, one query: today's date is in the prompt, so the model no longer
spends a query and a guard rejection finding out what "this year" means.

A PII request, same session:

```
› Who is the customer with email alice@example.com?

I provide aggregate sales, product, customer cohort, and performance analysis
for the Women's Division, but I cannot query or display individual personal
data such as email addresses.

Would you like to see customer spend or order performance broken down by
acquisition source, state, or age cohort instead?

[status=ok llm=1 sql=0 corrections=0 tok=3646->59 bytes=0 trace=5d4c8bb0c94d4fb3]
```

That refusal came from the model, but it is not what enforces the rule: had it
tried, the SQL guard would have rejected any query touching `email` before it
reached BigQuery.

And a report, same session — saved with every section `config/persona.yaml`
requires (Headline, What the data shows, Why it moved, Risks and unknowns,
Action items), because `save_report` refuses a body missing any of them:

```
› Create a report on my top 3 brands this year, with insights and action items for Q4

I have saved the report "Women's Division Top 3 Brands Report YTD 2026 & Q4
Outlook" to your library.

Executive Summary & Q4 Priorities

 • Jones New York ($19,646.90 Revenue | $10,736.00 Gross Margin): Leads overall
   revenue across a broad workwear assortment (Dresses, Outerwear, Blazers).
    • Action: Audit sizing on Pants & Capris to reduce its 12.00% return rate
      (9 returned out of 75 items) and protect margin.
 • Calvin Klein ($19,388.42 Revenue | $10,256.54 Gross Margin): Strongest volume
   driver (263 net units) with low return rates (6.29%).
    • Action: Expand Q4 inventory depth in Outerwear & Coats ($4,684.63 net
      revenue, 4.17% return rate) ahead of seasonal cold-weather demand.
 • 7 For All Mankind ($15,107.60 Revenue | $7,183.36 Gross Margin): Highly
   profitable premium pricing ($152.60 average selling price), heavily
   concentrated in Jeans ($11,385.80 net revenue).
    • Action: Introduce online fit-prediction tools on denim PDPs to curb
      elevated returns (12.50% in Jeans; 13.18% brand total).

Data covers 1 January 2026 through 22 September 2026 for the Women's Division.
Saved report 41ccfecd (/reports to list)

[status=ok llm=4 sql=2 corrections=0 tok=27148->2128 bytes=17,710,619 trace=21333b2b3ec7462d]
```

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
| `/help` | list these commands |
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
   itself is rejected — however the FROM clause wraps the table. A
   parenthesised join (`FROM (users AS u JOIN orders o ON …)`) parses as a
   subquery rather than a plain source, and once slipped past this rule with
   names and postcodes; the check now follows every table to the SELECT that
   owns it.

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
  precisely the failure that hurts at scale, and it is the one thing this
  blocks: a "yes" or a wrong count is answered with the count to type, and the
  proposal stays open.
- Anything else cancels and is handled as a normal question — the user is never
  trapped in a prompt.
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
redactions, refusals, retries, fallback-model use. *Refusals* are the policy
saying no — a PII, whole-row, write or out-of-dataset query, or a deletion that
named nothing to match — kept apart from guard rejections of merely malformed
SQL, because someone probing and a model fumbling are different alerts. The CLI
prints the turn's figures under every answer:

```
[status=ok llm=2 sql=1 corrections=0 tok=5424->605 bytes=8,674,709 trace=e8994d14bda540d1]
```

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
| Verification | **pytest** | 213 tests, offline, no credentials |

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
pytest                      # 213 tests, no credentials required
pytest tests/test_safety.py -v
pytest tests/test_demo.py -v            # the offline demo, end to end
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
  demo.py             the scripted model behind the offline demo
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
- Customer ids are pseudonymised by result column name (`user_id`,
  `customer_id`), which the prompt tells the model to use; a query aliasing
  `users.id` as something else returns the raw id. Raw ids are surrogate keys,
  not personal data, but in production the authorized views should expose only
  the pseudonym.
- A scoped user sees every order containing one of their products, and
  `orders.num_of_item` counts that order's other items too.
- `refusals` counts refusals the code makes; the model declining in prose is
  not counted, since detecting that needs an eval rather than a counter.
