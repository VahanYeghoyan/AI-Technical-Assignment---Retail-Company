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
| Safety & PII masking | Implemented, 39 tests |
| High-stakes oversight (destructive ops) | Implemented, 27 tests |
| Resilience & error handling | Implemented, 24 tests |
| Observability | Implemented, replayable traces |
| Hybrid intelligence (Golden Bucket) | Design only — [architecture](docs/architecture.md#1-hybrid-intelligence) |
| Learning loop | Design only |
| Quality assurance | Design only |
| Agility (persona) | Implemented (hot-reload) + design |

**121 tests pass with no credentials and no LLM quota.** That is deliberate: the
safety and oversight guarantees are the ones that must be verifiable in CI on any
machine, without reaching a third party.

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

### CLI commands

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
   is explicitly still allowed.

PII columns (verified against the live schema): `first_name`, `last_name`,
`email`, `street_address`, `postal_code`, `latitude`, `longitude`, and
`user_geom` — a `GEOGRAPHY` column that pinpoints a home and is easy to miss.

Then three more layers: result sets are scrubbed before they reach the model,
the final answer is scrubbed before display, and trace files are scrubbed on the
way in — a debug log should not quietly become the one place customer emails are
retained. "Top customers" works, but customers appear as salted pseudonyms
(`CUST-A83F21`), stable within a deployment so follow-up questions still resolve.

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
| Transient 5xx, rate limit | retry with exponential backoff + jitter |
| Quota/credits exhausted | **fatal — no retry.** More requests cannot succeed |
| Retired model (404) | fall back to the secondary model |
| Cost over cap | reject and ask the user to narrow |
| Permission error | fail fast, surface to the operator |

Cost control: every query is **dry-run first** and rejected before execution if
it would scan more than `BQ_MAX_BYTES_BILLED`, with `maximum_bytes_billed` set
on the real job as a second ceiling. A circuit breaker stops a BigQuery outage
from becoming a cost incident, and a per-turn model-call budget caps any loop.
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
one turn — every prompt, tool call, generated SQL, error and retry. Security
decisions are additionally flagged `audit=true` for longer retention.

Tracing is best-effort: if the sink fails, the turn continues and the failure is
counted. Observability must never be able to take down the chat.

---

## Framework choice — please read

**This does not use an agent framework.** The orchestration loop in
`retail_agent/agent.py` is written directly against the `google-genai` SDK.

I evaluated Google ADK (`google-adk` 2.9.1 installs cleanly on 3.14 and has
callbacks, a tool-confirmation hook and built-in eval). I chose not to use it
because every requirement in this brief lives in the *seams* between agent
steps — validate and rewrite SQL between the model asking and BigQuery running;
classify a failure and decide whether the model may retry; return a proposal
instead of performing a deletion; count tokens against a budget. Those seams are
the assignment. Expressing them as explicit code is clearer to review and to
defend than configuring a framework's hooks to intercept the same points, and it
keeps the agent testable offline with a stub provider.

The trade-off is real and worth stating: ADK would have supplied session
persistence, an eval harness and Vertex AI Agent Engine deployment that are
hand-rolled or design-only here.

> **Note for whoever submits this:** the brief asks you to state *your* level of
> experience with the framework. That sentence has to be written by you — I have
> deliberately not invented one. If you would rather submit an ADK-based version,
> the safety, reports, confirmation and observability modules are framework-
> independent and would port largely unchanged.

---

## Testing

```bash
pytest                      # 111 tests, no credentials required
pytest tests/test_safety.py -v
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
