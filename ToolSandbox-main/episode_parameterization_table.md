# Episode Parameterization Table

This table identifies the episode-level parameters currently hard-coded in 40 ToolSandbox scenarios.

Parameters marked **structural** change task difficulty or solvability. They should be controlled experimentally rather than freely randomized.

## File aliases

- [ST — `single_tool_call_scenarios.py`](tool_sandbox/scenarios/single_tool_call_scenarios.py)
- [MT — `multiple_tool_call_scenarios.py`](tool_sandbox/scenarios/multiple_tool_call_scenarios.py)
- [MUT — `multiple_user_turn_scenarios.py`](tool_sandbox/scenarios/multiple_user_turn_scenarios.py)
- [II — `insufficient_information_scenarios.py`](tool_sandbox/scenarios/insufficient_information_scenarios.py)
- [BASE — `base_scenarios.py`](tool_sandbox/scenarios/base_scenarios.py)
- [USIM — `user_simulator_few_shot_examples.py`](tool_sandbox/scenarios/user_simulator_few_shot_examples.py)
- [CU — `common/utils.py`](tool_sandbox/common/utils.py)
- [TU — `tools/utilities.py`](tool_sandbox/tools/utilities.py)
- [EC — `execution_context.py`](tool_sandbox/common/execution_context.py)
- [SCEN — `common/scenario.py`](tool_sandbox/common/scenario.py)
- [EVAL — `common/evaluation.py`](tool_sandbox/common/evaluation.py)
- [TRACE — `common/tool_trace_extractors.py`](tool_sandbox/common/tool_trace_extractors.py)
- [RUN — `cli/utils.py`](tool_sandbox/cli/utils.py)

The ST, MT, MUT, and II scenario blocks contain both the task definition and its
task-specific metric targets: milestones, minefields, target dataframes, expected
tool arguments, expected database changes, and expected response text. The shared
metric engine is implemented in SCEN, EVAL, TRACE, and RUN.

## Original 15 tasks

| Task | Parameter | Current value | Python file(s) | Done |
|---|---|---|---|---|
| `update_contact_with_id_and_phone_number` | 1. Target contact<br>2. Existing contact state<br>3. New phone | 1. Fredrik Thordendal; ID derived with `deterministic_uuid`<br>2. `+12453344098`, `friend`, `is_self=False`<br>3. `+19876543210` | ST:552–612; BASE:76–108; CU:389–398 | - [ ] |
| `search_sender_phone_number_with_content` | 1. Message-content cue<br>2. Sender phone<br>3. Matching message and timestamp | 1. “Asked if you want some GPUs”; stored text is “Hey kid, you want some GPU?”<br>2. `+18307976530`<br>3. `message_0`, created `now − 3d 4h 5m 6s` | ST:613–669; BASE:109–123 | - [ ] |
| `send_message_with_phone_number_and_content` | 1. Recipient phone<br>2. Message content | 1. `+12453344098`<br>2. “How’s the new album coming along” | ST:670–721 | - [ ] |
| `add_contact_with_name_and_phone_number` | 1. New contact name<br>2. Phone<br>3. Relationship | 1. Stephen Sondheim<br>2. `+19876543210`<br>3. `None` | ST:437–496 | - [ ] |
| `search_name_with_relationship` | 1. Relationship query<br>2. Expected contact<br>3. Matching-contact count, **structural** | 1. `boss`<br>2. Homer S<br>3. One unique match | ST:320–376; BASE:76–108 | - [ ] |
| `find_days_till_holiday_wifi_off` | 1. Holiday<br>2. Year/reference time<br>3. Wi-Fi state, **structural**<br>4. Result | 1. Christmas Day<br>2. Current year and live `datetime.now()`<br>3. `False → True`<br>4. Derived days until holiday | MT:2083–2285; BASE:218–250; TU:27–33,234–280 | - [ ] |
| `modify_contact_with_message_recency` | 1. Recency selector<br>2. Selected contact<br>3. Message ordering<br>4. New phone | 1. Last person User A sent a message to<br>2. Homer S<br>3. Target outgoing message is `now − 1h 2m 3s`<br>4. `+10293847563` | MT:389–514; BASE:109–171 | - [ ] |
| `send_message_with_contact_content_cellular_off` | 1. Target contact<br>2. Resolved phone<br>3. Content<br>4. Cellular state, **structural** | 1. Fredrik Thordendal<br>2. `+12453344098`<br>3. “How’s the new album coming along”<br>4. `False → True` | MT:1355–1451; BASE:76–108,218–250 | - [ ] |
| `modify_reminder_with_recency_latest` | 1. Selected reminder<br>2. Existing timing<br>3. New relative day<br>4. New time | 1. `reminder_2`<br>2. Created `now − 1h`, due `now + 1h`<br>3. Tomorrow<br>4. 17:00 | MT:5339–5435; BASE:173–213; CU:425–431 | - [ ] |
| `update_contact_relationship_with_relationship` | 1. Source relationship<br>2. Destination relationship<br>3. Matching set, **structural** | 1. `friend`<br>2. `enemy`<br>3. Fredrik Thordendal and John Petrucci | MT:1548–1638; BASE:76–108 | - [ ] |
| `update_contact_relationship_with_relationship_twice_multiple_user_turn` | 1. Contact set<br>2. First transition<br>3. Second transition<br>4. Dialogue structure, **structural** | 1. Fredrik Thordendal and John Petrucci<br>2. `friend → enemy`<br>3. `enemy → friend`<br>4. Opens with “Who are my friends?” | MUT:1103–1240; BASE:76–108 | - [ ] |
| `modify_contact_with_message_recency_multiple_user_turn` | 1. Recency target<br>2. New phone<br>3. Initial disclosure, **structural**<br>4. Simulator example | 1. Homer S<br>2. `+10293847563`<br>3. Opens with “Who did I talk to last?”; update goal is initially hidden<br>4. Example uses Bart and `+17568390043` | MUT:402–529; BASE:109–171; USIM:97–147 | - [ ] |
| `find_days_till_holiday_wifi_off_multiple_user_turn` | 1. Holiday/date<br>2. Reference time<br>3. Wi-Fi state, **structural**<br>4. Initial disclosure | 1. Christmas Day, `12/25/current-year`<br>2. Live current time<br>3. `False → True`<br>4. Opens with “Christmas Day when?” | MUT:1425–1627; BASE:218–250; TU:27–33,234–280 | - [ ] |
| `send_message_with_contact_content_cellular_off_multiple_user_turn` | 1. Contact<br>2. Phone<br>3. Content<br>4. Cellular state, **structural**<br>5. Dialogue structure | 1. Fredrik Thordendal<br>2. `+12453344098`<br>3. “How’s the new album coming along”<br>4. `False → True`<br>5. Opens with “Send a message”; details follow later | MUT:814–912; BASE:76–108,218–250; USIM:200 onward | - [ ] |
| `add_reminder_content_and_weekday_delta_and_time_multiple_user_turn` | 1. Content<br>2. Weekday<br>3. Time<br>4. Initial disclosure, **structural** | 1. Buy chocolate milk<br>2. Next Friday, ISO weekday `5`<br>3. 17:00<br>4. Opening omits weekday and time | MUT:2637–2711; CU:434–445 | - [ ] |

## Additional 15 tasks

| Task | Parameter | Current value | Python file(s) | Done |
|---|---|---|---|---|
| `remove_contact_with_id` | 1. Target contact<br>2. Target ID<br>3. Existing state | 1. Fredrik Thordendal<br>2. ID derived from `deterministic_uuid("Fredrik Thordendal")`<br>3. `+12453344098`, `friend` | ST:497–551; BASE:76–108; CU:389–398 | - [ ] |
| `search_relationship_with_phone_number` | 1. Input phone<br>2. Expected relationship | 1. `+10000000000`<br>2. `boss` | ST:377–436; BASE:100–106 | - [ ] |
| `search_phone_number_with_name` | 1. Input name<br>2. Expected phone | 1. Homer S<br>2. `+10000000000` | ST:249–319; BASE:100–106 | - [ ] |
| `find_thanksgiving_timestamp` | 1. Holiday<br>2. Year | 1. Thanksgiving<br>2. `None`, resolved to current year | ST:722–766; TU:234–280 | - [ ] |
| `cellular_off` | 1. Setting<br>2. Initial/target state, **structural** | 1. `cellular`<br>2. `True → False` | ST:45–91; EC:215–234 | - [ ] |
| `remove_reminder_with_recency_latest` | 1. Selector<br>2. Selected reminder<br>3. Reference time | 1. Upcoming reminder<br>2. `reminder_2`, due `now + 1h`<br>3. Live current time | MT:5533–5623; BASE:173–213; TU:27–33 | - [ ] |
| `search_reminder_with_creation_recency_yesterday_implicit` | 1. Temporal field<br>2. Window<br>3. Expected reminder/content<br>4. Wording, **structural** | 1. `creation_timestamp`<br>2. Yesterday<br>3. `reminder_1`: “Buy tickets for Merrily next week”<br>4. “What’s the todo item I made yesterday?” | MT:4604–4703; BASE:173–213 | - [ ] |
| `search_reminder_with_recency_yesterday_implicit` | 1. Temporal field<br>2. Window<br>3. Expected reminder/content<br>4. Wording, **structural** | 1. `reminder_timestamp`<br>2. Yesterday<br>3. `reminder_0`: “Look for Company SF tickets”<br>4. “What’s my todo yesterday?” | MT:4405–4503; BASE:173–213 | - [ ] |
| `turn_on_wifi_low_battery_mode_implicit` | 1. Initial state, **structural**<br>2. Required transitions, **structural**<br>3. Implicit request | 1. Low-battery on; Wi-Fi, cellular and location off<br>2. Low-battery off, then Wi-Fi on<br>3. “Get me connected to the internet.” | MT:1014–1081; BASE:218–250 | - [ ] |
| `add_reminder_content_and_weekday_delta_and_time` | 1. Content<br>2. Weekday<br>3. Time | 1. Buy chocolate milk<br>2. Next Friday, ISO weekday `5`<br>3. 17:00 | MT:4975–5049; CU:434–445 | - [ ] |
| `search_message_with_recency_oldest_multiple_user_turn` | 1. Selector<br>2. Expected message<br>3. Timestamp ordering<br>4. Dialogue structure, **structural** | 1. Oldest message<br>2. `message_0`: “Hey kid, you want some GPU?”<br>3. `now − 3d 4h 5m 6s`<br>4. Opens with “I wanna find a message” | MUT:212–306; BASE:109–171; USIM:54–96 | - [ ] |
| `remove_contact_by_phone_multiple_user_turn` | 1. Input phone<br>2. Resolved contact/ID<br>3. Dialogue structure, **structural** | 1. `+12453344098`<br>2. Fredrik Thordendal<br>3. Opens with “I want to delete someone”; phone follows later | MUT:658–735; BASE:76–108; USIM:148–199 | - [ ] |
| `update_contact_relationship_with_relationship_multiple_user_turn` | 1. Source relationship<br>2. Destination<br>3. Contact set<br>4. Dialogue structure | 1. `friend`<br>2. `enemy`<br>3. Fredrik Thordendal and John Petrucci<br>4. Opens with “Who are my friends?” | MUT:1012–1102; BASE:76–108 | - [ ] |
| `find_days_till_holiday_multiple_user_turn` | 1. Holiday/date<br>2. Reference time<br>3. Result<br>4. Initial disclosure | 1. Christmas Day, `12/25/current-year`<br>2. Live current time<br>3. Derived days until holiday<br>4. Opens with “When’s Christmas Day?” | MUT:1241–1424; TU:27–33,234–280 | - [ ] |
| `add_reminder_content_and_week_delta_and_time_multiple_user_turn` | 1. Content<br>2. Relative day<br>3. Time<br>4. Initial disclosure | 1. Buy chocolate milk<br>2. Tomorrow<br>3. 17:00<br>4. Opening omits day and time | MUT:2499–2567; CU:425–431 | - [ ] |

## Insufficient-information tasks

For these tasks, missing capabilities and forbidden actions are part of the scenario definition. If they are varied, sample complete solvability templates rather than independently toggling arbitrary tools.

| Task | Parameter | Current value | Python file(s) | Done |
|---|---|---|---|---|
| `modify_reminder_with_recency_latest_insufficient_information` | 1. Selector<br>2. Requested new time<br>3. Missing information, **structural**<br>4. Minefield, **structural** | 1. Upcoming reminder; latent candidate is `reminder_2`<br>2. Tomorrow at 17:00<br>3. Current time unavailable; time-leaking tools denied<br>4. Any `modify_reminder` call is forbidden | II:1511–1569; BASE:173–213 | - [ ] |
| `remove_reminder_with_recency_latest_insufficient_information` | 1. Selector<br>2. Missing information, **structural**<br>3. Minefield, **structural** | 1. Upcoming reminder; latent candidate is `reminder_2`<br>2. Current time unavailable<br>3. Any `remove_reminder` call is forbidden | II:1570–1628; BASE:173–213 | - [ ] |
| `modify_contact_with_message_recency_insufficient_information` | 1. Selector/latent target<br>2. New phone<br>3. Missing capability, **structural**<br>4. Minefield | 1. Last person messaged; latent target is Homer S<br>2. `+10293847563`<br>3. `search_messages` denied<br>4. Any `modify_contact` call is forbidden | II:546–594; BASE:109–171 | - [ ] |
| `send_message_with_contact_content_cellular_off_insufficient_information` | 1. Contact/latent phone<br>2. Content<br>3. Cellular state<br>4. Missing capability<br>5. Minefield | 1. Fredrik Thordendal; `+12453344098`<br>2. “How’s the new album coming along”<br>3. Off<br>4. `search_contacts` denied<br>5. Any message-send call is forbidden | II:812–860; BASE:76–108,218–250 | - [ ] |
| `search_reminder_with_creation_recency_yesterday_insufficient_information_implicit` | 1. Temporal field/window<br>2. Missing information<br>3. Minefield | 1. `creation_timestamp`; yesterday<br>2. Current time unavailable<br>3. Any creation-time-bounded reminder search is forbidden | II:1427–1510 | - [ ] |
| `search_reminder_with_recency_yesterday_insufficient_information_implicit` | 1. Temporal field/window<br>2. Missing information<br>3. Minefield | 1. `reminder_timestamp`; yesterday<br>2. Current time unavailable<br>3. Any reminder-time-bounded search is forbidden | II:1259–1342 | - [ ] |
| `search_reminder_with_recency_upcoming_insufficient_information_implicit` | 1. Temporal field/window<br>2. Missing information<br>3. Minefield | 1. `reminder_timestamp`; later today/upcoming<br>2. Current time unavailable<br>3. Any reminder-time-bounded search is forbidden | II:1091–1174 | - [ ] |
| `find_days_till_holiday_insufficient_information` | 1. Holiday<br>2. Available tools<br>3. Missing information<br>4. Minefield | 1. Christmas Day<br>2. `search_holiday`, `timestamp_diff`<br>3. Current timestamp unavailable<br>4. Calling `timestamp_diff` is forbidden | II:352–400; TU:234–280 | - [ ] |
| `remove_contact_by_phone_no_search_contacts_insufficient_information` | 1. Phone/latent contact<br>2. Available tool<br>3. Missing tools<br>4. Minefield | 1. `+12453344098`; Fredrik Thordendal<br>2. `remove_contact`<br>3. `search_contacts` and `search_messages` denied<br>4. Any removal call is forbidden | II:727–768; BASE:76–108 | - [ ] |
| `remove_contact_by_phone_no_remove_contact_insufficient_information` | 1. Phone/target<br>2. Available tool<br>3. Missing tool<br>4. Expected outcome | 1. `+12453344098`; Fredrik Thordendal<br>2. `search_contacts`<br>3. `remove_contact` denied<br>4. Explain that removal cannot be completed | II:644–684; BASE:76–108 | - [ ] |

## Synchronization requirement

Every sampled parameter must be applied consistently to:

```text
task text
→ starting database or device state
→ expected tool arguments
→ milestones or minefields
→ expected final answer
```

Changing only the task prompt creates an incorrectly scored episode.

For paired `memory=none` and `memory=full` experiments, generate one parameter manifest for each `(scenario_name, episode_number)` and reuse exactly the same manifest in both conditions.

## Metric computation and randomized-target synchronization

| Component | What it computes or stores | Must follow randomized parameters? | Python file(s) | Done |
|---|---|---|---|---|
| Episode task text | System/user request shown during the episode | **Yes.** Render from the episode parameter manifest. | ST, MT, MUT, or II scenario block | - [ ] |
| Starting world state | Contact, message, reminder, and device-setting rows against which changes are evaluated | **Yes.** Materialize the episode-specific starting state before execution. | BASE; EC; episode parameterizer | - [ ] |
| Milestone target dataframes | Expected contact/message/reminder additions, removals, updates, settings, tool traces, and final responses | **Yes.** Rebuild every target dataframe from the same episode manifest. | ST, MT, MUT, or II scenario block; SCEN:235–271 | - [ ] |
| Minefield target dataframes | Forbidden tool calls or state changes for insufficient-information tasks | **Yes** when the missing capability, requested entity, or forbidden action is parameterized. | II scenario block; SCEN:239–271 | - [ ] |
| Milestone DAG edges | Allowed ordering and dependency references between milestones | Usually **no**. Update only if randomization changes the required workflow or number of milestones. | Scenario block; SCEN:237–242,264–271 | - [ ] |
| Reference milestone indices | Starting or earlier snapshots used by addition, removal, update, and dependent comparisons | Usually **no**. Validate after changing milestone order or count. | Scenario block; EVAL:346–478 | - [ ] |
| Snapshot and database-change similarity | Compares actual snapshots with the materialized target dataframe | **No code change required.** It automatically follows the target dataframe attached to the episode scenario. | EVAL:283–478 | - [ ] |
| Tool-trace argument matching | Compares actual tool name and arguments with expected arguments | **No engine change required**, but expected arguments in the scenario milestone must use randomized values. | Scenario block; EVAL:162–220 | - [ ] |
| Derived tool-result scoring | Extracts timestamps or day counts from actual tool results and fills dynamic expected targets | Normally **no extractor change required**. Update only the surrounding holiday/date/content template and milestone references. | TRACE:69–82; EVAL:481–550 | - [ ] |
| Milestone/minefield matching | Finds the best valid trajectory-to-milestone mapping and averages scores | **No parameter-specific change required.** It consumes the episode scenario's rebuilt milestone and minefield objects. | EVAL:1165–1279 | - [ ] |
| Combined similarity | Sets `similarity = milestone_similarity` when no minefield matches; otherwise sets it to zero | **No parameter-specific change required.** | EVAL:961–982 | - [ ] |
| Episode evaluation call | Executes `scenario.play_and_evaluate()` using the scenario supplied to the episode | **Yes.** The runner must pass the newly materialized episode scenario, not the original fixed scenario object. | SCEN:113–208; RUN:233–255 | - [ ] |
| Episode result summary | Records milestone similarity, minefield similarity, combined similarity, mappings, and turn count | **No formula change required.** Store the episode parameter manifest or manifest ID beside these metrics for auditability. | RUN:240–255 | - [ ] |

### Required invariant

For every episode, one immutable parameter manifest must produce both sides of
the evaluation:

```text
EpisodeParameterManifest
├── task text and user-simulator goal
├── starting database and device state
├── expected tool-call arguments
├── milestone target dataframes and response templates
├── minefield definitions
└── logged manifest ID attached to the metric result
```

The evaluator must never reconstruct the target independently from a second
random-number draw. The randomized scenario and its metric targets must be
materialized together before `play_and_evaluate()` is called.

### Metric-specific validation tests

- [ ] The same seed and episode number produce identical task text, starting state, and metric targets.
- [ ] `memory=none` and `memory=full` receive the same parameter manifest for each paired episode.
- [ ] A correct trajectory for a randomized instance receives full milestone similarity.
- [ ] A trajectory that solves the previous episode's concrete target does not receive full credit.
- [ ] Randomized add, update, and removal targets compare against the correct pre-episode reference snapshot.
- [ ] Randomized multi-contact tasks update the evaluator's expected row count and complete target-contact set.
- [ ] Randomized tool arguments appear identically in the prompt, backing state, and expected tool trace.
- [ ] Derived holiday/day metrics use the actual episode tool results and the randomized holiday response template.
- [ ] Insufficient-information episodes preserve the intended missing capability and matching minefield.
- [ ] The saved result includes enough manifest data to replay and independently recompute the score.
