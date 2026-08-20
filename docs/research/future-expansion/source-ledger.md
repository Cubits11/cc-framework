# Source Ledger: A Worked Week 1 Artifact

## Why this document exists

Week 1 of the [eight-week plan](eight-week-plan.md) converts a narrative source
into research objects. This is a worked example of that conversion, using an
actual source: a recorded tarot reading addressed to a general sign-based
audience, brought into a working session as motivational material alongside two
active projects.

The point is not to evaluate the source. The point is to demonstrate the gate
every narrative passes through before it is allowed near a decision. A source
that survives the gate contributes a *question*. It never contributes an
*answer*.

The reading may remain meaningful as reflection. Nothing below asks anyone to
stop finding it meaningful.

## Classification labels

| Label | Definition | Evidential weight |
| --- | --- | --- |
| `REFLECTION` | A description of the subject's present state or recent behavior. | None. May be accurate and still carry no predictive content. |
| `IDENTITY AFFIRMATION` | A statement about who the subject is. | None. |
| `FORECAST` | A statement about what will happen. | None unless independently registered and observed. |
| `CAUSAL ASSERTION` | A statement that one thing brings about another. | None. Causal claims require causal assumptions and a design. |
| `ACTION HEURISTIC` | A rule for what to do next. | None as evidence, but may be a useful, separately justified practice. |
| `EMPIRICAL CLAIM` | A statement checkable against the world. | Only after predeclaration, observation, and control. |
| `MARKETING FUNNEL` | Content whose function is engagement or conversion for the source. | None, and a reason for caution about felt accuracy. |

A statement can carry more than one label. Where it does, the strictest handling
applies.

## The ledger

| Source statement | Classification | Testable translation | Disallowed inference |
| --- | --- | --- | --- |
| "Things are just about to turn for you"; the wheel is moving in your favor. | `FORECAST` | None directly. The nearest registerable question is whether a predeclared intervention changes a predeclared measure inside a fixed window. | That improvement is inevitable, scheduled, or owed. Not usable for runway, hiring, spend, or pricing. |
| "You're going to experience more signs, symbols, and synchronicities" indicating things are turning. | `CAUSAL ASSERTION` / `ACTION HEURISTIC` | Name the traction indicators and the observation window *before* any result is seen, then count them. | That noticing a coincidence afterwards is evidence that a turn occurred. Retrospective matching is the failure mode this whole package exists to block. |
| "There's been a lot of experimentation going on... you've been testing different approaches." | `REFLECTION` | Count registered experiments started, completed, and abandoned in a fixed prior period. This is directly measurable from an experiment registry. | That the description being accurate makes the forecasts accurate. Broadly applicable descriptions feel personal ([Mason & Budge, 2011](https://pubmed.ncbi.nlm.nih.gov/21315874/)). |
| "Nobody knows what they're doing. We're all just experimenting." | `ACTION HEURISTIC` | Not a claim. Usable as morale. | That the absence of certainty excuses the absence of protocol. |
| "This is exactly why this is going to work for you" - because the approach is unconventional. | `FORECAST` + `CAUSAL ASSERTION` | None. The implied mechanism (unconventional therefore successful) is not identifiable from this source. | That unconventionality predicts success. Selection effects make surviving unconventional bets highly visible and failed ones invisible. |
| "You're inventing something here... solving a problem in a way that nobody ever thought of solving it before." | `IDENTITY AFFIRMATION` + implicit novelty claim | Run a prior-art sweep ([Prompt 3](prompt-library.md#3-prior-art-destroyer)). Novelty is established by failing to find prior art, not by feeling original. | That "novel," "first," or "nobody has done this" may appear in any public copy. Those words require a named evidentiary standard. |
| "You're noticing the thing that other people haven't noticed." | `IDENTITY AFFIRMATION` | Consequence search ([Prompt 7](prompt-library.md#7-b2b-consequence-search-protocol)): has this failure class produced a documented, consequential outcome anywhere? | That an unnoticed thing is therefore an important thing. A sparse search result is a result, and it may mean the thing is unimportant. |
| "Divine timing. Right place, right time, right idea." | `CAUSAL ASSERTION` | None. Unfalsifiable as stated. | That timing is externally arranged, and therefore that acting now is lower risk than acting later. |
| "I'm seeing a blue notebook"; someone working on a laptop in a car; "your phone screen is cracked"; a person with curly hair connected to the innovation; a conversation near a vending machine. | `EMPIRICAL CLAIM` | These are the only literally checkable statements in the source. See [The falsification exercise](#the-falsification-exercise) below - they can be tested, but only under predeclaration, base-rate comparison, and a blinded control. | That a later match confirms anything. Each item is high-prevalence, and "connected to the innovation" is elastic enough to fit almost any encounter unless bounded in advance. |
| "You get overwhelmed by data sometimes... tied up in a big tangle." | `REFLECTION` | Measure where founder hours actually go by phase. The stated bottleneck and the measured bottleneck are frequently different. | That the felt bottleneck is the real bottleneck. |
| "The right piece of information comes in at the right time and breaks the loop." | `ACTION HEURISTIC` | Identify the single cheapest measurement that would change the current decision, and run that one first. | That the information will arrive on its own, or that waiting is a strategy. |
| "There is money attached to this... revenue arriving... a sale, a commission, a royalty, or a referral." | `FORECAST` (financial) | None from this source. Conversion questions belong to an already-registered pricing or funnel experiment that exists independently of the reading. | Hard block: no revenue projection, no pricing decision, no spend commitment, no runway assumption, and no investor- or customer-facing statement may cite or rest on this. |
| "We're breaking through a ceiling... your income or role or reach is infinitely bigger than you thought." | `FORECAST` | None. | That current constraints are illusory. Constraints are measured, not dispelled. |
| "An experiment might work once... figure out why it worked, how it worked, and can I make it work again." | `ACTION HEURISTIC` (method) | Does the result reproduce on a second run, a second environment, or an independent implementation? This is the Week 6 replication gate. | That a single success is a scalable model. One-off success supports a feasibility note only ([CONSORT pilot guidance](https://www.bmj.com/content/355/bmj.i5239)). |
| "Now you've discovered something that works, you don't go messing with it... I need the discipline to not change anything anymore." | `ACTION HEURISTIC` (method) | Freeze the protocol before observation and record deviations rather than absorbing them ([OSF registration guidance](https://help.osf.io/article/330-welcome-to-registrations)). | That freezing a protocol makes the result correct. A frozen protocol prevents silent revision; it does not validate. |
| "Redirect effort away from things that haven't worked and toward the area that's demonstrating momentum." | `ACTION HEURISTIC` | Predeclare the traction indicators and the stop/scale thresholds, then reallocate when a threshold is crossed ([Prompt 5](prompt-library.md#5-signal-versus-traction-designer)). | That momentum can be judged after the fact from whatever moved. Post-hoc threshold selection converts noise into a mandate. |
| "More inquiries, more questions, more emails coming in, better numbers, repeat customers, and increasing demand." | `FORECAST` naming real indicators | This is the one place the source names measurable quantities. Adopt the indicator *names*, discard the prediction that they will rise. | That attention equals inquiries, inquiries equal conversion, conversion equals revenue, or revenue equals retention. These are separate measures and are never substituted for one another. |
| "Claim authority over it, structure it, give it order." | `ACTION HEURISTIC` | Optional and low priority: does writing the protocol down reduce run-to-run variance? | That structure substitutes for evidence. Formalizing an unvalidated result makes it durable, not true. |
| "Don't let your emotions dictate what you do." | `ACTION HEURISTIC` | Consistent with predeclaring thresholds before results are visible. | That discipline about feelings implies discipline about inference. |
| "Follow the signs... something you've been experimenting with is starting to answer you back." | `ACTION HEURISTIC` + `FORECAST` | Only in the predeclared-indicator form above. | That ambiguous events count as replies. |
| "Hit the like button... subscribe... tell me in the comments what you're experimenting with... click the link for the extended reading." | `MARKETING FUNNEL` | Not a claim about the subject. | That felt resonance is independent of the source's incentive to produce resonance. The comment prompt also collects the subject's own specifics, which raises the apparent hit rate of future readings. |
| "The most important part of any tarot reading is you." | `REFLECTION` | Not a claim. | Nothing follows from it either way - though it is, structurally, an accurate description of where the content comes from. |

## What survives

Three method heuristics survive translation:

1. **Experiment, then replicate, then formalize.** Try things; when something
   works, find out why and whether it recurs before building on it.
2. **Allocate on predeclared indicators.** Move resources toward what moves the
   indicator you named in advance, not toward what looks like momentum in
   hindsight.
3. **Freeze the protocol once it works.** Stop tinkering during measurement;
   record deviations instead of absorbing them.

Each is independently supported by ordinary research practice. None of them is
supported *by the reading*. The reading is a prompt that surfaced them, which is
exactly the permitted use: a story generated a question.

Everything else in the ledger - the turn, the timing, the ceiling, the revenue -
carries no weight and enters no decision.

## The falsification exercise

The five concrete specifics (blue notebook, laptop in a car, cracked phone
screen, curly-haired person, vending-machine conversation) are the only part of
the source that could be tested at all. Testing them is optional and cheap. If
it is done, it is done properly or not at all.

**Decision it can change.** Not a business decision. It calibrates the observer:
if retrospective matching produces a high hit rate on control material, the rule
that indicators must be predeclared gets tighter, and narrative-derived signals
stay out of indicator selection entirely.

**Protocol requirements.**

- Predeclare, in writing and before any observation, what counts as a match for
  each of the five items - including the boundary of "connected to the
  innovation," which is otherwise elastic enough to absorb any encounter.
- Fix the observation window and the scorer before starting.
- Score against base rates, not against zero. Cracked screens, blue notebooks,
  curly hair, and vending machines are common; the comparison quantity is how
  often these appear in an arbitrary equivalent window, not whether they appear.
- Include a control: score the same five items against a reading drawn for a
  different sign, or a shuffled transcript, with the scorer blind to which
  transcript is which ([Mason & Budge, 2011](https://pubmed.ncbi.nlm.nih.gov/21315874/)).
- Record the result, including a null result, before interpreting it.

**Expected outcome.** Matches on several items, at a rate indistinguishable from
the control. That is the informative outcome, and it is worth having in writing
the next time a specific detail feels uncanny.

## Non-claims

- This ledger does not establish that the source is accurate, inaccurate,
  predictive, or causal. It establishes only what may and may not be inferred
  from it.
- Classifying a statement as testable does not mean it has been tested. No
  observation in this document has been made.
- The three surviving heuristics are not endorsed *because* the source stated
  them. They are endorsed because they are separately defensible, and they would
  survive the source being discarded entirely.
- No statement in the ledger's "testable translation" column authorizes human
  research, customer outreach, or external data collection. Those require
  separate explicit authorization and appropriate consent.
