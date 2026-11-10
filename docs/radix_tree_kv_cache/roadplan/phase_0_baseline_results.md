# Phase 0 results: baseline

The stock server reaches the ideal reuse for stages 1 and 2 whenever their state survives, but loses most of it when a slot is overwritten without a RAM cache, when the RAM cache is too small, for a pinned background task (BUG-001), and for SWA and hybrid models. The full tables are in [../phase_0_baseline.md](../phase_0_baseline.md).

## Key numbers

The phone matters most: a missed prompt token costs about 0.1 ms on the laptop and 6-22 ms on the phone.

| Model (phone) | A | C | D | E | F |
| --- | --- | --- | --- | --- | --- |
| Llama turn time | 22.0 s | 22.5 s | 36.5 s | 29.1 s | 45.3 s |
| Llama energy per turn | 41 J | 41 J | 65 J | 52 J | 81 J |
| Gemma turn time | 70.6 s | 48.0 s | 70.8 s | 79.3 s | 70.9 s |
| LFM2 turn time | 26.2 s | 25.9 s | 25.9 s | 31.1 s | 39.4 s |

- 66 laptop runs and 54 phone runs, 0 failed runs, 0 request errors.
- Config F reuses 0-6% of stages 1 and 2, against 85-89% possible.
- The RAM cache swap costs 85 ms per turn on the phone and 131 ms on the laptop.
- BUG-001: the background task reuses 0%, while 40-52% is possible.
- Gemma's 512-token SWA checkpoints are erased as invalid 44-88 times per run.

## Insights

What Phase 0 changed in the plan.

| Finding | Consequence | Phase |
| --- | --- | --- |
| Overwriting without a RAM cache (F) is the largest loss | A cache that keeps shared prefixes fixes it directly | 2 |
| On the phone, the RAM cache is as good as one slot per stage (A = C) | Restoring is always worth it on unified memory | 1 measures it |
| On the laptop, a swap can cost more than recomputing a short prompt | Restores need a threshold | 1, 2 |
| The small RAM cache (D) is a capacity limit for Llama, not BUG-003 | Store shared prefixes once, not whole prompts per entry | 2 |
| BUG-001 reproduces | Stateless slots remove this class of bug | 1 tests upstream fixes, 3 removes the cause |
| SWA and hybrid checkpoints land after the shared prefix | Snapshots at stage boundaries and on a grid | 4 |
| The background task slows stage 1 on the phone from 1.0 s to 5.2 s | Scheduling matters as much as caching | 6 |
| The KV pool never filled, so no slot purges ran | Testing purges needs a smaller `-c` | 3 |

## Open after Phase 0

- Gemma in config C on the phone reaches 51% in stage 3, but 0% on the laptop. The cause is not verified.
- BUG-003 is not confirmed; a log line with the removed entry's token count would settle it.
- Laptop energy is GPU board power only; whole-laptop energy needs RAPL.
