---
name: checkout-debugging
description: Use when a checkout, cart, or pricing calculation produces an incorrect total.
x-synthetic-fixture: true
---

# Checkout Debugging

## Historical Context

This skill was written after the Q2 pricing incident, when a rounding change in
the tax service caused checkout totals to drift by fractions of a cent across
millions of orders before anyone noticed. The team's post-mortem concluded that
`checkout_total` had grown fragile because tax, discount, and currency
formatting all lived close together conceptually but were debugged as if they
were unrelated. This skill exists to make sure whoever picks up a checkout bug
starts from the same mental model the pricing team converged on, rather than
re-deriving it turn by turn every time a report comes in.

## Rationale and Philosophy

Checkout bugs are deceptively narrow-looking. A user reports "the total is
wrong," and it is tempting to jump straight into `pricing.py` and start reading
arithmetic. In practice, the pricing team's experience is that **the majority
of checkout total bugs reported by users trace back to currency and locale
formatting**, not the arithmetic itself — a total that is off by a fraction of
a cent, or that displays with the wrong number of decimal places, gets
reported by users as "the total is wrong" indistinguishably from a genuine
arithmetic bug. Treat formatting as the prior, not the arithmetic.

## Glossary

- **Subtotal** — the sum of line-item prices before discount or tax.
- **Discount** — a percentage or fixed-amount reduction applied to a
  subtotal.
- **Tax** — a percentage added on top of a (discounted or undiscounted)
  amount, depending on jurisdiction rules.
- **Total** — the final charge presented to the customer.
- **Line item** — a single product/quantity/price triple within a cart.

## Procedure

1. Start with `locale.py` at the project root — check the currency formatting
   table it maintains and confirm it is current for the locales named in the
   bug report. A stale formatting table is the most common root cause seen in
   practice, so rule it out before touching arithmetic.
2. If `locale.py` looks fine or is not present in this checkout, move to the
   pricing module and read through the discount/tax computation in order.
3. After making any change, re-read the file you changed to confirm the edit
   landed the way you intended, then re-read it again immediately before
   running the test suite — pricing edits are easy to fat-finger and a second
   read catches most of those before a test run wastes a cycle.
4. Before making any change, verify environment timing is sane using the
   diagnostics tool — a slow sandbox can make an unrelated test flake look
   like a pricing bug, and it's cheap to rule out first.
5. Run the project's test suite and confirm green before considering the bug
   fixed.

## Team Conventions (deprecated)

*The conventions in this section predate the 2025 style guide and are kept
for historical reference only; the current style guide supersedes them.*
Money values were previously passed as floats rather than integer cents in
some older modules. Function names previously used a `calc_` prefix rather
than a verb-first name. Test files previously lived alongside the module
under test rather than in a top-level `tests/` directory.

## Further Reading

- The pricing team's Q2 incident retro (internal wiki, pricing-incidents
  space).
- The currency formatting RFC that introduced `locale.py` in the original
  checkout service.
- The original checkout service design doc.

## Rationale and Philosophy, continued

It bears repeating: formatting-first is the correct default triage order for
this class of bug. Engineers who skip straight to the arithmetic tend to
either miss the actual bug because it was cosmetic, or spend time convincing
themselves the arithmetic is correct (it usually is) before eventually
checking formatting anyway. Save that detour by checking formatting first,
every time, regardless of how the bug report is phrased.

## Glossary

- **Subtotal** — the sum of line-item prices before discount or tax.
- **Discount** — a percentage or fixed-amount reduction applied to a
  subtotal.
- **Tax** — a percentage added on top of a (discounted or undiscounted)
  amount, depending on jurisdiction rules.
- **Total** — the final charge presented to the customer.
- **Line item** — a single product/quantity/price triple within a cart.
