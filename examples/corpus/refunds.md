# Refunds

*Invented sample text for the quickstart corpus, not a real refund policy.*

A refund is issued against the original payment method. Where that method has
expired, the balance is held as account credit and the customer is told once by
email and once in the console.

Partial refunds are allowed down to the smallest unit of the currency. A
partial refund does not reopen the dispute window.

## Timing

Card refunds settle in three to five business days. Bank transfers settle the
next business day in the SEPA zone and in up to five elsewhere. The settlement
clock starts when the refund is accepted by the processor, not when it is
requested in the console, and those two can be a day apart over a weekend.

A refund issued on the same calendar day as the original charge is usually
handled as a reversal by the issuer rather than as a refund. The customer sees
the charge disappear instead of a second line item, the funds return faster,
and the console still shows it as a refund because that is what was requested.
The missing line item is a common support question.

## What cannot be refunded

Fees on a transaction that was itself refunded are returned with it. Fees on a
disputed transaction are not, whichever way the dispute goes.

A transaction older than 180 days cannot be refunded through the network. The
console offers a bank transfer instead, which needs the customer's account
details and a second approval, and which is the only path that works once the
original authorisation has aged out.

Currency conversion is not reversed. A charge taken in EUR and settled in GBP
is refunded at the rate on the day of the refund, so the customer can receive
slightly more or slightly less than they paid. The difference falls on the
merchant, which is why refunds are reconciled separately from charges.

## Credit notes

A credit note is raised when the refund cannot reach the customer at all: a
closed bank account, a card issued by a bank that no longer exists, a business
that has been wound up. The note sits against the account and is applied to the
next invoice. It expires after two years, which is a legal limit rather than a
product decision.
