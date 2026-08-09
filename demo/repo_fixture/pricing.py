"""Pricing calculations for checkout."""

TAX_RATE = 0.08  # 8% sales tax


def discount_amount_cents(subtotal_cents, discount_pct):
    """Dollar amount (in cents) knocked off by a percentage discount."""
    return round(subtotal_cents * discount_pct / 100)


def tax_cents(amount_cents):
    """Sales tax owed on an amount, in cents."""
    return round(amount_cents * TAX_RATE)


def checkout_total(subtotal_cents, discount_pct=0):
    """Final charge for a cart, in cents: subtotal, minus discount, plus tax."""
    discount = discount_amount_cents(subtotal_cents, discount_pct)
    tax = tax_cents(subtotal_cents)  # BUG: taxes the pre-discount subtotal
    return subtotal_cents - discount + tax
