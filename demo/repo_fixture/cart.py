"""Cart line items."""

from dataclasses import dataclass, field


@dataclass
class LineItem:
    name: str
    unit_price_cents: int
    quantity: int = 1


@dataclass
class Cart:
    items: list = field(default_factory=list)
    discount_pct: int = 0

    def add(self, name, unit_price_cents, quantity=1):
        self.items.append(LineItem(name, unit_price_cents, quantity))

    def subtotal_cents(self):
        return sum(item.unit_price_cents * item.quantity for item in self.items)
