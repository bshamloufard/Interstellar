from cart import Cart
from pricing import checkout_total


def test_checkout_total_no_discount():
    cart = Cart()
    cart.add("Widget", 5000, quantity=2)  # $100.00 subtotal
    assert checkout_total(cart.subtotal_cents()) == 10800  # +8% tax


def test_checkout_total_with_discount():
    cart = Cart()
    cart.add("Widget", 5000, quantity=2)  # $100.00 subtotal
    cart.discount_pct = 10
    total = checkout_total(cart.subtotal_cents(), cart.discount_pct)
    assert total == 9720  # $90 discounted subtotal + 8% tax on $90 = $97.20
