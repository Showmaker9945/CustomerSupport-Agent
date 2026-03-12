"""Tests for structured business data bootstrapping."""

from src.db.repositories import (
    get_latest_invoice_record,
    get_subscription_record,
    get_user_record,
    list_ticket_records,
)


def test_demo_seed_bootstraps_business_records(isolated_business_db):
    user = get_user_record("user_001")
    subscription = get_subscription_record("user_001")
    invoice = get_latest_invoice_record("user_001")
    tickets = list_ticket_records("user_001")

    assert user is not None
    assert user["email"] == "alice.johnson@example.com"
    assert subscription is not None
    assert subscription["plan_name"] == "Pro 团队版"
    assert invoice is not None
    assert invoice["invoice_id"] == "INV-202603-0001"
    assert len(invoice["line_items"]) == 2
    assert len(tickets) == 2
