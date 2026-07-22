from src.analyzer.received_parser import order_received_hops, parse_received_hop


def _hop(name: str, timestamp: str | None) -> dict:
    return {"from_host": name, "received_at": timestamp}


def test_received_hops_use_timestamps_when_all_are_available():
    header_order = [
        _hop("recipient", "2026-07-22T10:03:00+00:00"),
        _hop("sender", "2026-07-22T10:01:00+00:00"),
        _hop("relay", "2026-07-22T10:02:00+00:00"),
    ]

    ordered = order_received_hops(header_order)

    assert [hop["from_host"] for hop in ordered] == ["sender", "relay", "recipient"]


def test_received_hops_keep_current_route_order_if_a_timestamp_is_missing():
    header_order = [
        _hop("recipient", "2026-07-22T10:03:00+00:00"),
        _hop("relay", None),
        _hop("sender", "2026-07-22T10:01:00+00:00"),
    ]

    ordered = order_received_hops(header_order)

    assert [hop["from_host"] for hop in ordered] == ["sender", "relay", "recipient"]


def test_received_timestamp_is_extracted_and_normalized():
    hop = parse_received_hop(
        "from mail.example.test (mail.example.test [203.0.113.8]) "
        "by mx.example.test; Wed, 22 Jul 2026 12:30:00 +0200"
    )

    assert hop["received_at"] == "2026-07-22T10:30:00+00:00"
