import pytest

from airouter.terminal import TerminalInputBuffer


@pytest.mark.parametrize("report", [
    b"\x1b[<64;64;31M", b"\x1b[<65;64;31M", b"\x1b[<35;64;31M",
    b"\x1b[<0;10;10M", b"\x1b[<0;10;10m", b"\x1b[A", b"\x1b[1;5D",
])
def test_terminal_reports_are_forwarded_whole_at_every_split(report):
    for split in range(1, len(report)):
        buffer = TerminalInputBuffer()
        assert buffer.feed(b"halo" + report[:split]) == b"halo"
        assert buffer.feed(report[split:] + b"x") == report + b"x"
        assert not buffer.pending


def test_mouse_report_can_span_many_reads_without_leaking_digits():
    buffer = TerminalInputBuffer()
    report = b"\x1b[<64;64;31M"
    for byte in report[:-1]:
        assert buffer.feed(bytes([byte])) == b""
    assert buffer.feed(report[-1:]) == report


def test_lone_escape_can_be_flushed_for_cancel():
    buffer = TerminalInputBuffer()
    assert buffer.feed(b"\x1b") == b""
    assert buffer.flush_escape() == b"\x1b"
    assert buffer.feed(b"x") == b"x"


def test_idle_flush_keeps_partial_mouse_report():
    buffer = TerminalInputBuffer()
    assert buffer.feed(b"\x1b[<6") == b""
    assert buffer.flush_escape() == b""
    assert buffer.feed(b"4;64;31M") == b"\x1b[<64;64;31M"


@pytest.mark.parametrize("data", [
    "halo 日本語 😀 64;31;31M\r\n".encode(), b"\x1bx\x1b[A\x1b[B\x1b[C\x1b[D",
    b"\x1b[200~\x1b[<64;31;31M\ncontoh\x1b[201~x", b"\x1b[<64;31;31M" * 100,
])
def test_keyboard_paste_and_bursts_keep_their_exact_bytes(data):
    buffer = TerminalInputBuffer()
    assert b"".join(buffer.feed(bytes([byte])) for byte in data) == data
    assert not buffer.pending


def test_invalid_sequence_does_not_buffer_unbounded_input():
    buffer = TerminalInputBuffer()
    data = b"\x1b[<" + b"1;" * 100
    output = buffer.feed(data)
    assert len(buffer.pending) < 32
    assert output + buffer.pending == data
