class TerminalInputBuffer:
    def __init__(self):
        self.pending = bytearray()

    def feed(self, data: bytes) -> bytes:
        output = bytearray()
        for byte in data:
            if byte == 0x1B:
                output.extend(self.pending)
                self.pending.clear()
                self.pending.append(byte)
            elif self.pending:
                self.pending.append(byte)
                if len(self.pending) == 2 and byte == ord("["):
                    continue
                if (len(self.pending) > 2 and self.pending[1] == ord("[")
                        and 0x20 <= byte <= 0x3F and len(self.pending) < 32):
                    continue
                output.extend(self.pending)
                self.pending.clear()
            else:
                output.append(byte)
        return bytes(output)

    def flush_escape(self) -> bytes:
        if self.pending == b"\x1b":
            self.pending.clear()
            return b"\x1b"
        return b""
