# common/broadcast.py
# Utilities to serialize/deserialize torch state_dicts for ZeroMQ
import io, torch, base64

def state_to_blob(state_dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return base64.b64encode(buf.getvalue())

def blob_to_state(blob: bytes):
    raw = base64.b64decode(blob)
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu")
