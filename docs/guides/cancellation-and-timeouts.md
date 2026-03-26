# Cancellation and Timeouts

Owly enforces stream safety with two timeout controls:

- `request_timeout`: max time budget for provider call and per-stream iteration
- `first_token_timeout`: max wait for first emitted token

Configure globally:

```python
from owly_ai.infra.config import OwlyConfig

cfg = OwlyConfig(
    request_timeout=30.0,
    first_token_timeout=5.0,
)
```

Cancellation behavior:

- upstream stream is closed on cancellation
- no additional tokens should be emitted after cancellation
- provider-level errors are normalized into Owly exceptions
