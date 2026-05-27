"""IICP structured error type (ADR-016 §3)."""

from __future__ import annotations


class IicpError(Exception):
    """Typed error surface — never exposes raw HTTP details to callers."""

    def __init__(
        self,
        code: str,
        message: str,
        component: str,
        retryable: bool = False,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.component = component
        self.retryable = retryable
        self.http_status = http_status

    def __repr__(self) -> str:
        return (
            f"IicpError(code={self.code!r}, component={self.component!r}, "
            f"retryable={self.retryable}, http_status={self.http_status})"
        )


# Well-known error codes (Phase 1 range, IICP-E001..E010)
_RETRYABLE_CODES = {"IICP-E003", "IICP-E004", "IICP-E005"}


def from_http(status: int, body: dict, component: str) -> IicpError:
    """Build a typed IicpError from an HTTP response body."""
    code = body.get("code", f"IICP-E{status:03d}")
    message = body.get("message", body.get("error", "Unexpected error"))
    return IicpError(
        code=code,
        message=message,
        component=component,
        retryable=code in _RETRYABLE_CODES or status in (429, 503),
        http_status=status,
    )
