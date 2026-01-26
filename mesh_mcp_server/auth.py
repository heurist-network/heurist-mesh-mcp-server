import contextvars
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("mesh-mcp.auth")

request_auth_context: contextvars.ContextVar[Optional["AuthContext"]] = (
    contextvars.ContextVar("request_auth_context", default=None)
)


class AuthError(Enum):
    MISSING_KEY = "API key required in Authorization header"
    INVALID_FORMAT = "Invalid API key format. Expected format: user_id-api_key"
    INVALID_KEY = "Invalid API key"
    INSUFFICIENT_CREDITS = "Insufficient credits"
    USER_NOT_FOUND = "User data not found"
    DYNAMODB_ERROR = "Database error"


@dataclass
class AuthContext:
    """Holds authentication context for a request."""

    user_id: str
    api_key: str
    heurist_key: str  # Full key (user_id-api_key)
    remaining_credits: Decimal
    user_data: dict[str, Any]


class AuthValidator:
    """Validates API keys and manages credits via DynamoDB."""

    def __init__(self):
        self.table_name = os.getenv("DYNAMODB_TABLE_NAME", "api_credits")
        self.enabled = os.getenv("AUTH_ENABLED", "true").lower() == "true"

        if self.enabled:
            try:
                self.dynamodb = boto3.resource("dynamodb")
                self.table = self.dynamodb.Table(self.table_name)
                logger.info(f"Auth enabled, using DynamoDB table: {self.table_name}")
            except Exception as e:
                logger.error(f"Failed to initialize DynamoDB: {e}")
                raise
        else:
            self.dynamodb = None
            self.table = None
            logger.warning("Auth is DISABLED - all requests will be allowed")

    @staticmethod
    def parse_api_key(heurist_key: str) -> Tuple[str, str]:
        """Parse heurist key into (user_id, api_key) tuple.

        Supports both '-' and '#' as delimiters.
        """
        if "-" in heurist_key:
            parts = heurist_key.split("-", 1)
            return parts[0], parts[1]
        elif "#" in heurist_key:
            parts = heurist_key.split("#", 1)
            return parts[0], parts[1]
        else:
            raise ValueError(AuthError.INVALID_FORMAT.value)

    @staticmethod
    def extract_api_key_from_headers(
        authorization: Optional[str] = None,
        x_heurist_api_key: Optional[str] = None,
    ) -> Optional[str]:
        """Extract API key from HTTP headers.

        Supports (in order of priority):
        1. X-HEURIST-API-KEY header (preferred, matches Heurist REST API style)
        2. Authorization: Bearer <token>
        3. Authorization: <token> (raw, no scheme)
        """
        if x_heurist_api_key:
            return x_heurist_api_key.strip()
        if not authorization:
            return None
        authorization = authorization.strip()
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return authorization

    def validate_api_key(self, heurist_key: str) -> AuthContext:
        """Validate API key and return auth context.

        Raises ValueError with appropriate message on failure.
        """
        if not self.enabled:
            return AuthContext(
                user_id="anonymous",
                api_key="disabled",
                heurist_key=heurist_key or "anonymous",
                remaining_credits=Decimal("999999"),
                user_data={},
            )

        if not heurist_key:
            raise ValueError(AuthError.MISSING_KEY.value)

        try:
            user_id, api_key = self.parse_api_key(heurist_key)
        except ValueError:
            raise ValueError(AuthError.INVALID_FORMAT.value)

        try:
            api_key_response = self.table.get_item(
                Key={"user_id": user_id, "api_key": api_key}
            )
            if "Item" not in api_key_response:
                raise ValueError(AuthError.INVALID_KEY.value)
            user_data_response = self.table.get_item(
                Key={"user_id": user_id, "api_key": "USER_DATA"}
            )
            if "Item" not in user_data_response:
                raise ValueError(AuthError.USER_NOT_FOUND.value)
            user_data = user_data_response["Item"]
            remaining_credits = Decimal(str(user_data.get("remaining_credits", 0)))
            if remaining_credits <= 0:
                raise ValueError(AuthError.INSUFFICIENT_CREDITS.value)
            return AuthContext(
                user_id=user_id,
                api_key=api_key,
                heurist_key=heurist_key,
                remaining_credits=remaining_credits,
                user_data=user_data,
            )

        except ClientError as e:
            logger.error(f"DynamoDB error: {e}")
            raise ValueError(f"{AuthError.DYNAMODB_ERROR.value}: {str(e)}")

    def deduct_credits(self, user_id: str, amount: int) -> bool:
        """Deduct credits from user's balance.

        Uses atomic decrement to prevent race conditions.
        Returns True if successful, False if insufficient credits.
        """
        if not self.enabled or amount <= 0:
            return True

        try:
            self.table.update_item(
                Key={"user_id": user_id, "api_key": "USER_DATA"},
                UpdateExpression="SET remaining_credits = remaining_credits - :amount",
                ConditionExpression="remaining_credits >= :amount",
                ExpressionAttributeValues={":amount": Decimal(str(amount))},
            )
            logger.info(f"Deducted {amount} credits from user {user_id}")
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning(f"Insufficient credits for user {user_id}")
                return False
            logger.error(f"Failed to deduct credits: {e}")
            return False


_validator: Optional[AuthValidator] = None


def get_validator() -> AuthValidator:
    """Get or create the auth validator singleton."""
    global _validator
    if _validator is None:
        _validator = AuthValidator()
    return _validator


def get_current_auth_context() -> Optional[AuthContext]:
    """Get the auth context for the current request."""
    return request_auth_context.get()


def set_current_auth_context(ctx: Optional[AuthContext]) -> contextvars.Token:
    """Set the auth context for the current request."""
    return request_auth_context.set(ctx)
