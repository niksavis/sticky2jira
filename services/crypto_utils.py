"""
Crypto Utilities - Cross-platform encryption for sensitive data.
Uses cryptography library (Fernet/AES) for all platforms.
"""

import logging
from cryptography.fernet import Fernet
import os

logger = logging.getLogger(__name__)

# Encryption key stored in .env file (git-ignored)
KEY_FILE = ".encryption_key"


def _get_or_create_key() -> bytes:
    """Get or create encryption key from .encryption_key file."""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        logger.info(f"Created encryption key at {KEY_FILE}")
        return key


def encrypt_token(plaintext: str) -> bytes:
    """
    Encrypt a token using Fernet (AES-128).

    Args:
        plaintext: The token to encrypt (string)

    Returns:
        Encrypted bytes (safe to store in database as BLOB)

    Raises:
        Exception: If encryption fails
    """
    try:
        plaintext_bytes = plaintext.encode("utf-8")
        key = _get_or_create_key()
        fernet = Fernet(key)
        encrypted_bytes = fernet.encrypt(plaintext_bytes)

        logger.debug("Token encrypted successfully")
        return encrypted_bytes

    except Exception as e:
        logger.error(f"Token encryption failed: {e}")
        raise


def decrypt_token(encrypted_bytes: bytes) -> str:
    """
    Decrypt a token using Fernet (AES-128).

    Args:
        encrypted_bytes: The encrypted token (from database BLOB)

    Returns:
        Decrypted plaintext string

    Raises:
        Exception: If decryption fails
    """
    try:
        key = _get_or_create_key()
        fernet = Fernet(key)
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        plaintext = decrypted_bytes.decode("utf-8")

        logger.debug("Token decrypted successfully")
        return plaintext

    except Exception as e:
        logger.error(f"Token decryption failed: {e}")
        raise


def is_encrypted(data: bytes) -> bool:
    """
    Check if data appears to be Fernet-encrypted.

    Fernet encrypted data is base64-encoded and starts with specific version bytes.
    This is a heuristic check for migration from plaintext.

    Args:
        data: Bytes to check

    Returns:
        True if data looks encrypted, False if plaintext
    """
    if not data or len(data) < 4:
        return False

    # Try to decode as UTF-8 - if successful and readable, likely plaintext
    try:
        text = data.decode("utf-8")
        # Fernet tokens are base64 and start with 'gAAAAA' (version marker)
        # If it's readable text without base64 markers, it's plaintext
        if text.startswith("gAAAAA"):
            return True
        # If it decodes cleanly and doesn't look like base64, it's plaintext
        return False
    except (UnicodeDecodeError, AttributeError):
        # Binary data that doesn't decode = likely encrypted
        return True
