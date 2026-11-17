import os
import sys
import hmac
import ctypes
import base64
import hashlib
import secrets
from ctypes import wintypes

KDF_NAME = "pbkdf2_sha256"
DEFAULT_KDF_ITERS = 300_000
MIN_KDF_ITERS = 10_000
MAX_KDF_ITERS = 5_000_000

ENC_PREFIX_GCM = "enc:v1:"
ENC_PREFIX_FALLBACK = "enc:s1:"
_AAD_V1 = b"anpr:source:v1"
_AAD_S1 = b"anpr:source:s1"


def is_encrypted_source(value: str) -> bool:
    s = str(value or "").strip()
    return s.startswith(ENC_PREFIX_GCM) or s.startswith(ENC_PREFIX_FALLBACK)


def _derive_keys(master_key: bytes) -> tuple[bytes, bytes]:
    verifier = hmac.new(master_key, b"anpr-auth-verify-v1", hashlib.sha256).digest()
    enc_key = hmac.new(master_key, b"anpr-source-enc-v1", hashlib.sha256).digest()
    return verifier, enc_key


def normalize_auth_cfg(raw_auth: dict) -> dict:
    if not isinstance(raw_auth, dict) or not bool(raw_auth.get("enabled", False)):
        return {"enabled": False}

    kdf = str(raw_auth.get("kdf") or KDF_NAME).strip() or KDF_NAME
    try:
        iters = int(raw_auth.get("iterations", DEFAULT_KDF_ITERS))
    except Exception:
        iters = DEFAULT_KDF_ITERS
    iters = max(MIN_KDF_ITERS, min(MAX_KDF_ITERS, iters))

    salt = str(raw_auth.get("salt") or "").strip()
    verifier = str(raw_auth.get("verifier") or "").strip()
    return {
        "enabled": True,
        "kdf": kdf,
        "iterations": iters,
        "salt": salt,
        "verifier": verifier,
    }


def create_auth_material(password: str, iterations: int = DEFAULT_KDF_ITERS) -> tuple[dict, bytes]:
    if not isinstance(password, str) or password == "":
        raise ValueError("Password must be a non-empty string")
    iters = max(MIN_KDF_ITERS, min(MAX_KDF_ITERS, int(iterations)))
    salt = secrets.token_bytes(16)
    master = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=32)
    verifier, enc_key = _derive_keys(master)
    auth_cfg = {
        "enabled": True,
        "kdf": KDF_NAME,
        "iterations": iters,
        "salt": base64.b64encode(salt).decode("ascii"),
        "verifier": base64.b64encode(verifier).decode("ascii"),
    }
    return auth_cfg, enc_key


def verify_and_derive(password: str, auth_cfg: dict) -> tuple[bool, bytes | None]:
    if not isinstance(password, str) or password == "":
        return False, None
    norm = normalize_auth_cfg(auth_cfg)
    if not norm.get("enabled"):
        return False, None
    if norm.get("kdf") != KDF_NAME:
        return False, None
    try:
        salt = base64.b64decode(norm["salt"].encode("ascii"), validate=True)
        expected_verifier = base64.b64decode(norm["verifier"].encode("ascii"), validate=True)
    except Exception:
        return False, None
    if len(salt) < 8 or len(expected_verifier) != 32:
        return False, None

    master = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        int(norm["iterations"]),
        dklen=32,
    )
    computed_verifier, enc_key = _derive_keys(master)
    if hmac.compare_digest(computed_verifier, expected_verifier):
        return True, enc_key
    return False, None


# ----------------------------------------------------------------------
# Нативний AES-256-GCM через Windows CNG (bcrypt.dll) без сторонніх залежностей
# ----------------------------------------------------------------------
_BCRYPT = None
if sys.platform == "win32":
    try:
        class _BCRYPT_AUTH_INFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.ULONG),
                ("dwInfoVersion", wintypes.ULONG),
                ("pbNonce", ctypes.POINTER(ctypes.c_ubyte)),
                ("cbNonce", wintypes.ULONG),
                ("pbAuthData", ctypes.POINTER(ctypes.c_ubyte)),
                ("cbAuthData", wintypes.ULONG),
                ("pbTag", ctypes.POINTER(ctypes.c_ubyte)),
                ("cbTag", wintypes.ULONG),
                ("pbMacContext", ctypes.POINTER(ctypes.c_ubyte)),
                ("cbMacContext", wintypes.ULONG),
                ("cbAAD", wintypes.ULONG),
                ("cbData", ctypes.c_uint64),
                ("dwFlags", wintypes.ULONG),
            ]

        _dll = ctypes.windll.bcrypt
        _NTSTATUS = ctypes.c_long

        _dll.BCryptOpenAlgorithmProvider.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            wintypes.ULONG,
        ]
        _dll.BCryptOpenAlgorithmProvider.restype = _NTSTATUS

        _dll.BCryptCloseAlgorithmProvider.argtypes = [ctypes.c_void_p, wintypes.ULONG]
        _dll.BCryptCloseAlgorithmProvider.restype = _NTSTATUS

        _dll.BCryptSetProperty.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            wintypes.ULONG,
            wintypes.ULONG,
        ]
        _dll.BCryptSetProperty.restype = _NTSTATUS

        _dll.BCryptGetProperty.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
            wintypes.ULONG,
        ]
        _dll.BCryptGetProperty.restype = _NTSTATUS

        _dll.BCryptGenerateSymmetricKey.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
            wintypes.ULONG,
        ]
        _dll.BCryptGenerateSymmetricKey.restype = _NTSTATUS

        _dll.BCryptDestroyKey.argtypes = [ctypes.c_void_p]
        _dll.BCryptDestroyKey.restype = _NTSTATUS

        _dll.BCryptEncrypt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
            wintypes.ULONG,
        ]
        _dll.BCryptEncrypt.restype = _NTSTATUS

        _dll.BCryptDecrypt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
            wintypes.ULONG,
        ]
        _dll.BCryptDecrypt.restype = _NTSTATUS

        _BCRYPT = _dll
    except Exception:
        _BCRYPT = None


def _bcrypt_aes_gcm_open(enc_key: bytes):
    if _BCRYPT is None:
        raise RuntimeError("Windows CNG bcrypt.dll is unavailable")
    if not isinstance(enc_key, (bytes, bytearray)) or len(enc_key) != 32:
        raise ValueError("AES-256-GCM key must be 32 bytes")

    alg = ctypes.c_void_p()
    st = _BCRYPT.BCryptOpenAlgorithmProvider(ctypes.byref(alg), "AES", None, 0)
    if st != 0:
        raise OSError(f"BCryptOpenAlgorithmProvider failed: 0x{st & 0xffffffff:08x}")

    key = ctypes.c_void_p()
    try:
        mode = "ChainingModeGCM".encode("utf-16-le") + b"\x00\x00"
        mode_buf = ctypes.create_string_buffer(mode, len(mode))
        st = _BCRYPT.BCryptSetProperty(alg, "ChainingMode", mode_buf, len(mode), 0)
        if st != 0:
            raise OSError(f"BCryptSetProperty(ChainingMode) failed: 0x{st & 0xffffffff:08x}")

        obj_len = wintypes.ULONG(0)
        res_len = wintypes.ULONG(0)
        st = _BCRYPT.BCryptGetProperty(
            alg,
            "ObjectLength",
            ctypes.byref(obj_len),
            ctypes.sizeof(obj_len),
            ctypes.byref(res_len),
            0,
        )
        key_obj = (ctypes.c_ubyte * obj_len.value)() if (st == 0 and obj_len.value > 0) else None
        key_obj_len = int(obj_len.value) if key_obj is not None else 0

        key_buf = (ctypes.c_ubyte * 32).from_buffer_copy(enc_key)
        st = _BCRYPT.BCryptGenerateSymmetricKey(
            alg,
            ctypes.byref(key),
            key_obj,
            key_obj_len,
            key_buf,
            32,
            0,
        )
        if st != 0:
            raise OSError(f"BCryptGenerateSymmetricKey failed: 0x{st & 0xffffffff:08x}")
        return alg, key, key_obj
    except Exception:
        if key.value:
            _BCRYPT.BCryptDestroyKey(key)
        if alg.value:
            _BCRYPT.BCryptCloseAlgorithmProvider(alg, 0)
        raise


def _aes_gcm_encrypt_bytes(plaintext: bytes, enc_key: bytes) -> bytes:
    nonce = secrets.token_bytes(12)
    alg, key, _key_obj = _bcrypt_aes_gcm_open(enc_key)
    try:
        nonce_buf = (ctypes.c_ubyte * 12).from_buffer_copy(nonce)
        aad_buf = (ctypes.c_ubyte * len(_AAD_V1)).from_buffer_copy(_AAD_V1)
        tag_buf = (ctypes.c_ubyte * 16)()

        info = _BCRYPT_AUTH_INFO()
        info.cbSize = ctypes.sizeof(_BCRYPT_AUTH_INFO)
        info.dwInfoVersion = 1
        info.pbNonce = nonce_buf
        info.cbNonce = 12
        info.pbAuthData = aad_buf
        info.cbAuthData = len(_AAD_V1)
        info.pbTag = tag_buf
        info.cbTag = 16

        pt_len = len(plaintext)
        pt_buf = (ctypes.c_ubyte * max(1, pt_len))()
        if pt_len > 0:
            ctypes.memmove(pt_buf, plaintext, pt_len)
        ct_buf = (ctypes.c_ubyte * max(1, pt_len))()
        out_len = wintypes.ULONG(0)

        st = _BCRYPT.BCryptEncrypt(
            key,
            pt_buf if pt_len > 0 else None,
            pt_len,
            ctypes.byref(info),
            None,
            0,
            ct_buf if pt_len > 0 else None,
            pt_len,
            ctypes.byref(out_len),
            0,
        )
        if st != 0:
            raise OSError(f"BCryptEncrypt failed: 0x{st & 0xffffffff:08x}")
        return nonce + bytes(tag_buf) + bytes(ct_buf[: out_len.value])
    finally:
        if key.value:
            _BCRYPT.BCryptDestroyKey(key)
        if alg.value:
            _BCRYPT.BCryptCloseAlgorithmProvider(alg, 0)


def _aes_gcm_decrypt_bytes(payload: bytes, enc_key: bytes) -> bytes:
    if len(payload) < 28:
        raise ValueError("Encrypted payload too short")
    nonce = payload[:12]
    tag = payload[12:28]
    ct = payload[28:]

    alg, key, _key_obj = _bcrypt_aes_gcm_open(enc_key)
    try:
        nonce_buf = (ctypes.c_ubyte * 12).from_buffer_copy(nonce)
        aad_buf = (ctypes.c_ubyte * len(_AAD_V1)).from_buffer_copy(_AAD_V1)
        tag_buf = (ctypes.c_ubyte * 16).from_buffer_copy(tag)

        info = _BCRYPT_AUTH_INFO()
        info.cbSize = ctypes.sizeof(_BCRYPT_AUTH_INFO)
        info.dwInfoVersion = 1
        info.pbNonce = nonce_buf
        info.cbNonce = 12
        info.pbAuthData = aad_buf
        info.cbAuthData = len(_AAD_V1)
        info.pbTag = tag_buf
        info.cbTag = 16

        ct_len = len(ct)
        ct_buf = (ctypes.c_ubyte * max(1, ct_len))()
        if ct_len > 0:
            ctypes.memmove(ct_buf, ct, ct_len)
        pt_buf = (ctypes.c_ubyte * max(1, ct_len))()
        out_len = wintypes.ULONG(0)

        st = _BCRYPT.BCryptDecrypt(
            key,
            ct_buf if ct_len > 0 else None,
            ct_len,
            ctypes.byref(info),
            None,
            0,
            pt_buf if ct_len > 0 else None,
            ct_len,
            ctypes.byref(out_len),
            0,
        )
        if st != 0:
            raise ValueError("Failed to decrypt video source (invalid key or tampered data)")
        return bytes(pt_buf[: out_len.value])
    finally:
        if key.value:
            _BCRYPT.BCryptDestroyKey(key)
        if alg.value:
            _BCRYPT.BCryptCloseAlgorithmProvider(alg, 0)


# ----------------------------------------------------------------------
# Кросплатформений fallback (HMAC-SHA256-CTR + HMAC-SHA256 EtM)
# ----------------------------------------------------------------------
def _s1_keystream(cipher_key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        blk = hashlib.sha256(cipher_key + nonce + counter.to_bytes(4, "big")).digest()
        out.extend(blk)
        counter += 1
    return bytes(out[:length])


def _s1_encrypt_bytes(plaintext: bytes, enc_key: bytes) -> bytes:
    nonce = secrets.token_bytes(16)
    c_key = hmac.new(enc_key, b"s1-enc", hashlib.sha256).digest()
    m_key = hmac.new(enc_key, b"s1-mac", hashlib.sha256).digest()
    ks = _s1_keystream(c_key, nonce, len(plaintext))
    ct = bytes(a ^ b for a, b in zip(plaintext, ks))
    tag = hmac.new(m_key, _AAD_S1 + nonce + ct, hashlib.sha256).digest()
    return nonce + tag + ct


def _s1_decrypt_bytes(payload: bytes, enc_key: bytes) -> bytes:
    if len(payload) < 48:
        raise ValueError("Encrypted payload too short")
    nonce = payload[:16]
    tag = payload[16:48]
    ct = payload[48:]
    c_key = hmac.new(enc_key, b"s1-enc", hashlib.sha256).digest()
    m_key = hmac.new(enc_key, b"s1-mac", hashlib.sha256).digest()
    expected = hmac.new(m_key, _AAD_S1 + nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("Failed to decrypt video source (invalid key or tampered data)")
    ks = _s1_keystream(c_key, nonce, len(ct))
    return bytes(a ^ b for a, b in zip(ct, ks))


def encrypt_source(plaintext: str, enc_key: bytes) -> str:
    s = str(plaintext or "")
    if s == "":
        return ""
    if not isinstance(enc_key, (bytes, bytearray)) or len(enc_key) != 32:
        raise ValueError("Encryption key must be 32 bytes")
    raw = s.encode("utf-8")
    if _BCRYPT is not None:
        payload = _aes_gcm_encrypt_bytes(raw, enc_key)
        return ENC_PREFIX_GCM + base64.b64encode(payload).decode("ascii")
    payload = _s1_encrypt_bytes(raw, enc_key)
    return ENC_PREFIX_FALLBACK + base64.b64encode(payload).decode("ascii")


def decrypt_source(token: str, enc_key: bytes) -> str:
    s = str(token or "")
    if s == "":
        return ""
    if s.startswith(ENC_PREFIX_GCM):
        b64_part = s[len(ENC_PREFIX_GCM):]
        payload = base64.b64decode(b64_part.encode("ascii"), validate=True)
        return _aes_gcm_decrypt_bytes(payload, enc_key).decode("utf-8")
    if s.startswith(ENC_PREFIX_FALLBACK):
        b64_part = s[len(ENC_PREFIX_FALLBACK):]
        payload = base64.b64decode(b64_part.encode("ascii"), validate=True)
        return _s1_decrypt_bytes(payload, enc_key).decode("utf-8")
    return s
