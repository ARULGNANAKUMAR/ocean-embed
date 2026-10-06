"""
firebase_auth.py
OceanVerse AI — Firebase Authentication + Role-Based Access Control.

This module is additive: it does not touch ocean_platform.py, admin.py, or the
existing users/sessions tables. It introduces a *second, independent* identity
source (Firebase) and a small local role-mapping table, then exposes two
decorators — require_auth() and require_role() — that are the actual security
boundary for protected routes.

Design:
  1. The frontend signs the user in with the Firebase Web SDK and gets a
     short-lived Firebase ID token.
  2. The frontend POSTs that ID token once to /api/auth/firebase-session.
  3. This module verifies the token server-side with the Firebase Admin SDK
     (never trusts anything the client claims about itself), looks up/creates
     a row in firebase_roles keyed by the verified Firebase UID, and sets a
     Flask session cookie (signed with SECRET_KEY) containing uid/email/role.
  4. Every subsequent request is authorized off that server-signed session
     cookie via require_auth()/require_role() — never off a client-supplied
     header, body field, or query parameter.

Role assignment rule (Phase 5 requirement):
  - A UID is granted role="admin" ONLY if its verified email matches the
    ADMIN_EMAIL environment variable AT THE TIME OF FIRST LOGIN, or if an
    existing admin explicitly promotes another UID via the protected
    /api/admin/users/<uid>/role endpoint. A client can never set its own role.
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from functools import wraps

from flask import jsonify, redirect, request, session as flask_session

logger = logging.getLogger("firebase_auth")

# ──────────────────────────────────────────────────────────────
# FIREBASE ADMIN SDK INITIALIZATION
# ──────────────────────────────────────────────────────────────

_firebase_app = None
_firebase_available = False

try:
    import firebase_admin
    from firebase_admin import auth as fb_auth
    from firebase_admin import credentials

    _cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    _project_id = os.environ.get("FIREBASE_PROJECT_ID", "").strip()

    try:
        if _cred_path and os.path.exists(_cred_path):
            # Preferred: explicit service-account JSON, server-side only,
            # path supplied via env var. Never committed, never sent to
            # the frontend.
            cred = credentials.Certificate(_cred_path)
            _firebase_app = firebase_admin.initialize_app(cred, {
                "projectId": _project_id or None,
            })
            _firebase_available = True
        elif _project_id:
            # Fallback: Application Default Credentials (e.g. in a GCP-hosted
            # environment). Still verifies real Firebase-issued tokens.
            cred = credentials.ApplicationDefault()
            _firebase_app = firebase_admin.initialize_app(cred, {
                "projectId": _project_id,
            })
            _firebase_available = True
        else:
            logger.warning(
                "Firebase Admin SDK not initialized: set "
                "FIREBASE_SERVICE_ACCOUNT_PATH (and FIREBASE_PROJECT_ID) in "
                "your .env. Firebase-protected routes will reject all "
                "requests until this is configured."
            )
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        _firebase_available = False

except ImportError:
    logger.warning(
        "firebase-admin package not installed. Run "
        "`pip install firebase-admin` and configure .env to enable "
        "Firebase authentication."
    )


def firebase_ready() -> bool:
    return _firebase_available


# ──────────────────────────────────────────────────────────────
# LOCAL ROLE MAP (Firebase UID -> role)
# ──────────────────────────────────────────────────────────────

SCHEMA_FIREBASE_ROLES = """
CREATE TABLE IF NOT EXISTS firebase_roles (
    uid           TEXT PRIMARY KEY,
    email         TEXT,
    role          TEXT DEFAULT 'user',
    email_verified INTEGER DEFAULT 0,
    display_name  TEXT,
    created_at    TEXT,
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS security_audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT,
    uid        TEXT,
    email      TEXT,
    detail     TEXT,
    ip         TEXT,
    created_at TEXT
);
"""

VALID_ROLES = {"user", "admin"}


def init_firebase_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_FIREBASE_ROLES)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_security_event(conn: sqlite3.Connection, event_type: str, uid: str = "",
                        email: str = "", detail: str = "") -> None:
    """Audit log for security-relevant events. Never logs tokens/secrets."""
    try:
        init_firebase_tables(conn)
        conn.execute(
            "INSERT INTO security_audit_log (event_type, uid, email, detail, ip, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (event_type, uid, email, detail,
             request.remote_addr if request else "", _now()),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to write security audit log: {e}")


def get_or_create_role(conn: sqlite3.Connection, uid: str, email: str,
                        email_verified: bool, display_name: str = "") -> str:
    """
    Look up the role for a verified Firebase UID, creating a row on first
    login. The ONLY way a UID becomes 'admin' automatically is if its
    verified email matches ADMIN_EMAIL from the environment. This check
    happens here, server-side, against the token-verified email -- never
    against anything the client sends directly.
    """
    init_firebase_tables(conn)
    admin_email = os.environ.get("ADMIN_EMAIL", "").strip().lower()

    cur = conn.cursor()
    cur.execute("SELECT role FROM firebase_roles WHERE uid=?", (uid,))
    row = cur.fetchone()

    if row:
        role = row[0]
        conn.execute(
            "UPDATE firebase_roles SET email=?, email_verified=?, display_name=?, last_login=? WHERE uid=?",
            (email, 1 if email_verified else 0, display_name, _now(), uid),
        )
        conn.commit()
        return role

    role = "admin" if (admin_email and email.strip().lower() == admin_email) else "user"
    conn.execute(
        "INSERT INTO firebase_roles (uid, email, role, email_verified, display_name, created_at, last_login) "
        "VALUES (?,?,?,?,?,?,?)",
        (uid, email, role, 1 if email_verified else 0, display_name, _now(), _now()),
    )
    conn.commit()
    log_security_event(conn, "role_assigned", uid, email, f"role={role} (first login)")
    return role


def set_user_role(conn: sqlite3.Connection, target_uid: str, role: str) -> dict:
    """Explicit admin-only role change. Caller must already be verified admin."""
    if role not in VALID_ROLES:
        return {"status": "error", "error": f"Invalid role. Must be one of {sorted(VALID_ROLES)}"}
    init_firebase_tables(conn)
    cur = conn.cursor()
    cur.execute("SELECT uid FROM firebase_roles WHERE uid=?", (target_uid,))
    if not cur.fetchone():
        return {"status": "error", "error": "User not found"}
    conn.execute("UPDATE firebase_roles SET role=? WHERE uid=?", (role, target_uid))
    conn.commit()
    return {"status": "updated", "uid": target_uid, "role": role}


def list_firebase_users(conn: sqlite3.Connection) -> list:
    init_firebase_tables(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT uid, email, role, email_verified, display_name, created_at, last_login "
        "FROM firebase_roles ORDER BY created_at DESC"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ──────────────────────────────────────────────────────────────
# TOKEN VERIFICATION
# ──────────────────────────────────────────────────────────────

class TokenError(Exception):
    pass


def verify_id_token(id_token: str) -> dict:
    """
    Verify a Firebase ID token server-side. Raises TokenError on any
    failure (missing, malformed, expired, revoked, wrong project, bad
    signature). Never trust an unverified token's claims.
    """
    if not _firebase_available:
        raise TokenError("Firebase Admin SDK is not configured on the server")
    if not id_token:
        raise TokenError("Missing ID token")
    try:
        decoded = fb_auth.verify_id_token(id_token, check_revoked=True)
        return decoded
    except Exception as e:
        raise TokenError(str(e))


# ──────────────────────────────────────────────────────────────
# DECORATORS — the actual security boundary
# ──────────────────────────────────────────────────────────────

def _is_api_request() -> bool:
    """Heuristic: JSON/XHR clients get JSON 401/403; page loads get redirected."""
    accept = request.headers.get("Accept", "")
    return request.path.startswith("/api/") or "application/json" in accept


def require_auth(f):
    """
    Requires a valid server-side session established via
    /api/auth/firebase-session. Does NOT re-verify a Firebase ID token on
    every request (that would require the client to resend it every call);
    instead it trusts the Flask session cookie, which is signed with
    SECRET_KEY and was only ever populated after a real server-side token
    verification. This mirrors standard "verify once at login, sign a
    session after" practice.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not flask_session.get("fb_uid"):
            if _is_api_request():
                return jsonify({"error": "Authentication required. Please sign in."}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


def require_role(role: str):
    """Requires require_auth AND that the session's role matches exactly."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not flask_session.get("fb_uid"):
                if _is_api_request():
                    return jsonify({"error": "Authentication required. Please sign in."}), 401
                return redirect("/login")
            if flask_session.get("fb_role") != role:
                if _is_api_request():
                    return jsonify({"error": "Your account does not have permission to access this area."}), 403
                return redirect("/dashboard")
            return f(*args, **kwargs)
        return wrapper
    return decorator


def current_firebase_user() -> dict:
    """Returns the current session's identity dict, or {} if unauthenticated."""
    if not flask_session.get("fb_uid"):
        return {}
    return {
        "uid": flask_session.get("fb_uid"),
        "email": flask_session.get("fb_email"),
        "role": flask_session.get("fb_role"),
        "email_verified": flask_session.get("fb_email_verified", False),
        "display_name": flask_session.get("fb_display_name", ""),
    }


# ──────────────────────────────────────────────────────────────
# PUBLIC WEB CONFIG (safe to expose — these are NOT secrets)
# ──────────────────────────────────────────────────────────────

def get_public_firebase_config() -> dict:
    """
    Firebase Web SDK config values (apiKey, authDomain, etc.) are public
    configuration identifiers, not secrets -- they are designed to be
    embedded in client-side JS. Real access control is enforced by Firebase
    Security Rules / this backend's token verification, not by hiding these
    values. Nothing in this function ever returns a service-account key.
    """
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
    }
