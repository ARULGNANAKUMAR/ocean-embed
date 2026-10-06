# OceanVerse AI — Firebase Authentication & Security

This document describes the authentication and authorization architecture
added on top of the existing OceanVerse AI platform. It complements
`reports/firebase_security_audit.md` (the Phase 1 audit of what existed
before this work).

---

## 1. Authentication Architecture

```
Browser (Firebase Web SDK)
    │  email/password sign-in, registration, verification, reset
    ▼
Firebase Authentication  (Google-managed identity provider)
    │  issues short-lived Firebase ID token (JWT)
    ▼
POST /api/auth/firebase-session   { idToken }
    │
    ▼
firebase_auth.verify_id_token()   -- Firebase Admin SDK, server-side
    │  verifies signature, expiry, revocation against the real project
    ▼
firebase_auth.get_or_create_role() -- looks up / creates firebase_roles row
    │  role decided server-side (see Section 4)
    ▼
Flask session cookie set: fb_uid, fb_email, fb_role, fb_email_verified
    (signed with SECRET_KEY from .env; HttpOnly; SameSite=Lax;
     Secure when FLASK_ENV=production)
    │
    ▼
Every subsequent request -> @fbauth.require_auth / @fbauth.require_role("admin")
    reads the signed session cookie -- never a client-supplied header/body/
    query value -- to decide access.
```

The Firebase ID token itself is only sent to the backend **once**, at
sign-in. After that, the app relies on the server-signed session cookie,
the same pattern used by most session-based web apps layered on top of a
token-based identity provider. This avoids resending a token on every one
of the ~90 API calls the frontend makes while keeping the "verify signature
against Firebase" step happening for real, server-side, exactly once per
login.

## 2. Firebase Configuration

Firebase Web SDK config values (`FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`,
`FIREBASE_PROJECT_ID`, `FIREBASE_STORAGE_BUCKET`,
`FIREBASE_MESSAGING_SENDER_ID`, `FIREBASE_APP_ID`) are **public identifiers**,
not secrets — Firebase is designed for these to be embedded in client-side
JavaScript. The backend exposes them at `GET /api/firebase-config` so the
login page can initialize the SDK without hardcoding values into a template.

The **Firebase Admin SDK service-account key** is a real secret. It is only
ever referenced by a filesystem path (`FIREBASE_SERVICE_ACCOUNT_PATH` in
`.env`), is never sent to the frontend, is excluded from git via
`.gitignore` (`secrets/`, `*.serviceaccount.json`,
`firebase-service-account*.json`), and is only read server-side inside
`firebase_auth.py` at process startup.

## 3. Environment Variables

See `.env.example` for the full template. Summary:

| Variable | Secret? | Purpose |
|---|---|---|
| `SECRET_KEY` | Yes | Signs the Flask session cookie |
| `FLASK_ENV` | No | `production` enables `Secure` cookies |
| `FIREBASE_API_KEY` etc. (6 vars) | No | Public Web SDK config |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | Points to a secret | Path to Admin SDK credentials, file itself gitignored |
| `ADMIN_EMAIL` | No (but sensitive) | The email auto-granted admin on first login |

None of these are committed. `.env` itself is gitignored; only
`.env.example` (with blank values) is tracked.

## 4. Admin Role Architecture

There is **no way for a client to request or set its own role**. The rule,
enforced entirely server-side in `firebase_auth.get_or_create_role()`:

- On a UID's first successful login, if its **Firebase-verified** email
  (from the verified ID token, not anything the client sent as a form
  field) case-insensitively matches `ADMIN_EMAIL`, the UID's row in the
  local `firebase_roles` table is created with `role = "admin"`.
  Otherwise it is created with `role = "user"`.
- On every later login, the *existing* stored role is reused — a UID is
  never re-evaluated against `ADMIN_EMAIL` after creation, so a user
  cannot retroactively gain admin by somehow presenting a different email
  string later (verified test: `test_role_persists_across_logins_even_if_email_changes_case`).
- The **only** other way a UID becomes admin is an existing admin calling
  `PATCH /api/admin/firebase-users/<uid>/role`, which is itself gated by
  `@fbauth.require_role("admin")`.

This satisfies the brief's requirement to avoid the insecure
`if email == "admin@x.com": show_admin` pattern — the check happens once,
server-side, against a token Firebase itself verified, and is then a stored
fact rather than a per-request client claim.

## 5. User Role Architecture

Regular authenticated users get `role = "user"`. There are currently two
roles (`user`, `admin`); `firebase_auth.VALID_ROLES` is the single place
that would need extending for finer-grained roles later.

## 6. Backend Token Verification

`firebase_auth.verify_id_token()` calls
`firebase_admin.auth.verify_id_token(id_token, check_revoked=True)`, which:

- Verifies the JWT signature against Google's public keys for the
  configured Firebase project.
- Rejects expired tokens.
- Rejects tokens for a different Firebase project.
- Rejects revoked tokens (e.g. after a password change or explicit
  revocation) because `check_revoked=True` makes an extra call to check the
  token wasn't issued before the user's `tokensValidAfterTime`.

Any failure raises `firebase_auth.TokenError`, which `/api/auth/firebase-session`
turns into a `401` with a generic, non-revealing message
("Your session could not be verified. Please sign in again.").

## 7. Protected Routes

**Admin-only (`@fbauth.require_role("admin")`):**
`/admin`, `/database`, `/training`, `/api-management`, `/model-registry`,
`POST /api/admin/upload`, `POST /api/admin/retrain`, `GET /api/admin/models`,
`GET /api/admin/history`, `GET /api/admin/users`,
`PATCH /api/admin/users/<id>/role`, `PATCH /api/admin/firebase-users/<uid>/role`,
`POST /api/registry/datasets`, `DELETE /api/registry/datasets/<id>`,
`POST /api/registry/models/activate/<id>`, `POST /api/registry/models/rollback/<id>`,
`GET/POST /api/apikeys`, `POST /api/apikeys/test`, `PATCH /api/apikeys/<service>`,
`POST /api/scheduler/run`, `PATCH /api/scheduler/<job_name>`.

**Any authenticated user (`@fbauth.require_auth`):**
`/settings`, `/profile`, `/downloads`, `/activity`, `/reports`,
`GET/POST /api/settings`, `GET/POST/DELETE /api/favorites`,
`GET/POST/DELETE/PATCH /api/bookmarks`.

**Intentionally public** (per Phase 13 — scientific visualization should
not be unnecessarily restricted): `/`, `/earth`, `/dashboard`, `/heatwave`,
`/gismap`, `/map`, `/twin`, `/voice`, `/about`, `/login`, and the read-only
scientific/prediction/GIS-layer GET APIs.

See `reports/firebase_security_audit.md` Section 3 for the full
before/after route inventory.

## 8. File Upload Security

`POST /api/admin/upload` (admin-only) now:

- Runs the filename through `werkzeug.utils.secure_filename()`, stripping
  path separators and unsafe characters (blocks `../../etc/passwd`-style
  traversal).
- Rejects any extension outside `{.nc, .nc4, .csv, .json, .txt}`.
- Resolves the final path and confirms it stays inside the configured raw
  data directory.
- If a file with the sanitized name already exists, appends a random
  suffix instead of silently overwriting it.

## 9. Firebase Security Rules

This implementation uses **Firebase Authentication only** — no Firestore or
Firebase Storage. Per the brief's Phase 14 guidance, no Firestore/Storage
security rules were introduced, since the app's actual data continues to
live in the existing SQLite database, which is protected by the Flask-layer
`require_auth`/`require_role` decorators described above instead.

## 10. Security Testing

`tests/test_firebase_security.py` — isolated unit tests against
`firebase_auth.py` using a minimal Flask app (runs without the full
scientific-computing dependency stack). `tests/test_all.py`,
`tests/test_phase1.py`, and `tests/test_phase3_twin.py` were updated with
`flask_client_user` / `flask_client_admin` fixtures and now exercise:

1. Unauthenticated request to a protected page → `302` redirect to `/login`
2. Unauthenticated request to a protected API → `401`
3. Authenticated non-admin hitting an admin route/page → `403` (API) / `302` (page)
4. Authenticated admin hitting an admin route/page → `200`
5. Invalid/garbage Firebase ID token → rejected (`TokenError`)
6. First-login role assignment only honors `ADMIN_EMAIL`, and a UID cannot
   retroactively self-promote by re-presenting a different email
7. Invalid role names rejected by `set_user_role`
8. Unsafe upload filenames (traversal, bad extension) rejected

See the final development report for exact pass/fail results and the
sandbox limitation that prevented running the full existing suite in this
environment (no network access to install `pytest`/`torch`/`xarray`/etc.).

## 11. Deployment Checklist

1. In the Firebase Console: create a project, enable **Authentication →
   Sign-in method → Email/Password**.
2. Project Settings → General → add a Web app, copy the six config values
   into `.env` (`FIREBASE_API_KEY` etc.).
3. Project Settings → Service Accounts → **Generate new private key**, save
   the JSON somewhere outside version control (e.g. `secrets/` — already
   gitignored), and point `FIREBASE_SERVICE_ACCOUNT_PATH` at it.
4. Set `ADMIN_EMAIL` to the real administrator's email.
5. Set a strong random `SECRET_KEY` (`python -c "import secrets; print(secrets.token_hex(32))"`).
6. Set `FLASK_ENV=production` on the real deployment so session cookies get
   the `Secure` flag (requires serving over HTTPS).
7. `pip install -r requirements.txt` (now includes `firebase-admin`).
8. Sign in once with the `ADMIN_EMAIL` account to bootstrap the first admin
   row in `firebase_roles`; promote further admins afterward via
   `PATCH /api/admin/firebase-users/<uid>/role`.
9. Restart the Flask app; confirm `/api/firebase-config` returns non-empty
   values before relying on login.

## 12. Secrets Management

- No secret is hardcoded in source anymore (`SECRET_KEY` and the old static
  admin token were both removed from `app.py`/`admin.py`'s call sites).
- `.env` and any `*serviceaccount*.json` are gitignored.
- The Admin SDK credential path is the only server-side secret this
  integration introduces; it never crosses into a template, a JS bundle, or
  an API response.

## 13. Known Limitations

- **`admin.py`'s `_ADMIN_TOKEN` constant and `verify_admin_token()` function
  still exist in the file** for backward compatibility with any external
  tooling that imported them directly, but **no route in `app.py` calls
  `verify_admin_token()` anymore** — every admin route now uses
  `@fbauth.require_role("admin")`. Removing the function outright was left
  for a follow-up pass since some existing tests reference it as a mock
  target without exercising it over HTTP; deleting it is a one-line change
  but wasn't done here to minimize risk of an unrelated import error.
- **The legacy username/password auth system in `ocean_platform.py`
  (`authenticate`, `create_session`, `validate_session`, the `users`/
  `sessions` tables) is still present and its routes
  (`/api/auth/login`, `/api/auth/register`, etc.) still work.** It is not
  wired into any `require_auth`/`require_role` check anymore — Firebase is
  the sole path to a real session now — but it hasn't been deleted, per the
  brief's "make the smallest safe changes necessary" instruction. It should
  be removed in a follow-up once you're confident no client code still
  depends on it.
- **Schema mismatch, not fully resolved:** `favorite_locations`,
  `map_bookmarks`, and `user_settings` have an `INTEGER user_id` column
  tied to the legacy `users.id`. Firebase UIDs are strings. SQLite's type
  affinity means storing a string UID in that column works without error,
  but it's not a clean design — a proper fix is a migration that re-keys
  these tables to a `TEXT uid` column referencing `firebase_roles.uid`, as
  recommended in the Phase 1 audit (Section 8, point 2). Functionality
  works today; this is a cleanliness/foreign-key-integrity debt, not a
  security hole.
- **CSRF:** the session cookie uses `SameSite=Lax`, which mitigates most
  cross-site POST forgery for a same-origin SPA-style app, but no explicit
  CSRF token was added to state-changing forms. Low risk given `SameSite=Lax`
  and the app's all-JSON API surface (no traditional `<form>` posts), but
  worth revisiting before a public multi-origin deployment.
- **Rate limiting:** no rate limiting was added to `/api/auth/firebase-session`
  or other endpoints. Firebase's own client SDK has built-in abuse
  protection for sign-in attempts, but the exchange endpoint itself has no
  additional throttling. Consider `flask-limiter` if this becomes a
  public-internet deployment.
- **Not tested in this environment:** the actual Firebase project
  integration (no real Firebase project credentials were available here),
  and the full existing pytest suite could not be executed end-to-end due
  to the sandbox having no network access to install the project's heavy
  dependencies (`torch`, `xarray`, `flask_socketio`, `pytest`, etc.). See
  the final development report for what *was* verified.
