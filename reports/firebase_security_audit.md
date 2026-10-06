# OceanEmbed / OceanVerse AI — Firebase Security Audit (Phase 1)

**Scope:** Audit of the existing authentication, authorization, and route-protection
architecture prior to any Firebase integration. No code has been changed as part of
this document — it is a read-only inspection of `OceanVerse_AI_v4_Final/`.

---

## 1. Project Entry Point

- Flask app instantiated in `app.py` (`app = Flask(__name__, ...)`), wrapped with
  `flask_socketio.SocketIO` for live updates.
- Run via `python app.py` → calls `run_all_phases()` (loads `config.json`, runs the
  scientific pipeline phases) → `socketio.run(app, host, port, debug=False, allow_unsafe_werkzeug=True)`.
- `app.config["SECRET_KEY"] = "sih-ocean-intelligence-2025"` — **hardcoded, committed
  secret**, used to sign the Flask session cookie. Anyone with this value can forge
  session cookies.
- `app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024` — 500MB global upload cap;
  no per-endpoint limit.

## 2. There Are Two Independent, Unconnected Auth Systems

### 2a. User authentication — custom SQLite/session system (`ocean_platform.py`)

- Table `users(id, username, email, password_hash, role, display_name, avatar_seed,
  created_at, last_login, is_active)`.
- Password hashing: `hashlib.sha256(f"{salt}{password}".encode())` — **SHA-256 is not
  a password hash** (no work factor, trivially brute-forced/GPU-crackable vs. bcrypt/
  argon2/scrypt). Salt handling should be confirmed per-user (worth checking
  `_hash_password` call sites for a fixed vs. random salt).
- `authenticate(conn, username, password)` in `ocean_platform.py:307` — compares hash,
  returns user row.
- `create_session(conn, user_id)` — creates a random token in `sessions` table.
- `validate_session(conn, token)` — looks up token, presumably checks `expires_at`.
- Flask route glue in `app.py`:
  - `POST /api/auth/login` — sets `flask_session["token"]` and `flask_session["user_id"]`
    on success.
  - `POST /api/auth/logout` — invalidates session, clears `flask_session`.
  - `GET /api/auth/me` — returns current user via `_current_user()`.
  - `POST /api/auth/register` — creates a user with `role="guest"` (self-serve
    registration; no email verification).
- `_current_user()` (app.py:614) reads `flask_session["token"]`, calls
  `plat_module.validate_session`. **This is the only per-request identity check for
  the "user" system**, and it is applied inconsistently — many endpoints
  (`/api/favorites`, `/api/bookmarks`, `/api/settings`, `/api/registry/datasets`,
  `/api/apikeys`, `/api/scheduler`, etc.) call `_current_user()` but **fall back to a
  default `uid = 1` or `uid = 0` if there is no session**, rather than rejecting the
  request. Effectively these endpoints work fully unauthenticated.
- **`/login` and `/dashboard` and every other page route in `app.py` render templates
  unconditionally** — there is no `@login_required`-style decorator anywhere in the
  codebase. Any page (`/dashboard`, `/admin`, `/settings`, `/training`,
  `/model-registry`, etc.) is reachable by a fully anonymous browser; only the data
  the JS subsequently fetches may or may not be gated (and per above, mostly isn't).

### 2b. Admin authorization — static hardcoded token (`admin.py`)

- `admin.py:24`: `_ADMIN_TOKEN = hashlib.sha256(b"sih_ocean_admin_2025").hexdigest()`
  — a **fixed, source-committed secret**. Hashing a hardcoded string doesn't add
  security; the effective password is `sih_ocean_admin_2025`, visible to anyone with
  repo access (and guessable/brute-forceable since it's a SHA-256 of a short known-ish
  string once leaked).
- `verify_admin_token(token)` (admin.py:27): `return token == _ADMIN_TOKEN`. Plain
  string equality (not constant-time — minor timing-attack surface, low priority next
  to the bigger issue that the token is static and shared).
- Every admin API route (`/api/admin/upload`, `/api/admin/retrain`,
  `/api/admin/models`, `/api/admin/history`, `/api/admin/users`,
  `PATCH /api/admin/users/<id>/role`) takes `token` **as a URL query parameter**:
  `request.args.get("token", "")`. Query-string tokens land in server access logs,
  browser history, and Referer headers — this is a second, independent problem from
  the token being static.
- **There is no linkage between this admin token and the `users`/`sessions` system.**
  Knowing the one hardcoded string grants admin API access regardless of which (if
  any) user is logged in. There is no concept of "this specific user account is an
  admin" enforced anywhere server-side — `role` exists as a column on `users` and is
  editable via `PATCH /api/admin/users/<id>/role`, but nothing reads `user.role` to
  gate any route. The `role` column is currently decorative.
- The `/admin` **page** route itself (`app.py:104`, `page_admin()`) has **no
  protection at all** — it just renders `admin.html`. Only the *data-fetching* admin
  API calls check the token; the page shell is fully public. A logged-out visitor can
  browse to `/admin` and see the UI (buttons will presumably fail when they hit the
  token-gated APIs, but the page itself, and any admin.html-embedded data, is exposed).

## 3. Authorization Summary — What's Actually Protected

| Category | Mechanism | Verdict |
|---|---|---|
| Admin API routes (`/api/admin/*`) | Static shared token in query string | Weak — single shared secret, no user identity, logged in URLs |
| `/admin` page | None | **Unprotected** |
| Regular authenticated API routes (`/api/favorites`, `/api/settings`, `/api/bookmarks`, `/api/registry/*`, `/api/apikeys`, `/api/scheduler`, `/api/notifications`, `/api/activity`) | `_current_user()` with silent fallback to a default user id | **Unprotected in practice** — no session required |
| Page routes (`/dashboard`, `/prediction`, `/heatwave`, `/gismap`, `/settings`, `/profile`, `/downloads`, `/database`, `/training`, `/activity`, `/api-management`, `/model-registry`, `/reports`, `/map`, `/twin`) | None | **Unprotected** |
| Public scientific read APIs (`/api/datasets`, `/api/summary`, `/api/prediction/latest`, `/api/heatwave`, map layer APIs, digital twin APIs) | None (intentionally public-ish) | Fine to stay public/read-only, but currently indistinguishable from the routes above that *should* require auth |
| `POST /api/registry/datasets`, `DELETE /api/registry/datasets/<id>`, model activate/rollback, dataset upload, API key vault writes | None beyond whatever `_current_user()` provides (i.e., effectively none) | **Unprotected destructive/administrative operations** |

Net effect: almost the entire application — including operations that write to the
database, delete records, change model state, and manage API keys — is reachable
without any credential today. The one exception is the four/five `token=`-gated admin
API endpoints, which are gated by a single hardcoded shared secret rather than by
identity.

## 4. Frontend

- `templates/login.html` exists as the current login page (design to be preserved).
- No JS auth-state guard was found wired to the other page templates (each page
  fetches its own data client-side; there's no shared "redirect to /login if not
  authenticated" script found in `static/js`).
- No evidence of role-aware nav hiding tied to a real identity check (would need to
  confirm in `static/js`, but given the backend has no consistent identity check to
  hide behind, frontend-only hiding — if present — would not be a security boundary
  regardless).

## 5. Database / Storage

- SQLite (`data/database/ocean_platform.db` by default, path driven by
  `_state["db_path"]`). Connection helper `_db()` in `app.py` opens a fresh
  connection per request and calls `plat_module.init_phase1_tables` each time
  (idempotent `CREATE TABLE IF NOT EXISTS`, so cheap but not free).
- `sessions` table stores raw tokens (not hashed) with `expires_at` — no code path
  observed that purges expired sessions, and no rotation on privilege change.
- Passwords: SHA-256, no verified per-user salt strategy confirmed from a
  static read alone — needs a direct read of `_hash_password`/`create_user` call
  sites to confirm whether the salt is random-per-user (stored) or a fixed constant.
  Either way, SHA-256 without a deliberately slow KDF is inadequate for password
  storage and should not be reused once Firebase Authentication is introduced (see
  Phase 2 requirement: passwords will no longer live in this DB at all).

## 6. File Upload Security (Admin Dataset Upload)

- `POST /api/admin/upload` (app.py:~310): after token check, does
  `file.save(os.path.join(raw_dir, file.filename))` — **`file.filename` is used
  directly, unsanitized**. This is a path-traversal and overwrite risk (e.g.
  `../../app.py` or an absolute-path-like filename depending on Werkzeug version
  behavior) and does not restrict extensions or validate NetCDF structure before
  writing to disk. This needs `werkzeug.utils.secure_filename` (or an equivalent
  allow-list/rename strategy) plus extension/content validation before Phase 11 work
  is considered complete.

## 7. Error Handling / Debug Exposure

- `socketio.run(..., debug=False, ...)` — debug mode is off for the production entry
  point, good. No global Flask error handler was found customizing 500 responses, so
  default Flask/Werkzeug error pages apply; should be confirmed they don't leak
  tracebacks in whatever hosting environment this is deployed to (Flask's own default
  is safe when `debug=False`, so this is likely fine, but worth an explicit test in
  Phase 17/20).
- CORS is wide open only on the SocketIO layer: `cors_allowed_origins="*"`. No
  `flask-cors` config was found for the regular HTTP API, so default same-origin
  browser behavior likely applies there, but the SocketIO `*` should be narrowed once
  a real deployment origin is known.
- No `.gitignore` file exists in the project root at all — meaning nothing is
  currently excluded from version control, including whatever `.env` file might be
  added later unless one is created now.

## 8. Recommended Firebase Integration Points

1. **Replace the entire custom `users`/`sessions`/password-hash system** with Firebase
   Authentication for identity (email/password, verification, reset). Firebase becomes
   the sole source of truth for "who is this person and is their email verified,"
   consistent with Phase 2's directive not to store passwords app-side.
2. **Keep the existing SQLite `users`-adjacent tables** (`favorite_locations`,
   `map_bookmarks`, `user_settings`, `activity_log`, etc.) but re-key them by Firebase
   UID instead of the current internal `users.id` — this preserves all existing
   feature functionality (favorites, bookmarks, settings) while removing the parallel
   password store. A thin `role_map` table (Firebase UID → role) is the simplest
   server-side role source if custom claims aren't used immediately.
3. **Introduce a single `require_auth()` / `require_role("admin")` decorator pair**
   that verifies the Firebase ID token from the `Authorization: Bearer <token>` header
   using the Firebase Admin SDK, and apply it to every route currently listed as
   "Unprotected" in Section 3 above — both page routes (redirect to `/login` if
   missing) and the destructive/administrative API routes (401/403 JSON response).
4. **Retire the static `_ADMIN_TOKEN` entirely** and replace all `token=` query-param
   checks in `app.py`'s admin endpoints and `admin.py` with `require_role("admin")`
   backed by Firebase custom claims (or the `role_map` table), set through a one-time
   server-side bootstrap script keyed to `ADMIN_EMAIL` from the new `.env`, not through
   any user-modifiable request field.
5. **Fix `SECRET_KEY`** to be loaded from environment configuration rather than
   hardcoded, since Flask session signing still matters for CSRF-token storage /
   any remaining lightweight server-side session use even after Firebase owns login.
6. **Sanitize upload filenames** (`secure_filename` + extension allow-list) as part of
   the same pass, since file upload is one of the "destructive endpoint" categories
   Phase 6/11 require protecting regardless of the auth backend.

## 9. Non-Goals / What Not to Touch

Per the task brief, the scientific pipeline (`pipeline.py`, `ai_engine.py`,
`evaluation.py`, `preprocessing.py`, `digital_twin.py`, `gis_engine.py`,
`argo_validation.py`, ARGO data structure under `data/raw/argo/`, and all existing
templates' visual design) are out of scope for this security pass and were not
modified in producing this audit.

---

**Status:** Phase 1 (audit) complete. No code has been modified yet. Given the
number of routes involved (~25+ page/API routes need new protection) and that this
touches every layer (DB schema, backend auth, every admin route, file upload, and the
login frontend), the recommendation is to implement Phases 2–9 (Firebase wiring,
`.env`, RBAC, backend route protection, frontend auth guard) as a single reviewable
pass next, then Phases 10–17 (hardening, upload security, logging, tests) as a
follow-up pass, so each stage can actually be tested against the running app rather
than claimed sight-unseen.
