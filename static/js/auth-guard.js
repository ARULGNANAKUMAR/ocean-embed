/**
 * auth-guard.js
 * Lightweight client-side session helper for pages protected by
 * @fbauth.require_auth / @fbauth.require_role("admin") on the backend.
 *
 * IMPORTANT: this script is UX only. The actual security boundary is the
 * server-side decorator on each Flask route. If this script fails to load
 * or is bypassed, protected pages still redirect server-side (401/403),
 * and protected APIs still reject unauthenticated/unauthorized requests.
 *
 * Usage: include after main.js on any page that should show the signed-in
 * user, a working Logout button, and role-aware nav items.
 *   <script src="{{ url_for('static', filename='js/auth-guard.js') }}"></script>
 */
(function () {
  async function fetchSession() {
    try {
      const res = await fetch('/api/auth/firebase-me', { credentials: 'same-origin' });
      if (!res.ok) return null;
      const data = await res.json();
      return data.user || null;
    } catch (e) {
      return null;
    }
  }

  async function logout() {
    try {
      await fetch('/api/auth/firebase-logout', { method: 'POST', credentials: 'same-origin' });
    } catch (e) { /* ignore network errors, still redirect */ }
    window.location.href = '/login';
  }
  window.OceanAuth = window.OceanAuth || {};
  window.OceanAuth.logout = logout;
  window.OceanAuth.getSession = fetchSession;

  document.addEventListener('DOMContentLoaded', async () => {
    const user = await fetchSession();

    // Populate any element with [data-auth-email] / [data-auth-role]
    document.querySelectorAll('[data-auth-email]').forEach(el => {
      el.textContent = user ? user.email : 'Not signed in';
    });
    document.querySelectorAll('[data-auth-role]').forEach(el => {
      el.textContent = user ? user.role : '';
    });

    // Hide elements marked admin-only unless the session role is admin.
    document.querySelectorAll('[data-admin-only]').forEach(el => {
      el.style.display = (user && user.role === 'admin') ? '' : 'none';
    });

    // Wire up any element marked as a logout trigger.
    document.querySelectorAll('[data-auth-logout]').forEach(el => {
      el.addEventListener('click', (e) => { e.preventDefault(); logout(); });
    });
  });
})();
