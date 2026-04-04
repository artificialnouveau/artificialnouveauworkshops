/**
 * auth.js — Simple password gate for workshop pages.
 *
 * Include this script at the top of any page to require a password.
 * Uses sessionStorage so students only enter it once per browser session.
 *
 * Change the password below before each workshop.
 */

(function () {
  const PASSWORD = 'artificialnouveau2026';

  // Skip on localhost for development
  if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') return;

  // Already authenticated this session
  if (sessionStorage.getItem('workshop_auth') === 'true') return;

  // Hide page content
  document.documentElement.style.display = 'none';

  document.addEventListener('DOMContentLoaded', function () {
    document.documentElement.style.display = '';
    document.body.innerHTML = `
      <div id="auth-gate" style="
        position:fixed; inset:0; z-index:99999;
        background:#0a0a0a;
        display:flex; align-items:center; justify-content:center;
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      ">
        <div style="text-align:center; max-width:360px; padding:20px;">
          <h1 style="color:#e0e0e0; font-size:1.3rem; margin-bottom:8px;">Workshop Access</h1>
          <p style="color:#777; font-size:0.85rem; margin-bottom:24px;">Enter the workshop password to continue.</p>
          <input type="password" id="auth-input" placeholder="Password" autofocus style="
            display:block; width:100%; padding:12px 14px;
            background:#141414; border:1px solid #333; border-radius:8px;
            color:#e0e0e0; font-family:'SF Mono','Fira Code',monospace; font-size:0.9rem;
            text-align:center; margin-bottom:12px; outline:none;
          ">
          <button id="auth-btn" style="
            display:block; width:100%; padding:12px;
            background:#4a9eff; border:none; border-radius:8px;
            color:#000; font-family:'SF Mono','Fira Code',monospace; font-size:0.9rem;
            font-weight:600; cursor:pointer;
          ">Enter</button>
          <p id="auth-error" style="color:#ff4a4a; font-size:0.8rem; margin-top:12px; display:none;">Incorrect password</p>
        </div>
      </div>
    `;

    const input = document.getElementById('auth-input');
    const btn = document.getElementById('auth-btn');
    const error = document.getElementById('auth-error');

    function tryAuth() {
      if (input.value === PASSWORD) {
        sessionStorage.setItem('workshop_auth', 'true');
        location.reload();
      } else {
        error.style.display = 'block';
        input.style.borderColor = '#ff4a4a';
        input.value = '';
        input.focus();
      }
    }

    btn.addEventListener('click', tryAuth);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') tryAuth();
    });
  });
})();
