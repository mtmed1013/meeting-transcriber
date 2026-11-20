/*
 * Meeting Transcriber shared frontend helpers. Plain script, no build step, offline.
 *
 * Load it as a classic script before the page script:
 *   <script src="assets/app.js"></script>
 * It exposes one global, window.MeetingApp, and (when loaded under CommonJS, e.g. node tests)
 * module.exports with the same object.
 *
 * Contents:
 *   Tauri bridge   isTauri(), invoke(cmd, args), listen(event, handler)
 *                  Uses the Tauri v2 globals (withGlobalTauri). Outside Tauri, invoke() rejects with
 *                  an Error whose code is "TAURI_UNAVAILABLE" so the page can show a calm message,
 *                  and listen() resolves to a no-op unlisten function.
 *   Theme          applyTheme("dark" | "light" | "system")
 *   Formatters     formatBytes(bytes), formatDuration(seconds)
 *   DOM            el(tag, attrs, ...children): builds nodes with textContent/attributes only,
 *                  never innerHTML, so untrusted strings (transcripts, rule text) cannot inject markup.
 */
(function (root) {
  "use strict";

  // ---------- Tauri bridge ----------

  function tauriGlobal() {
    var t = root && root.__TAURI__;
    return t && typeof t === "object" ? t : null;
  }

  function isTauri() {
    var t = tauriGlobal();
    return !!(t && t.core && typeof t.core.invoke === "function");
  }

  function unavailableError(what) {
    var err = new Error("Tauri is not available: " + what + " requires the desktop app.");
    err.code = "TAURI_UNAVAILABLE";
    return err;
  }

  /**
   * Calls a Tauri command. Resolves with the command result; rejects with an Error (code
   * "TAURI_UNAVAILABLE") when the page is opened outside the Tauri shell.
   */
  function invoke(cmd, args) {
    if (!isTauri()) {
      return Promise.reject(unavailableError("invoke(" + String(cmd) + ")"));
    }
    try {
      return Promise.resolve(root.__TAURI__.core.invoke(cmd, args));
    } catch (err) {
      return Promise.reject(err);
    }
  }

  /**
   * Subscribes to a Tauri event. Resolves with an unlisten function. Outside Tauri it resolves
   * with a no-op, so callers can always call the returned function on teardown.
   */
  function listen(event, handler) {
    var t = tauriGlobal();
    if (!t || !t.event || typeof t.event.listen !== "function") {
      return Promise.resolve(function noop() {});
    }
    return Promise.resolve(t.event.listen(event, handler));
  }

  // ---------- Theme ----------

  /**
   * Sets the colour theme on <html data-theme>. "dark" and "light" force a palette; "system"
   * removes the attribute so the prefers-color-scheme rule in ui.css decides.
   * Returns the mode that was applied.
   */
  function applyTheme(mode) {
    var html = root.document && root.document.documentElement;
    if (mode !== "dark" && mode !== "light" && mode !== "system") {
      throw new RangeError('applyTheme: mode must be "dark", "light" or "system", got ' + String(mode));
    }
    if (html) {
      if (mode === "system") html.removeAttribute("data-theme");
      else html.setAttribute("data-theme", mode);
    }
    return mode;
  }

  // ---------- Formatters ----------

  var BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];

  /**
   * Human-readable size using 1024-based steps (matching the app's GB figures).
   * formatBytes(0) -> "0 B", formatBytes(1536) -> "1.5 KB", formatBytes(3 * 1024 ** 3) -> "3.0 GB".
   * Non-finite or negative input returns "—".
   */
  function formatBytes(bytes) {
    var n = Number(bytes);
    if (!isFinite(n) || n < 0) return "—";
    if (n < 1024) return Math.floor(n) + " B";
    var i = 0;
    while (n >= 1024 && i < BYTE_UNITS.length - 1) {
      n /= 1024;
      i += 1;
    }
    return n.toFixed(1) + " " + BYTE_UNITS[i];
  }

  function pad2(n) {
    return n < 10 ? "0" + n : String(n);
  }

  /**
   * Formats a duration given in seconds as m:ss, or h:mm:ss from one hour up.
   * Fractions are truncated, so 59.9 -> "0:59". formatDuration(0) -> "0:00",
   * formatDuration(3725) -> "1:02:05". Non-finite or negative input returns "—".
   */
  function formatDuration(seconds) {
    var n = Number(seconds);
    if (!isFinite(n) || n < 0) return "—";
    var total = Math.floor(n);
    var h = Math.floor(total / 3600);
    var m = Math.floor((total % 3600) / 60);
    var s = total % 60;
    if (h > 0) return h + ":" + pad2(m) + ":" + pad2(s);
    return m + ":" + pad2(s);
  }

  // ---------- DOM ----------

  var FORBIDDEN_ATTRS = { innerhtml: true, outerhtml: true };
  // Attributes that must be set as DOM properties rather than attributes to behave correctly.
  var PROPERTY_ATTRS = { value: true, checked: true, disabled: true, selected: true, hidden: true };

  function appendChild(parent, child) {
    if (child === null || child === undefined || child === false || child === true) return;
    if (Array.isArray(child)) {
      child.forEach(function (c) { appendChild(parent, c); });
      return;
    }
    if (typeof child === "object" && typeof child.nodeType === "number") {
      parent.appendChild(child);
      return;
    }
    parent.appendChild(root.document.createTextNode(String(child)));
  }

  /**
   * Builds an element without innerHTML.
   *   el("button", { class: "ui-btn ui-btn--primary", onclick: fn, disabled: true }, "Save")
   *   el("div", { class: "ui-row" }, el("span", { class: "ui-label" }, label), [child, child2])
   * Attrs: "class" (or "className") sets the class; "style" accepts an object of CSS properties;
   * "on<event>" with a function adds a listener; true sets a boolean attribute, false/null/undefined
   * skip it. Strings, numbers, nodes, arrays and null/false children are accepted; strings always
   * become text nodes. Attribute names "innerHTML"/"outerHTML" are rejected.
   */
  function el(tag, attrs) {
    if (typeof tag !== "string" || !/^[a-z][a-z0-9-]*$/i.test(tag)) {
      throw new TypeError("el: invalid tag name " + String(tag));
    }
    var node = root.document.createElement(tag);
    var props = attrs && typeof attrs === "object" && !Array.isArray(attrs) && typeof attrs.nodeType !== "number" ? attrs : null;
    var children = Array.prototype.slice.call(arguments, props || attrs === null || attrs === undefined ? 2 : 1);

    if (props) {
      Object.keys(props).forEach(function (key) {
        var value = props[key];
        var lower = key.toLowerCase();
        if (FORBIDDEN_ATTRS[lower]) {
          throw new TypeError("el: attribute " + key + " is not allowed; use text children instead");
        }
        if (value === null || value === undefined || value === false) return;
        if (key === "class" || key === "className") {
          node.className = String(value);
        } else if (key === "style" && typeof value === "object") {
          Object.keys(value).forEach(function (prop) { node.style[prop] = value[prop]; });
        } else if (lower.indexOf("on") === 0 && typeof value === "function") {
          node.addEventListener(lower.slice(2), value);
        } else if (PROPERTY_ATTRS[lower] && key in node) {
          node[key] = value === true ? true : value;
        } else if (value === true) {
          node.setAttribute(key, "");
        } else {
          node.setAttribute(key, String(value));
        }
      });
    }
    children.forEach(function (child) { appendChild(node, child); });
    return node;
  }

  var api = {
    isTauri: isTauri,
    invoke: invoke,
    listen: listen,
    applyTheme: applyTheme,
    formatBytes: formatBytes,
    formatDuration: formatDuration,
    el: el
  };

  root.MeetingApp = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
