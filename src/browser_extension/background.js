// ProactiveClaw Browser Extension — MV3 Service Worker
// Manages chrome.debugger attachment and CDP command execution.
// WebSocket lives in the offscreen document (offscreen.js) for persistence.

let attachedTabId = null;

// --- Snapshot script (inlined) ---
const SNAPSHOT_SCRIPT = `
(function() {
  const MAX_LEN = 8000;
  const MARGIN = 200; // px above/below viewport to include
  const INTERACTIVE = new Set([
    'A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'DETAILS', 'SUMMARY'
  ]);
  const INTERACTIVE_ROLES = new Set([
    'button', 'link', 'checkbox', 'radio', 'tab', 'menuitem',
    'option', 'switch', 'textbox', 'combobox', 'searchbox'
  ]);

  window.__pclaw_refs = {};
  let refCounter = 0;
  let output = '';

  const vpTop = window.scrollY - MARGIN;
  const vpBottom = window.scrollY + window.innerHeight + MARGIN;

  function inViewport(el) {
    const rect = el.getBoundingClientRect();
    const absTop = rect.top + window.scrollY;
    const absBottom = rect.bottom + window.scrollY;
    // Element overlaps the extended viewport
    return absBottom >= vpTop && absTop <= vpBottom;
  }

  function isHidden(el) {
    if (!el.offsetParent && el.tagName !== 'BODY' && el.tagName !== 'HTML') return true;
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return true;
    return false;
  }

  function isInteractive(el) {
    if (INTERACTIVE.has(el.tagName)) return true;
    const role = el.getAttribute('role');
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.hasAttribute('onclick') || el.hasAttribute('tabindex')) return true;
    if (el.contentEditable === 'true') return true;
    return false;
  }

  function getLabel(el) {
    const tag = el.tagName.toLowerCase();
    let label = '';

    if (tag === 'a') {
      label = 'link';
      const text = (el.textContent || '').trim().slice(0, 60);
      if (text) label += ' "' + text + '"';
      if (el.href) label += ' -> ' + el.href.slice(0, 80);
    } else if (tag === 'button' || el.getAttribute('role') === 'button') {
      label = 'button';
      const text = (el.textContent || '').trim().slice(0, 60);
      if (text) label += ' "' + text + '"';
    } else if (tag === 'input') {
      const type = el.type || 'text';
      label = 'input[' + type + ']';
      if (el.placeholder) label += ' placeholder="' + el.placeholder.slice(0, 40) + '"';
      if (el.value) label += ' value="' + el.value.slice(0, 40) + '"';
      if (el.name) label += ' name="' + el.name + '"';
    } else if (tag === 'select') {
      label = 'select';
      if (el.name) label += ' name="' + el.name + '"';
      const selected = el.options[el.selectedIndex];
      if (selected) label += ' selected="' + selected.text.slice(0, 30) + '"';
    } else if (tag === 'textarea') {
      label = 'textarea';
      if (el.name) label += ' name="' + el.name + '"';
      if (el.value) label += ' value="' + el.value.slice(0, 40) + '"';
    } else {
      label = tag;
      const text = (el.textContent || '').trim().slice(0, 60);
      if (text) label += ' "' + text + '"';
    }

    return label;
  }

  function walk(el, depth) {
    if (output.length > MAX_LEN) return;
    if (!el || el.nodeType !== 1) return;
    if (isHidden(el)) return;

    // Skip entire subtrees that are fully outside the viewport
    if (!inViewport(el)) return;

    const indent = '  '.repeat(Math.min(depth, 10));

    if (isInteractive(el)) {
      refCounter++;
      window.__pclaw_refs[refCounter] = el;
      const label = getLabel(el);
      output += indent + '[ref:' + refCounter + '] ' + label + '\\n';
      if (el.tagName === 'A' || el.tagName === 'BUTTON') return;
    } else {
      const tag = el.tagName.toLowerCase();
      if (['h1','h2','h3','h4','h5','h6'].includes(tag)) {
        const text = (el.textContent || '').trim().slice(0, 100);
        if (text) output += indent + tag + ': ' + text + '\\n';
        return;
      }
      if (tag === 'p' || tag === 'li' || tag === 'td' || tag === 'th' || tag === 'label' || tag === 'span') {
        const directText = Array.from(el.childNodes)
          .filter(n => n.nodeType === 3)
          .map(n => n.textContent.trim())
          .join(' ')
          .slice(0, 100);
        if (directText && directText.length > 2) {
          output += indent + tag + ': ' + directText + '\\n';
        }
      }
      if (tag === 'img') {
        const alt = el.alt || el.title || '';
        output += indent + 'img' + (alt ? ' alt="' + alt.slice(0, 60) + '"' : '') + '\\n';
        return;
      }
    }

    for (const child of el.children) {
      if (output.length > MAX_LEN) break;
      walk(child, depth + 1);
    }
  }

  const scrollPct = Math.round(100 * window.scrollY / Math.max(1, document.body.scrollHeight - window.innerHeight));
  output += 'Page: ' + document.title + '\\n';
  output += 'URL: ' + location.href + '\\n';
  output += 'Scroll: ' + scrollPct + '% (' + Math.round(window.scrollY) + 'px / ' + document.body.scrollHeight + 'px)\\n';
  output += '---\\n';

  walk(document.body, 0);

  if (output.length > MAX_LEN) {
    output = output.slice(0, MAX_LEN) + '\\n... [truncated — scroll down for more content]';
  }

  return output;
})()
`;

// --- Offscreen document management ---

async function ensureOffscreen() {
  const existing = await chrome.offscreen.hasDocument();
  if (!existing) {
    await chrome.offscreen.createDocument({
      url: "offscreen.html",
      reasons: ["WEB_RTC"],  // closest valid reason for persistent connection
      justification: "Maintain WebSocket connection to ProactiveClaw server",
    });
    console.log("[pclaw] Offscreen document created");
  }
}

// --- Restore attachedTabId from storage on service worker restart ---

chrome.storage.session.get("attachedTabId", (data) => {
  if (data.attachedTabId) {
    attachedTabId = data.attachedTabId;
    console.log("[pclaw] Restored attachedTabId:", attachedTabId);
  }
});

// --- CDP command handling (called from offscreen.js via chrome.runtime.onMessage) ---

function executeCDP(method, params) {
  return new Promise((resolve, reject) => {
    chrome.debugger.sendCommand(
      { tabId: attachedTabId },
      method,
      params || {},
      (res) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(res);
        }
      }
    );
  });
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== "cdp_command") return false;

  const { id, method, params } = msg;

  if (attachedTabId === null) {
    sendResponse({
      type: "cdp_result",
      id,
      error: "No tab attached — click the extension icon on a tab first",
    });
    return false;
  }

  // Handle async: return true to keep sendResponse channel open
  (async () => {
    try {
      let result;
      if (method === "snapshot") {
        result = await executeCDP("Runtime.evaluate", {
          expression: SNAPSHOT_SCRIPT,
          returnByValue: true,
        });
        sendResponse({
          type: "cdp_result",
          id,
          result: { snapshot: result.result.value },
        });
      } else {
        result = await executeCDP(method, params);
        sendResponse({ type: "cdp_result", id, result });
      }
    } catch (err) {
      sendResponse({
        type: "cdp_result",
        id,
        error: err.message || String(err),
      });
    }
  })();

  return true; // keeps the message channel open for async sendResponse
});

// --- Tab attach/detach ---

function enableCDPDomains(tabId) {
  chrome.debugger.sendCommand({ tabId }, "DOM.enable", {});
  chrome.debugger.sendCommand({ tabId }, "Page.enable", {});
  chrome.debugger.sendCommand({ tabId }, "Runtime.enable", {});
}

function attachTab(tabId) {
  chrome.debugger.attach({ tabId }, "1.3", () => {
    if (chrome.runtime.lastError) {
      console.error("[pclaw] Attach failed:", chrome.runtime.lastError.message);
      return;
    }
    attachedTabId = tabId;
    chrome.storage.session.set({ attachedTabId: tabId });
    enableCDPDomains(tabId);
    chrome.action.setBadgeText({ text: "ON", tabId });
    chrome.action.setBadgeBackgroundColor({ color: "#4CAF50", tabId });
    console.log("[pclaw] Attached to tab", tabId);
  });
}

function detachTab(tabId) {
  chrome.debugger.detach({ tabId }, () => {
    if (chrome.runtime.lastError) {
      // Already detached, ignore
    }
    attachedTabId = null;
    chrome.storage.session.set({ attachedTabId: null });
    chrome.action.setBadgeText({ text: "", tabId });
    console.log("[pclaw] Detached from tab", tabId);
  });
}

// --- Extension icon click: toggle attach ---

chrome.action.onClicked.addListener((tab) => {
  if (attachedTabId === tab.id) {
    detachTab(tab.id);
  } else {
    if (attachedTabId !== null) {
      detachTab(attachedTabId);
    }
    attachTab(tab.id);
  }
});

// --- Handle debugger detach (user closed tab, etc.) ---

chrome.debugger.onDetach.addListener((source, reason) => {
  if (source.tabId === attachedTabId) {
    attachedTabId = null;
    chrome.storage.session.set({ attachedTabId: null });
    chrome.action.setBadgeText({ text: "" });
    console.log("[pclaw] Debugger detached:", reason);
  }
});

// --- Init: ensure offscreen document is running ---
ensureOffscreen();
