// DOM-walking snapshot script — injected via Runtime.evaluate
// Returns a text tree with [ref:N] markers on interactive elements
// Stores ref→element mapping in window.__pclaw_refs

const SNAPSHOT_SCRIPT = `
(function() {
  const MAX_LEN = 8000;
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

  function isVisible(el) {
    if (!el.offsetParent && el.tagName !== 'BODY' && el.tagName !== 'HTML') return false;
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    return true;
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
    if (!isVisible(el)) return;

    const indent = '  '.repeat(Math.min(depth, 10));

    if (isInteractive(el)) {
      refCounter++;
      window.__pclaw_refs[refCounter] = el;
      const label = getLabel(el);
      output += indent + '[ref:' + refCounter + '] ' + label + '\\n';
      // Don't recurse into interactive children of interactive parents for clarity
      if (el.tagName === 'A' || el.tagName === 'BUTTON') return;
    } else {
      // For structural elements, show headings and text blocks
      const tag = el.tagName.toLowerCase();
      if (['h1','h2','h3','h4','h5','h6'].includes(tag)) {
        const text = (el.textContent || '').trim().slice(0, 100);
        if (text) output += indent + tag + ': ' + text + '\\n';
        return;
      }
      if (tag === 'p' || tag === 'li' || tag === 'td' || tag === 'th' || tag === 'label' || tag === 'span') {
        // Only show if it has direct text (not just children)
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

  // Build page header
  output += 'Page: ' + document.title + '\\n';
  output += 'URL: ' + location.href + '\\n';
  output += '---\\n';

  walk(document.body, 0);

  if (output.length > MAX_LEN) {
    output = output.slice(0, MAX_LEN) + '\\n... [truncated — scroll down for more content]';
  }

  return output;
})()
`;
