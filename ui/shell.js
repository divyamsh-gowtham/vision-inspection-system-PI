/* ProInspect HMI - shared shell behaviour.
 *
 * Every screen includes this. It owns the machine header, the navigation
 * rail, the clock, the host bridge, the power menu, the one read-only poll of
 * the counter service's /status, and the panel event log, so those exist
 * identically on each page without being copied into each one.
 *
 * Nothing here writes to the counter service. Every value shown comes from
 * /status or from the kiosk host; anything without a source is shown as
 * not available, never as a stand-in number.
 */
(function () {
  'use strict';

  var App = window.App = {};

  App.esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c];
    });
  };
  function pad(n){ return String(n).padStart(2,'0'); }
  App.pad = pad;
  App.fmtDur = function (s) {
    s = Math.max(0, Math.round(s));
    var h = Math.floor(s/3600), m = Math.floor((s%3600)/60), x = s%60;
    return h ? h+'h '+pad(m)+'m' : (m ? m+'m '+pad(x)+'s' : x+'s');
  };
  function hms(d){ return pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds()); }
  App.hms = hms;

  // ---------------------------------------------------------------- bridge
  // The page cannot touch the system. It posts an action key to the kiosk
  // host, which runs it against a fixed allowlist. See kiosk.py.
  App.hostSend = function (action) {
    var h = window.webkit && window.webkit.messageHandlers
            && window.webkit.messageHandlers.proinspect;
    if (!h) { return false; }
    h.postMessage(JSON.stringify({ action: action }));
    return true;
  };

  // ---------------------------------------------------------------- header
  // Pages declare their nav key and screen name on <body data-nav data-title>.
  function ind(id, label){
    return '<div class="ind off" id="'+id+'"><span class="lamp"></span>' +
           '<span class="k">'+label+'</span><b class="det">&mdash;</b></div>';
  }
  function buildHeader(){
    var h = document.getElementById('hdr');
    if (!h) return;
    var title = document.body.getAttribute('data-title') || '';
    h.innerHTML =
      '<div class="h-brand"><img class="logo" data-logo src="logo-icon.png" alt="ProInspect">PROINSPECT</div>' +
      '<div><span class="k">Station</span><b id="unit">&mdash;</b></div>' +
      '<div class="h-page"><span class="k">Screen</span><b>'+App.esc(title)+'</b></div>' +
      ind('stSvc','Service') + ind('stCam','Camera') + ind('stEng','OCR') + ind('stRun','Run') +
      '<div class="h-sp"></div>' +
      '<div class="h-clock"><b id="hm">--:--:--</b><span class="k" id="dt">&mdash;</span></div>' +
      '<div class="h-state st-conn" id="hState">CONNECTING</div>';
  }

  // The logo is visible from the start; nothing waits on a load event that may
  // already have fired. If the file cannot be loaded it is replaced by a
  // visible marker rather than silently hidden.
  function checkLogos(){
    document.querySelectorAll('img[data-logo]').forEach(function(img){
      function bad(){
        if (!img.parentNode) return;
        var m = document.createElement('span');
        m.className = 'logo-miss';
        m.textContent = 'LOGO MISSING';
        m.title = img.getAttribute('src') + ' could not be loaded';
        img.parentNode.replaceChild(m, img);
        console.error('ProInspect logo not found: ' + img.src);
      }
      // 'error' may already have fired before this script ran, and 'complete'
      // can be true while the image is still decoding, so the final check
      // waits for window load (every image settled) or, because a camera
      // stream can hold that event back, five seconds.
      function verify(){ if (img.complete && !img.naturalWidth) bad(); }
      img.addEventListener('error', bad);
      window.addEventListener('load', verify);
      setTimeout(verify, 5000);
    });
  }

  var MONTHS =['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
  function startClock(){
    var hm = document.getElementById('hm'), dt = document.getElementById('dt');
    if (!hm) return;
    function tick(){
      var d = new Date();
      hm.textContent = hms(d);
      if (dt) dt.textContent = pad(d.getDate())+'-'+MONTHS[d.getMonth()]+'-'+d.getFullYear();
    }
    tick(); setInterval(tick, 1000);
  }

  // ------------------------------------------------------------- nav rail
  // One source for navigation so every screen carries the same rail. Routes
  // are the existing pages; nothing here creates a new one.
  var NAV = [
    { key:'home',     href:'index.html',    label:'OVERVIEW' },
    { key:'inspect',  href:'products.html', label:'INSPECTION', id:'navInspect' },
    { key:'job',      href:'products.html', label:'JOB SETUP' },
    { key:'reports',  href:'reports.html',  label:'REPORTS' },
    { key:'settings', href:'settings.html', label:'SETTINGS' }
  ];
  function buildSide(){
    var side = document.getElementById('side');
    if (!side) return;
    var here = document.body.getAttribute('data-nav') || '';
    side.innerHTML =
      '<nav class="nav" aria-label="Main">' + NAV.map(function(n){
        return '<a href="'+n.href+'"'+(n.id ? ' id="'+n.id+'"' : '') +
               (n.key === here ? ' class="on" aria-current="page"' : '')+'>'+n.label+'</a>';
      }).join('') + '</nav>' +
      '<div class="sfoot"><div class="ver">ProInspect<br><span id="ver">&mdash;</span></div>' +
      '<div class="pwrbtn" data-power tabindex="0" role="button">POWER</div></div>';
  }
  // INSPECTION opens the live screen of the job that is running. With no
  // active run it goes to job setup instead: opening run.html starts a run.
  function updateInspectLink(run){
    var a = document.getElementById('navInspect');
    if (!a) return;
    var href = (run && run.active && run.product)
      ? 'run.html?product=' + encodeURIComponent(run.product) + '&recipe=' + encodeURIComponent(run.recipe || '')
      : 'products.html';
    if (a.getAttribute('href') !== href) a.setAttribute('href', href);
  }

  // ------------------------------------------------------- station status
  // Read-only: one GET of the counter service's own /status. Pages subscribe
  // (App.onStatus) instead of polling on their own.
  var SVC = App.SVC = 'http://127.0.0.1:8080';
  var subs = [], pollMs = 2500, misses = 0;
  App.onStatus = function(fn){ subs.push(fn); };
  App.setPollInterval = function(ms){ pollMs = ms; };

  function setInd(id, cls, det){
    var e = document.getElementById(id);
    if (!e) return;
    if (e.className !== 'ind ' + cls) e.className = 'ind ' + cls;
    var d = e.querySelector('.det');
    if (d && d.textContent !== det) d.textContent = det;
    var t = e.querySelector('.k').textContent + ': ' + det;
    if (e.title !== t) e.title = t;
  }

  // The machine's overall state, from /status alone. Fault outranks setup,
  // setup outranks stopped/running.
  App.machineState = function(s, up){
    if (!up || !s) return { key:'fault', label:'FAULT', why:'Counter service not responding' };
    var cam = s.camera || {}, c = s.counter || {}, run = s.run || {};
    if (!cam.online) return { key:'fault', label:'FAULT', why:'Camera: ' + (cam.error || 'no signal') };
    if (c.learning) return { key:'setup', label:'SETUP', why:'Learning empty belt' };
    if (cam.focusing) return { key:'setup', label:'SETUP', why:'Focus sweep running' };
    if (!c.has_reference) return { key:'setup', label:'SETUP', why:'No belt reference' };
    if (run.active) return { key:'run', label:'RUNNING', why:'' };
    return { key:'stop', label:'STOPPED', why:'No active run' };
  };
  var STATE_CLS = { run:'st-run', stop:'st-stop', setup:'st-setup', fault:'st-fault' };

  function showStatus(s){
    var cam = s.camera || {}, run = s.run || {};
    setInd('stSvc', 'ok', 'ONLINE');
    if (cam.online) setInd('stCam', 'ok', Math.round(cam.fps || 0) + ' FPS');
    else setInd('stCam', 'bad', (cam.error || 'NO SIGNAL').toUpperCase());
    var eng = s.ocr_engine || (s.ocr && s.ocr.engine) || '';
    if (eng === 'gpu') setInd('stEng', 'ok', 'GPU');
    else if (eng) setInd('stEng', 'warn', 'CPU');
    else setInd('stEng', 'off', 'UNKNOWN');
    setInd('stRun', run.active ? 'ok' : 'off', run.active ? 'ACTIVE' : 'STOPPED');
    paintState(App.machineState(s, true));
    updateInspectLink(run);
  }
  function showDown(){
    setInd('stSvc', 'bad', 'NOT RESPONDING');
    setInd('stCam', 'off', 'UNKNOWN');
    setInd('stEng', 'off', 'UNKNOWN');
    setInd('stRun', 'off', 'UNKNOWN');
    paintState(App.machineState(null, false));
  }
  function paintState(st){
    var e = document.getElementById('hState');
    if (!e) return;
    var cls = 'h-state ' + STATE_CLS[st.key];
    if (e.className !== cls) e.className = cls;
    if (e.textContent !== st.label) e.textContent = st.label;
    if (e.title !== st.why) e.title = st.why;
  }
  function pollStatus(){
    fetch(SVC + '/status', { cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(s){
        misses = 0; showStatus(s); observe(s, true);
        subs.forEach(function(fn){ try { fn(s, true); } catch (e) { console.error(e); } });
      })
      .catch(function(){
        // Two misses in a row before calling it down: one slow reply is not an outage.
        if (++misses >= 2) {
          showDown(); observe(null, false);
          subs.forEach(function(fn){ try { fn(null, false); } catch (e) { console.error(e); } });
        }
      })
      .then(function(){ setTimeout(pollStatus, pollMs); });
  }

  // ------------------------------------------------------------ event log
  // The counter service keeps no event history, so this log holds only what
  // this panel itself observed in /status: conditions appearing and clearing,
  // run start/stop, and packs whose final read was a FAIL. It is kept for the
  // browser session only and is NOT a machine record; screens say so.
  var EV_KEY = 'proinspect.panel-events.v1', EV_MAX = 100;
  var ev = { list: [], conds: {}, packs: {}, run: null, seq: 0 };
  try { var saved = JSON.parse(sessionStorage.getItem(EV_KEY) || 'null'); if (saved && saved.list) ev = saved; } catch (e) {}
  var evSubs = [];
  App.events = function(){ return ev.list; };
  App.onEvents = function(fn){ evSubs.push(fn); try { fn(ev.list); } catch (e) {} };
  function evSave(){
    try { sessionStorage.setItem(EV_KEY, JSON.stringify(ev)); } catch (e) {}
    evSubs.forEach(function(fn){ try { fn(ev.list); } catch (e) {} });
  }
  function evAdd(sev, text, cond, at){
    ev.seq += 1;
    var item = { id: ev.seq, t: at || hms(new Date()), sev: sev, text: text, cond: cond || '', state: cond ? 'ACTIVE' : '' };
    ev.list.unshift(item);
    if (ev.list.length > EV_MAX) ev.list.length = EV_MAX;
    return item;
  }
  // A condition raises one event while it holds and marks it cleared when it goes.
  function cond(key, on, sev, text){
    var was = ev.conds[key];
    if (on && !was) { ev.conds[key] = evAdd(sev, text, key).id; return true; }
    if (!on && was) {
      ev.list.forEach(function(it){ if (it.id === was) it.state = 'CLEARED ' + hms(new Date()); });
      delete ev.conds[key]; return true;
    }
    return false;
  }
  function observe(s, up){
    var changed = cond('svc', !up, 'ALARM', 'Counter service not responding');
    if (up && s) {
      var cam = s.camera || {}, c = s.counter || {}, run = s.run || {};
      changed = cond('cam', !cam.online, 'ALARM', 'Camera offline: ' + (cam.error || 'no signal')) || changed;
      changed = cond('belt', !c.has_reference && !c.learning, 'WARN', 'No empty-belt reference: counting unreliable') || changed;
      changed = cond('learn', !!c.learning, 'INFO', 'Learning empty belt') || changed;
      changed = cond('focus', !!cam.focusing, 'INFO', 'Camera focus sweep') || changed;
      var eng = s.ocr_engine || '';
      changed = cond('cpuocr', !!eng && eng !== 'gpu', 'WARN', 'OCR on CPU fallback (Tesseract), GPU engine not loaded') || changed;
      var pend = s.ocr_pending || 0;
      changed = cond('backlog', pend >= 3, 'WARN', 'OCR backlog: reader is ' + pend + ' frames behind') || changed;
      if (ev.run !== null && ev.run !== !!run.active) {
        evAdd('INFO', run.active ? 'Run started: ' + (run.product || '?') + ' / ' + (run.recipe || '?') : 'Run stopped at count ' + (run.count || 0));
        changed = true;
      }
      ev.run = !!run.active;
      // A pack's log entry can still improve while it is the newest one; every
      // older entry is final. Only final FAILs are logged.
      var log = s.ocr_log || [], key = String(run.started_at || 0);
      for (var i = 0; i < log.length - 1; i++) {
        var e = log[i], id = key + ':' + e.n;
        if (ev.packs[id]) continue;
        ev.packs[id] = 1;
        if (e.ok === false) {
          evAdd('WARN', 'Pack #' + e.n + ' FAIL' + (e.batch ? ': read as ' + e.batch : ': no known batch code read'), '', e.at);
          changed = true;
        }
      }
      var ids = Object.keys(ev.packs);
      if (ids.length > 400) ids.slice(0, ids.length - 200).forEach(function(k){ delete ev.packs[k]; });
    }
    if (changed) evSave();
  }

  // Shared rendering of the event table body.
  App.renderEvents = function(tbody, max){
    var list = ev.list.slice(0, max || EV_MAX);
    var html = list.length ? list.map(function(it){
      var cls = it.sev === 'ALARM' ? 'fail' : it.sev === 'WARN' ? 'warn' : 'info';
      var stc = it.state === 'ACTIVE' ? (it.sev === 'ALARM' ? 'fail' : it.sev === 'WARN' ? 'warn' : 'info') : '';
      return '<tr><td class="mono">'+App.esc(it.t)+'</td><td><span class="tag '+cls+'">'+it.sev+'</span></td>' +
             '<td class="ell">'+App.esc(it.text)+'</td>' +
             '<td>'+(it.state ? '<span class="tag '+stc+'">'+App.esc(it.state)+'</span>' : '<span class="muted">&mdash;</span>')+'</td></tr>';
    }).join('') : '<tr><td colspan="4" class="empty">No events observed by this panel.</td></tr>';
    if (tbody.__html !== html) { tbody.__html = html; tbody.innerHTML = html; }
  };

  // ------------------------------------------------- inspection evidence
  // Did a PASS rest on the printed code, or only on product text? The
  // service's verdict can come from keywords such as "SUGAR FREE" printed on
  // the artwork. This does not change the verdict; it only says whether a
  // code field (batch id, lot, date, time) backs it. Confirmed when any of:
  //   - the reader parsed a batch id, lot, date or time on this pack;
  //   - the service's own field match against the recipe sample passed;
  //   - an evidence keyword is code-like (letters AND digits, 5+ characters,
  //     e.g. E29D776361 or "BN KIF2071").
  App.codeConfirmed = function(o){
    o = o || {};
    var f = o.fields || {}, m = o.match || {}, v = o.verdict || {};
    function has(x){ return Array.isArray(x) ? x.length > 0 : !!(x && String(x).trim()); }
    if (has(f.bns) || has(f.bn) || has(f.lots) || has(f.lot) || has(f.dates) || has(f.time)) return true;
    if (m.checked && m.ok) return true;
    return String(v.evidence || '').split(/,\s*/).some(function(tok){
      var a = tok.replace(/[^A-Za-z0-9]/g, '');
      return a.length >= 5 && /[A-Za-z]/.test(a) && /[0-9]/.test(a);
    });
  };

  // ---------------------------------------------------------- power menu
  var PM_HTML =
  '<div class="sheet" id="pwrSheet"><div class="box pm">' +
    '<header id="pmHead">' +
      '<div class="hl"><h3>POWER</h3><span class="sub">STATION <span id="pmUnit">&mdash;</span></span></div>' +
      '<button class="btn sm xbtn" id="pmX" aria-label="Close">X</button>' +
    '</header>' +
    '<div class="err" id="pmErr"></div>' +
    '<div class="opts" id="pmOpts">' +
      opt('idle','STANDBY','Blanks the panel. Inspection keeps running; any touch or key wakes it.') +
      opt('restart','RESTART APPLICATION','Reloads the ProInspect screen only. Inspection keeps running.') +
      '<div class="secthead">ADVANCED &mdash; INTERRUPTS INSPECTION</div>' +
      opt('sleep','SLEEP','Suspends to RAM. The machine stops inspecting and leaves the network.') +
      opt('reboot','RESTART MACHINE','Full reboot. Around a minute before it inspects again.') +
      opt('shutdown','SHUT DOWN','Powers off. Someone has to press the button at the machine to start it again.') +
    '</div>' +
    '<div class="confirm" id="pmConfirm">' +
      '<h4 id="cfTitle">CONFIRM</h4><p id="cfBody"></p>' +
      '<div class="row"><button class="btn" id="cfCancel">CANCEL</button>' +
      '<button class="btn danger" id="cfGo">CONFIRM</button></div>' +
      '<div class="cd" id="cfCd"></div>' +
    '</div>' +
    '<footer id="pmFoot"><button class="btn" id="pmClose">CLOSE</button></footer>' +
  '</div></div>';

  function opt(act, title, desc){
    return '<div class="opt" tabindex="0" data-act="'+act+'">' +
      '<div class="t">'+title+'</div><div class="d">'+desc+'</div><span class="go">&gt;</span></div>';
  }

  // Standby and an app restart are recoverable from the panel itself, so they
  // fire immediately. Anything that strands the machine asks first.
  var NEEDS_CONFIRM = {
    sleep:    { t:'SLEEP THE MACHINE?',
                b:'Inspection stops and the machine leaves the network. Waking it needs a key press at the station.' },
    reboot:   { t:'RESTART THE MACHINE?',
                b:'Any inspection in progress is lost. The machine is unavailable for about a minute.' },
    shutdown: { t:'SHUT THE MACHINE DOWN?',
                b:'Any inspection in progress is lost. Someone has to press the power button at the machine to start it again.' }
  };

  function initPower(){
    document.body.insertAdjacentHTML('beforeend', PM_HTML);
    var pwr = document.getElementById('pwrSheet');
    var pmOpts = document.getElementById('pmOpts');
    var pmHead = document.getElementById('pmHead');
    var pmFoot = document.getElementById('pmFoot');
    var pmConf = document.getElementById('pmConfirm');
    var pmErr  = document.getElementById('pmErr');
    var timer  = null;

    function showErr(m){ pmErr.textContent = m; pmErr.classList.add('on'); }
    function clearErr(){ pmErr.classList.remove('on'); pmErr.textContent = ''; }

    function back(){
      if (timer) { clearInterval(timer); timer = null; }
      pmConf.classList.remove('on');
      pmConf.querySelector('.row').style.display = '';
      pmOpts.classList.remove('off'); pmHead.classList.remove('off'); pmFoot.classList.remove('off');
    }
    function open(){
      clearErr(); back();
      var u = document.getElementById('unit');
      document.getElementById('pmUnit').textContent = u ? u.textContent : '—';
      pwr.classList.add('on');
    }
    function close(){ pwr.classList.remove('on'); if (timer){ clearInterval(timer); timer = null; } }
    App.openPower = open;

    function ask(action){
      var c = NEEDS_CONFIRM[action];
      document.getElementById('cfTitle').textContent = c.t;
      document.getElementById('cfBody').textContent  = c.b;
      pmOpts.classList.add('off'); pmHead.classList.add('off'); pmFoot.classList.add('off');
      pmConf.classList.add('on');
      document.getElementById('cfGo').onclick = function(){ fire(action); };

      // An armed confirm left open by accident disarms itself rather than
      // waiting for someone to lean on the screen.
      var left = 15, cd = document.getElementById('cfCd');
      cd.textContent = 'CANCELS AUTOMATICALLY IN ' + left + ' S';
      timer = setInterval(function(){
        left--;
        if (left <= 0) { back(); return; }
        cd.textContent = 'CANCELS AUTOMATICALLY IN ' + left + ' S';
      }, 1000);
    }

    function fire(action){
      if (timer) { clearInterval(timer); timer = null; }
      if (!App.hostSend(action)) { back(); showErr('Host bridge unavailable.'); return; }
      document.getElementById('cfTitle').textContent = 'WORKING';
      document.getElementById('cfBody').textContent  = 'The machine is carrying out the request.';
      document.getElementById('cfCd').textContent    = '';
      pmConf.querySelector('.row').style.display = 'none';
    }

    pmOpts.querySelectorAll('.opt').forEach(function(o){
      function go(){
        var a = o.dataset.act;
        clearErr();
        if (NEEDS_CONFIRM[a]) { ask(a); }
        else if (App.hostSend(a)) { close(); }
        else { showErr('Host bridge unavailable.'); }
      }
      o.addEventListener('click', go);
      o.addEventListener('keydown', function(e){
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
      });
    });

    document.getElementById('pmClose').addEventListener('click', close);
    document.getElementById('pmX').addEventListener('click', close);
    document.getElementById('cfCancel').addEventListener('click', back);
    pwr.addEventListener('click', function(e){ if (e.target === pwr) close(); });

    document.querySelectorAll('[data-power]').forEach(function(el){
      el.addEventListener('click', open);
      el.addEventListener('keydown', function(e){
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      });
    });

    // The host calls this only when an action did NOT take effect, so the
    // panel never sits on "Working" after a silent failure.
    window.onHostResult = function(r){
      if (r && r.ok === false) {
        back(); showErr((r.action || 'Action') + ' failed: ' + (r.detail || 'unknown error'));
        pwr.classList.add('on');
      }
    };
    document.addEventListener('keydown', function(e){
      if (e.key === 'Escape') close();
    });
  }

  // Injected by the kiosk host once the page has loaded.
  window.setUnit = function(n){
    var e = document.getElementById('unit');
    if (e) e.textContent = n;
  };
  window.setVersion = function(v){
    var e = document.getElementById('ver');
    if (e) e.textContent = 'v' + v;
  };

  buildHeader();
  buildSide();
  checkLogos();
  pollStatus();

  document.addEventListener('DOMContentLoaded', function(){
    startClock(); initPower();
    // An appliance screen never shows a context menu or a drag ghost.
    document.addEventListener('contextmenu', function(e){ e.preventDefault(); });
    document.addEventListener('dragstart',   function(e){ e.preventDefault(); });
  });
})();
