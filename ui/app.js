/* ProInspect HMI - shared shell behaviour.
 *
 * Every screen includes this. It owns the clock, the host bridge, and the
 * power menu, so those exist identically on each page without being copied
 * into each one.
 */
(function () {
  'use strict';

  var App = window.App = {};

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

  // ---------------------------------------------------------------- clock
  var DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  var MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  function pad(n){ return String(n).padStart(2,'0'); }
  App.pad = pad;

  function startClock(){
    var hm = document.getElementById('hm'), dt = document.getElementById('dt');
    if (!hm) return;
    function tick(){
      var d = new Date();
      hm.textContent = pad(d.getHours()) + ':' + pad(d.getMinutes());
      if (dt) dt.textContent = DAYS[d.getDay()]+' '+d.getDate()+' '+MONTHS[d.getMonth()];
    }
    tick(); setInterval(tick, 1000);
  }

  // ---------------------------------------------------------- power menu
  var PM_HTML =
  '<div class="sheet" id="pwrSheet"><div class="box pm">' +
    '<header id="pmHead">' +
      '<button class="xbtn" id="pmX" aria-label="Close">' +
        '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>' +
      '<div><h3>Power</h3><div class="sub">Station <span id="pmUnit">&mdash;</span></div></div>' +
    '</header>' +
    '<div class="err" id="pmErr"></div>' +
    '<div class="opts" id="pmOpts">' +
      opt('idle','Standby','Blanks the panel. Inspection keeps running; any touch or key wakes it.',
          '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/>') +
      opt('restart','Restart application','Reloads ProInspect only. Fastest way to clear a UI fault.',
          '<path d="M3 12a9 9 0 0 1 15.5-6.2M21 12a9 9 0 0 1-15.5 6.2"/><path d="M18 3v4h-4M6 21v-4h4"/>') +
      '<div class="secthead">Advanced &mdash; interrupts inspection</div>' +
      opt('sleep','Sleep','Suspends to RAM. The machine stops inspecting and leaves the network.',
          '<path d="M2 12h5l2-4 3 8 2-4h8"/>') +
      opt('reboot','Restart machine','Full reboot. Around a minute before it inspects again.',
          '<path d="M12 3v9"/><path d="M6.3 6.3a8 8 0 1 0 11.4 0"/>') +
      opt('shutdown','Shut down','Powers off. Someone has to press the button at the machine to start it again.',
          '<path d="M12 3v9"/><path d="M6.3 6.3a8 8 0 1 0 11.4 0"/>') +
    '</div>' +
    '<div class="confirm" id="pmConfirm">' +
      '<h4 id="cfTitle">Confirm</h4><p id="cfBody"></p>' +
      '<div class="row"><button class="btn" id="cfCancel">Cancel</button>' +
      '<button class="btn danger" id="cfGo">Confirm</button></div>' +
      '<div class="cd" id="cfCd"></div>' +
    '</div>' +
    '<footer id="pmFoot"><button class="btn" id="pmClose">Close</button></footer>' +
  '</div></div>';

  function opt(act, title, desc, svg){
    return '<div class="opt" tabindex="0" data-act="'+act+'">' +
      '<span class="ic"><svg viewBox="0 0 24 24">'+svg+'</svg></span>' +
      '<div><div class="t">'+title+'</div><div class="d">'+desc+'</div></div>' +
      '<span class="chev"><svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg></span>' +
    '</div>';
  }

  // Standby and an app restart are recoverable from the panel itself, so they
  // fire immediately. Anything that strands the machine asks first.
  var NEEDS_CONFIRM = {
    sleep:    { t:'Sleep the machine?',
                b:'Inspection stops and the machine leaves the network. Waking it needs a key press at the station.' },
    reboot:   { t:'Restart the machine?',
                b:'Any inspection in progress is lost. The machine is unavailable for about a minute.' },
    shutdown: { t:'Shut the machine down?',
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
      document.getElementById('pmUnit').textContent =
        u ? u.textContent.replace(/^Station /,'') : '—';
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
      cd.textContent = 'Cancels automatically in ' + left + 's';
      timer = setInterval(function(){
        left--;
        if (left <= 0) { back(); return; }
        cd.textContent = 'Cancels automatically in ' + left + 's';
      }, 1000);
    }

    function fire(action){
      if (timer) { clearInterval(timer); timer = null; }
      if (!App.hostSend(action)) { back(); showErr('Host bridge unavailable.'); return; }
      document.getElementById('cfTitle').textContent = 'Working…';
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
    // panel never sits on "Working..." after a silent failure.
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

  // ------------------------------------------------------------- shell
  function initLogo(){
    var l = document.getElementById('logo');
    if (!l) return;
    // The image can finish loading before this script runs, in which case
    // 'load' has already fired; show it if it is already decoded too.
    function show(){ if (l.naturalWidth) l.classList.add('ready'); }
    l.addEventListener('load', show);
    l.addEventListener('error', function(){ console.error('ProInspect logo not found: ' + l.src); });
    if (l.complete) show();
  }

  // Injected by the kiosk host once the page has loaded.
  window.setUnit = function(n){
    var e = document.getElementById('unit');
    if (e) e.textContent = 'Station ' + n;
  };
  window.setVersion = function(v){
    var e = document.getElementById('ver');
    if (e) e.textContent = 'v' + v;
  };

  document.addEventListener('DOMContentLoaded', function(){
    initLogo(); startClock(); initPower();
    // An appliance screen never shows a context menu or a drag ghost.
    document.addEventListener('contextmenu', function(e){ e.preventDefault(); });
    document.addEventListener('dragstart',   function(e){ e.preventDefault(); });
  });
})();
