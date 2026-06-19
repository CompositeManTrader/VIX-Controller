(function(){
'use strict';
var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function $(s,c){return (c||document).querySelector(s)}
function $$(s,c){return Array.prototype.slice.call((c||document).querySelectorAll(s))}
function lerp(a,b,t){return a+(b-a)*t}

/* ============ malla de volatilidad (canvas) ============ */
function createMesh(canvas,opt){
  if(!canvas) return;
  opt = opt||{};
  var alpha = opt.alpha||0.34, labels = !!opt.labels;
  var ctx = canvas.getContext('2d');
  var W=0,H=0,cols=52,rows=30;
  var codes=['VX F6','VX G6','VX H6','VX J6','VX K6','VX M6','VX N6','VX Q6'];
  var mx=0,my=0,tx=0,ty=0;
  function resize(){
    var dpr=Math.min(window.devicePixelRatio||1,2);
    W=canvas.clientWidth; H=canvas.clientHeight;
    canvas.width=W*dpr; canvas.height=H*dpr;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    cols=W<720?34:52; rows=W<720?20:30;
  }
  resize();
  window.addEventListener('resize',resize);
  if(labels && !REDUCED){
    window.addEventListener('pointermove',function(e){
      tx=e.clientX/Math.max(W,1)-.5; ty=e.clientY/Math.max(H,1)-.5;
    },{passive:true});
  }
  function draw(t){
    ctx.clearRect(0,0,W,H);
    mx=lerp(mx,tx,.05); my=lerp(my,ty,.05);
    var ang=mx*.1, ca=Math.cos(ang), sa=Math.sin(ang);
    var cx=W/2, cy=H*.58, spanX=Math.max(W,680)*.62, spanY=spanX*.5, amp=46*(1+my*.18);
    var P=[],i,j,u,v,x,y,z,xr,yr,p;
    for(i=0;i<cols;i++){
      P.push([]); u=i/(cols-1)*2-1;
      for(j=0;j<rows;j++){
        v=j/(rows-1)*2-1;
        z=(Math.sin(u*3.1+t*.62)*.5+Math.cos(v*2.3-t*.41)*.34+Math.sin((u+v)*4.2+t*.27)*.22)*amp;
        x=u*spanX; y=v*spanY;
        xr=x*ca-y*sa; yr=x*sa+y*ca;
        P[i].push({x:cx+(xr-yr)*.72, y:cy+(xr+yr)*.3-z});
      }
    }
    ctx.lineWidth=1;
    ctx.strokeStyle='rgba(201,163,92,'+alpha+')';
    for(j=0;j<rows;j++){
      ctx.beginPath();
      for(i=0;i<cols;i++){p=P[i][j]; if(i)ctx.lineTo(p.x,p.y); else ctx.moveTo(p.x,p.y)}
      ctx.stroke();
    }
    ctx.strokeStyle='rgba(201,163,92,'+alpha*.55+')';
    for(i=0;i<cols;i+=2){
      ctx.beginPath();
      for(j=0;j<rows;j++){p=P[i][j]; if(j)ctx.lineTo(p.x,p.y); else ctx.moveTo(p.x,p.y)}
      ctx.stroke();
    }
    if(labels){
      var jr=Math.floor(rows*.36), k, idx, last, r;
      ctx.font='400 10px "JetBrains Mono",monospace'; ctx.textAlign='center';
      for(k=0;k<codes.length;k++){
        idx=Math.floor((k+1.2)/(codes.length+1.6)*cols);
        p=P[idx][jr]; last=(k===codes.length-1);
        if(last){
          r=2.4+Math.sin(t*2.2)*.8;
          ctx.fillStyle='rgba(201,163,92,.95)';
          ctx.shadowColor='rgba(201,163,92,.55)'; ctx.shadowBlur=8;
          ctx.beginPath(); ctx.arc(p.x,p.y,r,0,7); ctx.fill();
          ctx.shadowBlur=0; ctx.fillStyle='rgba(201,163,92,.6)';
        }else{
          ctx.fillStyle='rgba(201,163,92,.9)';
          ctx.beginPath(); ctx.arc(p.x,p.y,2,0,7); ctx.fill();
          ctx.fillStyle='rgba(201,163,92,.5)';
        }
        ctx.fillText(codes[k],p.x,p.y-12);
      }
    }
  }
  if(REDUCED){ draw(2.4); window.addEventListener('resize',function(){draw(2.4)}); return; }
  var t0=performance.now();
  (function loop(now){
    if(!document.hidden) draw((now-t0)/1000*.85);
    requestAnimationFrame(loop);
  })(t0);
}
createMesh($('#meshCanvas'),{alpha:.34,labels:true});
createMesh($('#meshCanvas2'),{alpha:.14,labels:false});

/* ============ halo de candela ============ */
(function(){
  var halo=$('#halo');
  if(!halo||REDUCED||window.matchMedia('(hover: none)').matches) return;
  var hx=innerWidth/2,hy=innerHeight/2,gx=hx,gy=hy;
  window.addEventListener('pointermove',function(e){gx=e.clientX;gy=e.clientY},{passive:true});
  (function loop(){
    hx=lerp(hx,gx,.06); hy=lerp(hy,gy,.06);
    halo.style.transform='translate('+(hx-390)+'px,'+(hy-390)+'px)';
    requestAnimationFrame(loop);
  })();
})();

/* ============ progreso + nav ============ */
(function(){
  var bar=$('#progressBar'), nav=$('#nav');
  if(!nav)return;
  function onScroll(){
    var h=document.documentElement;
    if(bar){var p=h.scrollTop/Math.max(h.scrollHeight-h.clientHeight,1);bar.style.transform='scaleX('+p+')'}
    nav.classList.toggle('scrolled',h.scrollTop>40);
  }
  window.addEventListener('scroll',onScroll,{passive:true});
  onScroll();
})();

/* ============ menú móvil ============ */
(function(){
  var nav=$('#nav'), burger=$('#navBurger'), links=$('#navLinks');
  if(!burger||!links)return;
  function close(){
    nav.classList.remove('open');
    burger.setAttribute('aria-expanded','false');
    burger.setAttribute('aria-label','Abrir menú');
  }
  burger.addEventListener('click',function(){
    var open=nav.classList.toggle('open');
    burger.setAttribute('aria-expanded',open?'true':'false');
    burger.setAttribute('aria-label',open?'Cerrar menú':'Abrir menú');
  });
  links.addEventListener('click',function(e){if(e.target.closest('a'))close()});
  document.addEventListener('keydown',function(e){if(e.key==='Escape')close()});
})();

/* ============ reveal ============ */
(function(){
  var io=new IntersectionObserver(function(es){
    es.forEach(function(en){
      if(en.isIntersecting){en.target.classList.add('in');io.unobserve(en.target)}
    });
  },{threshold:.18});
  $$('.reveal').forEach(function(el){io.observe(el)});
  if(REDUCED)$$('.reveal').forEach(function(el){el.classList.add('in')});
})();

/* ============ decode ============ */
var DIG='0123456789', LET='ABCDEFGHIKLMNOPRSTUVXZ';
function runDecode(el){
  var fin=el.dataset.final!==undefined?el.dataset.final:el.textContent;
  el.dataset.final=fin;
  if(REDUCED){el.textContent=fin;return}
  var start=performance.now(), dur=800;
  (function step(now){
    var p=Math.min((now-start)/dur,1);
    var n=Math.floor(p*fin.length), out=fin.slice(0,n), i, c;
    for(i=n;i<fin.length;i++){
      c=fin[i];
      if(/[0-9]/.test(c)) out+=DIG[Math.random()*10|0];
      else if(/[A-ZÁÉÍÓÚÑ]/i.test(c)) out+=LET[Math.random()*LET.length|0];
      else out+=c;
    }
    el.textContent=out;
    if(p<1)requestAnimationFrame(step); else el.textContent=fin;
  })(start);
}
(function(){
  var io=new IntersectionObserver(function(es){
    es.forEach(function(en){
      if(en.isIntersecting){runDecode(en.target);io.unobserve(en.target)}
    });
  },{threshold:.6});
  $$('[data-decode]').forEach(function(el){io.observe(el)});
})();

/* ============ contadores ============ */
(function(){
  function runCount(el){
    var target=parseFloat(el.dataset.count);
    if(REDUCED){el.textContent=String(target);return}
    var dur=1900, start=performance.now();
    (function step(now){
      var p=Math.min((now-start)/dur,1);
      var e=p===1?1:1-Math.pow(2,-10*p);
      el.textContent=String(Math.round(target*e));
      if(p<1)requestAnimationFrame(step);
    })(start);
  }
  var io=new IntersectionObserver(function(es){
    es.forEach(function(en){
      if(en.isIntersecting){runCount(en.target);io.unobserve(en.target)}
    });
  },{threshold:.6});
  $$('[data-count]').forEach(function(el){io.observe(el)});
})();

/* ============ reloj UTC ============ */
(function(){
  var els=$$('[data-clock]');
  if(!els.length)return;
  function pad(n){return n<10?'0'+n:String(n)}
  function tick(){
    var d=new Date();
    var s=pad(d.getUTCHours())+':'+pad(d.getUTCMinutes())+':'+pad(d.getUTCSeconds())+' UTC';
    els.forEach(function(el){el.textContent=s});
  }
  tick(); setInterval(tick,1000);
})();

/* ============ telemetría oscilante ============ */
(function(){
  var defs={
    vix:{base:16.42,amp:.7,dec:2,pre:''},
    vix2:{base:16.42,amp:.7,dec:2,pre:''},
    vvix:{base:91.78,amp:2.6,dec:2,pre:''},
    vvix2:{base:91.78,amp:2.6,dec:2,pre:''},
    vrp:{base:3.42,amp:.4,dec:2,pre:'+'},
    vrp2:{base:3.42,amp:.4,dec:2,pre:'+',suf:' pts'},
    slope:{base:.84,amp:.16,dec:2,pre:'+'},
    slope2:{base:.84,amp:.16,dec:2,pre:'+'},
    slope3:{base:.84,amp:.16,dec:2,pre:'+'},
    pct:{base:42,amp:5,dec:0,pre:'P '},
    corr:{base:.31,amp:.08,dec:2,pre:''}
  };
  var els=$$('[data-tel]');
  if(!els.length)return;
  function update(first){
    els.forEach(function(el){
      var d=defs[el.dataset.tel]; if(!d)return;
      if(d.cur===undefined)d.cur=d.base;
      d.cur+=(Math.random()-.5)*d.amp*.45;              // deriva suave
      if(d.cur>d.base+d.amp)d.cur=d.base+d.amp;
      if(d.cur<d.base-d.amp)d.cur=d.base-d.amp;
      var txt=d.pre+d.cur.toFixed(d.dec)+(d.suf||'');
      el.dataset.final=txt;
      if(first)runDecode(el); else el.textContent=txt;   // decode solo al revelar
    });
  }
  if(REDUCED){update(true);return}
  setTimeout(function(){update(true)},1200);
  setInterval(function(){update(false)},16000);
})();

/* ============ gráfico VRP (tesis) ============ */
(function(){
  var svg=$('#vrpChart'); if(!svg)return;
  var W=560,H=300,pad=16,n=160,i,x;
  var vmin=4,vmax=62;
  function Y(v){return H-pad-(v-vmin)/(vmax-vmin)*(H-2*pad)}
  function X(i){return pad+i/(n-1)*(W-2*pad)}
  function spike(x,x0,w,a){var d=x-x0;return a*Math.exp(-(d*d)/(w*w))}
  var imp=[],rea=[];
  for(i=0;i<n;i++){
    x=i/(n-1);
    var base=18.5+Math.sin(x*5.5)*2.4+Math.sin(x*11+1.3)*1.4+Math.sin(x*23+.6)*.7;
    var iv=base+spike(x,.16,.03,24)+spike(x,.72,.025,30)+spike(x,.45,.02,6);
    var rv=base-3.6-Math.sin(x*9+2)*1.1-Math.sin(x*17+.4)*.8
            +spike(x,.165,.025,30)+spike(x,.725,.02,38)+spike(x,.452,.018,7);
    imp.push(iv); rea.push(Math.max(rv,5));
  }
  function path(arr){
    var d='M'+X(0)+' '+Y(arr[0]).toFixed(1);
    for(i=1;i<n;i++)d+='L'+X(i).toFixed(1)+' '+Y(arr[i]).toFixed(1);
    return d;
  }
  var area=path(imp);
  for(i=n-1;i>=0;i--)area+='L'+X(i).toFixed(1)+' '+Y(rea[i]).toFixed(1);
  area+='Z';
  var lastX=X(n-1).toFixed(1), lastY=Y(imp[n-1]).toFixed(1);
  var pulseAnim=REDUCED?'':'<animate attributeName="opacity" values="1;.35;1" dur="2.4s" repeatCount="indefinite"/>';
  svg.innerHTML=
    '<defs><linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">'+
    '<stop offset="0" stop-color="#C9A35C" stop-opacity=".16"/>'+
    '<stop offset="1" stop-color="#C9A35C" stop-opacity=".02"/></linearGradient></defs>'+
    '<line x1="'+pad+'" y1="'+(H-pad)+'" x2="'+(W-pad)+'" y2="'+(H-pad)+'" stroke="rgba(201,163,92,.25)" stroke-width="1"/>'+
    '<path d="'+area+'" fill="url(#gA)"/>'+
    '<path d="'+path(rea)+'" fill="none" stroke="rgba(232,227,213,.7)" stroke-width=".9"/>'+
    '<path d="'+path(imp)+'" fill="none" stroke="#C9A35C" stroke-width="1.4"/>'+
    '<circle cx="'+lastX+'" cy="'+lastY+'" r="3" fill="#66E0C2">'+pulseAnim+'</circle>'+
    '<text x="'+(W-pad)+'" y="'+(Y(imp[n-1])-12)+'" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="9" letter-spacing="2" fill="#66E0C2">DIFERENCIAL +3.4</text>'+
    '<text x="'+pad+'" y="'+(H-pad+0)+'" dy="-6" font-family="JetBrains Mono,monospace" font-size="9" fill="#79808C">2006</text>'+
    '<text x="'+(W/2)+'" y="'+(H-pad)+'" dy="-6" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="#79808C">2016</text>'+
    '<text x="'+(W-pad)+'" y="'+(H-pad)+'" dy="-6" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="9" fill="#79808C">2026</text>';
})();

/* ============ sparkline VRP-01 ============ */
(function(){
  var c=$('#spark1'); if(!c)return;
  var ctx=c.getContext('2d'), pts=[], n=90, i, s=7;
  function rnd(){s=(s*16807)%2147483647;return s/2147483647}
  var v=.5;
  for(i=0;i<n;i++){v+= (rnd()-.46)*.09; v=Math.max(.08,Math.min(.95,v)); pts.push(v)}
  var frac=REDUCED?1:0, started=false;
  function draw(){
    var dpr=Math.min(window.devicePixelRatio||1,2);
    var W=c.clientWidth,H=c.clientHeight;
    c.width=W*dpr;c.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,W,H);
    var m=Math.max(2,Math.floor(n*frac)), j;
    ctx.beginPath();
    for(j=0;j<m;j++){
      var x=j/(n-1)*W, y=H-3-pts[j]*(H-8);
      if(j)ctx.lineTo(x,y); else ctx.moveTo(x,y);
    }
    ctx.strokeStyle='rgba(201,163,92,.85)';ctx.lineWidth=1.2;ctx.stroke();
    ctx.lineTo((m-1)/(n-1)*W,H);ctx.lineTo(0,H);ctx.closePath();
    ctx.fillStyle='rgba(201,163,92,.07)';ctx.fill();
    if(frac>=1&&!REDUCED){
      var lx=W,ly=H-3-pts[n-1]*(H-8);
      ctx.fillStyle='rgba(102,224,194,.9)';
      ctx.beginPath();ctx.arc(lx-2,ly,2.4,0,7);ctx.fill();
    }
  }
  draw();
  window.addEventListener('resize',draw);
  var io=new IntersectionObserver(function(es){
    es.forEach(function(en){
      if(en.isIntersecting&&!started){
        started=true;io.unobserve(c);
        if(REDUCED){draw();return}
        var t0=performance.now();
        (function step(now){
          frac=Math.min((now-t0)/1400,1);
          draw();
          if(frac<1)requestAnimationFrame(step);
        })(t0);
      }
    });
  },{threshold:.4});
  io.observe(c);
})();

/* ============ banda de regímenes RGM-02 ============ */
(function(){
  var band=$('#band2'); if(!band)return;
  var seq='ccccccttssttccccccccccttcccccccttsssttccccccc';
  var html='',i,ch;
  for(i=0;i<seq.length;i++){ch=seq[i];html+='<i class="'+(ch==='c'?'c':ch==='t'?'t':'s')+'"></i>'}
  html+='<i class="now"></i>';
  band.innerHTML=html;
})();

/* ============ red XAS-03 ============ */
(function(){
  var net=$('#net3'); if(!net||REDUCED)return;
  var nodes=$$('.node',net).slice(1), edges=$$('.edge',net), k=0;
  setInterval(function(){
    nodes.forEach(function(nd){nd.setAttribute('fill','rgba(201,163,92,.5)')});
    edges.forEach(function(e){e.setAttribute('stroke','rgba(201,163,92,.3)')});
    nodes[k].setAttribute('fill','#66E0C2');
    edges[k].setAttribute('stroke','rgba(102,224,194,.7)');
    k=(k+1)%nodes.length;
  },1300);
})();

/* ============ term structure (monitor) ============ */
(function(){
  var c=$('#tsCanvas'); if(!c)return;
  var ctx=c.getContext('2d');
  var A=[16.4,17.8,18.9,19.7,20.3,20.8,21.2,21.5,21.7];
  var B=[24.6,22.9,21.8,21.1,20.7,20.4,20.2,20.1,20.0];
  var cur=A.slice(), from=A.slice(), to=A, state='CONTANGO';
  var animStart=-1, ANIM=2400, HOLD=12000;
  var lab=$('#tsState');
  function ease(p){return p<.5?2*p*p:1-Math.pow(-2*p+2,2)/2}
  var lastW=-1,lastH=-1;
  function draw(t){
    var dpr=Math.min(window.devicePixelRatio||1,2);
    var W=c.clientWidth,H=c.clientHeight;
    if(W!==lastW||H!==lastH){lastW=W;lastH=H;c.width=W*dpr;c.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0)}
    ctx.clearRect(0,0,W,H);
    var padL=10,padR=14,padT=16,padB=24,i;
    var vmin=15,vmax=26;
    function X(i){return padL+i/(A.length-1)*(W-padL-padR)}
    function Y(v){return padT+(1-(v-vmin)/(vmax-vmin))*(H-padT-padB)}
    ctx.strokeStyle='rgba(201,163,92,.14)';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(padL,H-padB);ctx.lineTo(W-padR,H-padB);ctx.stroke();
    ctx.beginPath();
    for(i=0;i<cur.length;i++){
      if(i)ctx.lineTo(X(i),Y(cur[i])); else ctx.moveTo(X(i),Y(cur[i]));
    }
    ctx.strokeStyle='#C9A35C';ctx.lineWidth=1.4;ctx.stroke();
    ctx.font='400 9px "JetBrains Mono",monospace';ctx.textAlign='center';
    for(i=0;i<cur.length;i++){
      ctx.fillStyle='rgba(201,163,92,.85)';
      ctx.beginPath();ctx.arc(X(i),Y(cur[i]),2,0,7);ctx.fill();
      ctx.fillStyle='rgba(121,128,140,.8)';
      ctx.fillText('M'+(i+1),X(i),H-8);
    }
    var r=REDUCED?2.8:2.8+Math.sin(t/450)*1;
    ctx.fillStyle='rgba(102,224,194,.95)';
    ctx.shadowColor='rgba(102,224,194,.8)';ctx.shadowBlur=10;
    ctx.beginPath();ctx.arc(X(0),Y(cur[0]),r,0,7);ctx.fill();ctx.shadowBlur=0;
  }
  if(REDUCED){draw(0);window.addEventListener('resize',function(){draw(0)});return}
  var lastSwitch=performance.now();
  (function loop(now){
    if(animStart<0 && now-lastSwitch>HOLD){
      animStart=now; from=cur.slice(); to=(to===A)?B:A;
      state=(to===A)?'CONTANGO':'BACKWARDATION';
    }
    if(animStart>=0){
      var p=Math.min((now-animStart)/ANIM,1), e=ease(p), i;
      for(i=0;i<cur.length;i++)cur[i]=lerp(from[i],to[i],e);
      if(p>=1){
        animStart=-1; lastSwitch=now;
        if(lab){lab.dataset.final=state;runDecode(lab)}
      }
    }
    if(!document.hidden)draw(now);
    requestAnimationFrame(loop);
  })(performance.now());
})();

/* ============ gauge percentil VRP ============ */
(function(){
  var ticks=$('#gaugeTicks'), needle=$('#gaugeNeedle'), val=$('#gaugeVal');
  if(!ticks||!needle)return;
  var i,a,r1=80,r2=86,html='';
  for(i=0;i<=10;i++){
    a=(-90+i*18)*Math.PI/180;
    html+='<line x1="'+(100+Math.sin(a)*r1)+'" y1="'+(106-Math.cos(a)*r1)+
          '" x2="'+(100+Math.sin(a)*r2)+'" y2="'+(106-Math.cos(a)*r2)+'"/>';
  }
  ticks.innerHTML=html;
  var p=42;
  function set(v){
    needle.style.transform='rotate('+(-90+v*1.8)+'deg)';
    if(val)val.textContent=String(Math.round(v));
  }
  set(p);
  if(REDUCED)return;
  setInterval(function(){
    p=Math.max(18,Math.min(74,p+(Math.random()-.5)*16));
    set(p);
  },5200);
})();

/* ============ matriz de correlaciones ============ */
(function(){
  var c=$('#matrixCanvas'); if(!c)return;
  var ctx=c.getContext('2d');
  var assets=['VIX','MOVE','CDX','DXY','UST','SPX'];
  var base=[
    [1,.62,.48,.21,.35,-.72],
    [.62,1,.41,.18,.55,-.44],
    [.48,.41,1,.12,.3,-.51],
    [.21,.18,.12,1,.25,-.2],
    [.35,.55,.3,.25,1,-.33],
    [-.72,-.44,-.51,-.2,-.33,1]];
  var lastW=-1,lastH=-1;
  function draw(t){
    var dpr=Math.min(window.devicePixelRatio||1,2);
    var W=c.clientWidth,H=c.clientHeight;
    if(W!==lastW||H!==lastH){lastW=W;lastH=H;c.width=W*dpr;c.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0)}
    ctx.clearRect(0,0,W,H);
    var n=assets.length, labW=38, labH=16;
    var cw=(W-labW-4)/n, ch=(H-labH-4)/n, i,j,v,a;
    ctx.font='400 8.5px "JetBrains Mono",monospace';
    for(i=0;i<n;i++){
      ctx.fillStyle='rgba(121,128,140,.8)';
      ctx.textAlign='right';
      ctx.fillText(assets[i],labW-6,labH+i*ch+ch/2+3);
      ctx.textAlign='center';
      ctx.fillText(assets[i],labW+i*cw+cw/2,10);
      for(j=0;j<n;j++){
        v=base[i][j];
        if(i!==j) v=v+Math.sin(t/3000+i*1.7+j*2.3)*.08;
        a=Math.abs(v);
        ctx.fillStyle=v>=0?'rgba(201,163,92,'+(a*.55+.04)+')':'rgba(102,224,194,'+(a*.4+.04)+')';
        ctx.fillRect(labW+j*cw+1,labH+i*ch+1,cw-2,ch-2);
      }
    }
  }
  if(REDUCED){draw(0);window.addEventListener('resize',function(){draw(0)});return}
  (function loop(now){
    if(!document.hidden)draw(now);
    requestAnimationFrame(loop);
  })(0);
})();

/* ============ registro de eventos ============ */
(function(){
  var list=$('#logList'); if(!list)return;
  function pad(n){return n<10?'0'+n:String(n)}
  function utc(){var d=new Date();return pad(d.getUTCHours())+':'+pad(d.getUTCMinutes())+':'+pad(d.getUTCSeconds())}
  var msgs=[
    'RGM-02 · régimen estable · <b>P(calma)=0.86</b>',
    'VRP-01 · prima implícita-realizada <b>+3.4 pts</b>',
    'XAS-03 · divergencia crédito-vol · <b>dentro de umbral</b>',
    'TS · pendiente M1–M2 <b>+0.84</b> · contango',
    'RISK · exposición dentro de límites · <b>OK</b>',
    'DATA · ingesta completada · <b>42 series</b> · 0 errores',
    'VRP-01 · percentil 1A · <b>P 42</b>',
    'XAS-03 · correlación media <b>0.31</b> · sin alerta'
  ];
  var k=0;
  function add(){
    var li=document.createElement('li');
    li.innerHTML='<span class="t">'+utc()+'</span>'+msgs[k%msgs.length];
    k++;
    list.insertBefore(li,list.firstChild);
    while(list.children.length>6)list.removeChild(list.lastChild);
  }
  add();add();add();
  if(!REDUCED)setInterval(add,2600);
})();

/* ============ firma ============ */
(function(){
  var svg=$('#sigSvg'); if(!svg)return;
  var paths=$$('path',svg);
  if(REDUCED)return;
  paths.forEach(function(pa){
    var len=pa.getTotalLength();
    pa.style.strokeDasharray=len;
    pa.style.strokeDashoffset=len;
  });
  var io=new IntersectionObserver(function(es){
    es.forEach(function(en){
      if(!en.isIntersecting)return;
      io.unobserve(svg);
      var delay=0;
      paths.forEach(function(pa){
        pa.style.transition='stroke-dashoffset 1.5s '+'cubic-bezier(.16,1,.3,1) '+delay+'ms';
        delay+=700;
        requestAnimationFrame(function(){pa.style.strokeDashoffset='0'});
      });
    });
  },{threshold:.5});
  io.observe(svg);
})();

/* ============ formulario ============ */
(function(){
  var form=$('#accessForm'), ok=$('#formOk');
  if(!form)return;
  var fields=$$('input,select',form);
  function errFor(f){return document.getElementById('e-'+f.id.slice(2))}
  function setErr(f,msg){
    var e=errFor(f); if(!e)return;
    if(msg){f.setAttribute('aria-invalid','true');e.textContent=msg;e.classList.add('show')}
    else{f.removeAttribute('aria-invalid');e.textContent='';e.classList.remove('show')}
  }
  function validate(f){
    if(!f.value){setErr(f,'Campo obligatorio');return false}
    if(f.type==='email'&&!f.checkValidity()){setErr(f,'Introduzca un correo válido');return false}
    setErr(f,'');return true;
  }
  fields.forEach(function(f){
    f.addEventListener('input',function(){if(f.getAttribute('aria-invalid'))validate(f)});
    f.addEventListener('change',function(){if(f.getAttribute('aria-invalid'))validate(f)});
  });
  form.addEventListener('submit',function(e){
    e.preventDefault();
    var valid=true, first=null;
    fields.forEach(function(f){
      if(!validate(f)){valid=false;if(!first)first=f}
    });
    if(!valid){if(first)first.focus();return}
    form.style.display='none';
    ok.style.display='block';
    ok.focus();
  });
})();

})();
