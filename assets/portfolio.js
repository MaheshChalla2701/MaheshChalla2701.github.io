/* =========================================================
   Mahesh Challa — Portfolio interactions (vanilla JS)
   ========================================================= */
(function () {
  "use strict";

  /* ---------- Year ---------- */
  var yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* ---------- Theme toggle ---------- */
  var root = document.documentElement;
  var themeToggle = document.getElementById("themeToggle");
  var stored = (function () {
    try { return localStorage.getItem("theme"); } catch (e) { return null; }
  })();
  if (stored) root.setAttribute("data-theme", stored);
  function syncToggleIcon() {
    if (themeToggle) themeToggle.textContent = root.getAttribute("data-theme") === "light" ? "☀️" : "🌙";
  }
  syncToggleIcon();
  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
      syncToggleIcon();
    });
  }

  /* ---------- Mobile menu ---------- */
  var menuBtn = document.getElementById("menuBtn");
  var navLinks = document.getElementById("navLinks");
  if (menuBtn && navLinks) {
    menuBtn.addEventListener("click", function () { navLinks.classList.toggle("open"); });
    navLinks.addEventListener("click", function (e) {
      if (e.target.tagName === "A") navLinks.classList.remove("open");
    });
  }

  /* ---------- Header shadow + back-to-top on scroll ---------- */
  var header = document.getElementById("header");
  var toTop = document.getElementById("toTop");
  function onScroll() {
    var y = window.scrollY || window.pageYOffset;
    if (header) header.classList.toggle("scrolled", y > 20);
    if (toTop) toTop.classList.toggle("show", y > 500);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  if (toTop) {
    toTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  /* ---------- Typewriter ---------- */
  (function typewriter() {
    var el = document.getElementById("typed");
    if (!el) return;
    var words = ["intelligent mobile apps", "deep learning models", "AI products", "things people love"];
    var w = 0, c = 0, deleting = false;
    function tick() {
      var word = words[w];
      el.textContent = word.substring(0, c);
      if (!deleting) {
        if (c < word.length) { c++; setTimeout(tick, 75); }
        else { deleting = true; setTimeout(tick, 1400); }
      } else {
        if (c > 0) { c--; setTimeout(tick, 38); }
        else { deleting = false; w = (w + 1) % words.length; setTimeout(tick, 250); }
      }
    }
    tick();
  })();

  /* ---------- Skills tag cloud (3D sphere) ---------- */
  (function tagCloud() {
    if (typeof TagCloud === "undefined") return;
    var texts = [
      "Flutter", "Dart", "Firebase", "Python", "TensorFlow", "PyTorch",
      "Deep Learning", "CNN", "Transformers", "Computer Vision", "MediaPipe",
      "OpenCV", "NumPy", "Pandas", "FastAPI", "Flask", "REST APIs",
      "Android", "iOS", "Machine Learning", "Git", "GitHub", "SQL", "ML Ops"
    ];
    var instance = null;
    function build() {
      var radius = window.innerWidth >= 768 ? 180 : 130;
      if (instance) { try { instance.destroy(); } catch (e) {} }
      instance = TagCloud(".tagcloud", texts, {
        radius: radius,
        maxSpeed: "normal",
        initSpeed: "normal",
        keep: true
      });
    }
    build();
    var rt;
    window.addEventListener("resize", function () {
      clearTimeout(rt);
      rt = setTimeout(build, 250);
    });
  })();

  /* ---------- Scroll reveal ---------- */
  (function reveal() {
    var items = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window)) {
      items.forEach(function (el) { el.classList.add("visible"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry, i) {
        if (entry.isIntersecting) {
          entry.target.style.transitionDelay = (Math.min(i, 4) * 80) + "ms";
          entry.target.classList.add("visible");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    items.forEach(function (el) { io.observe(el); });
  })();

  /* ---------- Particle network background (canvas) ---------- */
  (function particles() {
    var canvas = document.getElementById("particles");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var w, h, dots, mouse = { x: null, y: null };
    var prefersReduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function accentColor() {
      return document.documentElement.getAttribute("data-theme") === "light"
        ? "13, 148, 136" : "45, 212, 191";
    }

    function size() {
      var rect = canvas.parentElement.getBoundingClientRect();
      w = canvas.width = rect.width;
      h = canvas.height = rect.height;
      var count = Math.min(90, Math.floor((w * h) / 14000));
      dots = [];
      for (var i = 0; i < count; i++) {
        dots.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.45,
          vy: (Math.random() - 0.5) * 0.45,
          r: Math.random() * 1.8 + 0.6
        });
      }
    }

    function draw() {
      var rgb = accentColor();
      ctx.clearRect(0, 0, w, h);
      for (var i = 0; i < dots.length; i++) {
        var d = dots[i];
        d.x += d.vx; d.y += d.vy;
        if (d.x < 0 || d.x > w) d.vx *= -1;
        if (d.y < 0 || d.y > h) d.vy *= -1;

        ctx.beginPath();
        ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(" + rgb + ", 0.7)";
        ctx.fill();

        for (var j = i + 1; j < dots.length; j++) {
          var o = dots[j];
          var dx = d.x - o.x, dy = d.y - o.y;
          var dist = dx * dx + dy * dy;
          if (dist < 16000) {
            ctx.beginPath();
            ctx.moveTo(d.x, d.y);
            ctx.lineTo(o.x, o.y);
            ctx.strokeStyle = "rgba(" + rgb + "," + (0.12 * (1 - dist / 16000)) + ")";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }

        if (mouse.x !== null) {
          var mdx = d.x - mouse.x, mdy = d.y - mouse.y;
          var md = mdx * mdx + mdy * mdy;
          if (md < 22000) {
            ctx.beginPath();
            ctx.moveTo(d.x, d.y);
            ctx.lineTo(mouse.x, mouse.y);
            ctx.strokeStyle = "rgba(129,140,248," + (0.18 * (1 - md / 22000)) + ")";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(draw);
    }

    var raf;
    size();
    if (!prefersReduced) draw();
    else { // static single frame
      draw();
      cancelAnimationFrame(raf);
    }

    window.addEventListener("resize", function () {
      cancelAnimationFrame(raf);
      size();
      if (!prefersReduced) draw();
    });
    canvas.parentElement.addEventListener("mousemove", function (e) {
      var rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    });
    canvas.parentElement.addEventListener("mouseleave", function () {
      mouse.x = mouse.y = null;
    });
  })();

})();
