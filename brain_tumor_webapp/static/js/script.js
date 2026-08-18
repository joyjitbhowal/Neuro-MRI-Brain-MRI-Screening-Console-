/* BDNeuro-MRI scan console: theme, nav states, mobile menu, scrollspy,
   reveal animations, toasts, back-to-top, scroll progress, AJAX
   classification with loading overlay, and PWA service-worker registration. */

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

document.addEventListener("DOMContentLoaded", () => {

  /* ================================================== theme (light/dark) */
  const root = document.documentElement;
  const toggleBtn = document.getElementById("theme-toggle");
  const saved = localStorage.getItem("bdneuro-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  let theme = saved || (prefersDark ? "dark" : "light");
  root.setAttribute("data-theme", theme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      theme = theme === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", theme);
      localStorage.setItem("bdneuro-theme", theme);
    });
  }

  /* =========================== scroll state: nav elevation + progress + to-top */
  const nav = document.querySelector(".app-nav");
  const progress = document.getElementById("scroll-progress");
  const backTop = document.getElementById("back-to-top");
  let ticking = false;

  const onScroll = () => {
    const y = window.scrollY;
    const h = document.documentElement.scrollHeight - window.innerHeight;
    if (nav) nav.classList.toggle("scrolled", y > 8);
    if (progress) progress.style.transform = "scaleX(" + (h > 0 ? Math.min(y / h, 1) : 0) + ")";
    if (backTop) backTop.classList.toggle("visible", y > 600);
    ticking = false;
  };
  const requestTick = () => {
    if (!ticking) { ticking = true; requestAnimationFrame(onScroll); }
  };
  window.addEventListener("scroll", requestTick, { passive: true });
  onScroll();

  if (backTop) {
    backTop.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
  }

  /* ====================================================== mobile menu */
  const toggleMenuBtn = document.getElementById("nav-toggle");
  const navPanel = document.getElementById("nav-panel");

  if (toggleMenuBtn && navPanel) {
    const setMenu = open => {
      navPanel.classList.toggle("open", open);
      toggleMenuBtn.classList.toggle("open", open);
      toggleMenuBtn.setAttribute("aria-expanded", String(open));
      toggleMenuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    };
    toggleMenuBtn.addEventListener("click", () => {
      setMenu(!navPanel.classList.contains("open"));
    });
    navPanel.querySelectorAll("a").forEach(a =>
      a.addEventListener("click", () => setMenu(false)));
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && navPanel.classList.contains("open")) setMenu(false);
    });
    document.addEventListener("click", e => {
      if (navPanel.classList.contains("open") && !nav.contains(e.target)) setMenu(false);
    });
  }

  /* ====================================================== scrollspy nav */
  const navLinks = Array.from(document.querySelectorAll(".nav-links a[href^='#']"));
  const sections = navLinks
    .map(a => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);

  const setActive = link => {
    navLinks.forEach(a => {
      const on = a === link;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  };

  if ("IntersectionObserver" in window && sections.length) {
    const spy = new IntersectionObserver(
      entries => {
        const visible = entries
          .filter(e => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const link = navLinks.find(a => a.getAttribute("href") === "#" + visible.target.id);
        if (link) setActive(link);
      },
      { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.25, 0.5, 1] }
    );
    sections.forEach(s => spy.observe(s));
  }

  /* ================================================= reveal on scroll */
  const revealEls = document.querySelectorAll(".reveal");
  if (revealEls.length) {
    if (reduced || !("IntersectionObserver" in window)) {
      revealEls.forEach(el => el.classList.add("in-view"));
    } else {
      const io = new IntersectionObserver(
        entries => {
          entries.forEach(e => {
            if (e.isIntersecting) {
              e.target.classList.add("in-view");
              io.unobserve(e.target);
            }
          });
        },
        { threshold: 0.12 }
      );
      revealEls.forEach(el => io.observe(el));
    }
  }

  /* ================================================= benchmark bars */
  const barsWrap = document.querySelector(".bars");
  if (barsWrap) {
    const go = () => {
      if (!barsWrap.classList.contains("go")) barsWrap.classList.add("go");
    };
    if (reduced || !("IntersectionObserver" in window)) {
      go();
    } else {
      const io = new IntersectionObserver(
        entries => {
          if (entries.some(e => e.isIntersecting)) { go(); io.disconnect(); }
        },
        { threshold: 0.25 }
      );
      io.observe(barsWrap);
    }
  }

  /* ============================================================ toasts */
  const toastWrap = document.getElementById("toast-wrap");

  const toast = (msg, type) => {
    if (!toastWrap) return;
    const el = document.createElement("div");
    el.className = "toast " + (type || "info");
    el.setAttribute("role", type === "error" ? "alert" : "status");
    el.textContent = msg;
    toastWrap.appendChild(el);
    requestAnimationFrame(() => el.classList.add("show"));
    setTimeout(() => {
      el.classList.remove("show");
      setTimeout(() => el.remove(), 300);
    }, 5000);
  };

  /* ============================================== AJAX classification */
  const form = document.getElementById("upload-form");
  const drop = document.getElementById("dropzone");
  const fileInput = document.getElementById("image");
  const loader = document.getElementById("loader");
  const loaderStep = document.getElementById("loader-step");
  const loaderFill = document.getElementById("loader-fill");
  const resultPanel = document.getElementById("result-panel");
  const analyzeBtn = document.getElementById("analyze-btn");
  const dzName = document.getElementById("dz-file-name");
  const dzSize = document.getElementById("dz-file-size");

  const esc = s => String(s).replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const ALLOWED_EXTS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"];
  const MAX_BYTES = 16 * 1024 * 1024;

  const fmtSize = b =>
    b < 1048576 ? (b / 1024).toFixed(0) + " KB" : (b / 1048576).toFixed(1) + " MB";

  const validateFile = (name, size) => {
    const ext = (name.split(".").pop() || "").toLowerCase();
    if (ALLOWED_EXTS.indexOf(ext) === -1) {
      toast("Unsupported file type. Use JPG, PNG, BMP or TIFF.", "error");
      return false;
    }
    if (size > MAX_BYTES) {
      toast("File is too large (max 16 MB).", "error");
      return false;
    }
    return true;
  };

  const STEPS = [
    "Loading models", "Running 7 models", "Scoring the ensemble", "Finalizing verdict"
  ];

  function showLoader() {
    loader.hidden = false;
    let i = 0;
    loaderStep.textContent = STEPS[0] + "\u2026";
    loaderFill.style.width = "0%";
    const stepTimer = setInterval(() => {
      i = Math.min(i + 1, STEPS.length - 1);
      loaderStep.style.opacity = 0;
      setTimeout(() => {
        loaderStep.textContent = STEPS[i] + "\u2026";
        loaderStep.style.opacity = 1;
      }, 220);
    }, 1400);
    const fillTimer = setInterval(() => {
      const w = parseFloat(loaderFill.style.width) || 0;
      if (w < 82) loaderFill.style.width = Math.min(w + 9 + Math.random() * 10, 82) + "%";
    }, 240);
    loader._timers = [stepTimer, fillTimer];
  }

  function hideLoader() {
    loaderFill.style.width = "100%";
    setTimeout(() => {
      loader.hidden = true;
      (loader._timers || []).forEach(clearInterval);
    }, 350);
  }

  const SKELETON =
    '<div class="skeleton" role="status" aria-label="Analyzing your scan">' +
    '  <div class="skeleton-line" style="width:42%; height:22px;"></div>' +
    '  <div class="skeleton-line" style="width:100%;"></div>' +
    '  <div class="skeleton-line" style="width:92%;"></div>' +
    '  <div class="skeleton-line" style="width:70%;"></div>' +
    '  <div class="skeleton-line" style="width:85%;"></div>' +
    '  <div class="skeleton-line" style="width:55%;"></div>' +
    "</div>";

  const EMPTY =
    '<div style="text-align:center; padding:64px 20px;">' +
    '  <div style="font-family:var(--font-mono); color:var(--cyan); font-size:26px; margin-bottom:10px;">&#9678;</div>' +
    '  <h4 style="font-size:16px; margin:0 0 6px;">Awaiting input</h4>' +
    '  <p style="color:var(--text-muted); font-size:13.5px; margin:0;">' +
    '    Run the scan and the verdict, class probabilities and the per-model breakdown will appear here.' +
    "  </p>" +
    "</div>";

  const setBusy = busy => {
    if (!analyzeBtn) return;
    analyzeBtn.classList.toggle("is-loading", busy);
    analyzeBtn.disabled = busy;
  };

  async function runPrediction(blob, filename) {
    if (!validateFile(filename, blob.size)) return;
    showLoader();
    setBusy(true);
    resultPanel.innerHTML = SKELETON;
    try {
      const fd = new FormData();
      fd.append("image", blob, filename || "scan.jpg");
      const resp = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok || data.error) throw new Error(data.error || "prediction failed");
      renderResults(data, blob);
    } catch (err) {
      resultPanel.innerHTML = EMPTY;
      toast(err.message === "Failed to fetch"
        ? "Could not reach the analysis service. Please try again."
        : err.message, "error");
      resultPanel.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "nearest" });
    } finally {
      hideLoader();
      setBusy(false);
    }
  }

  function renderResults(d, blob) {
    const names = d.class_names;
    const w = d.weighted;
    const isTumor = w.label !== "no_tumor";
    const pct = x => (x * 100).toFixed(1) + "%";

    const verdict =
      '<div class="verdict ' + (isTumor ? "verdict-tumor" : "verdict-clean") + '">' +
      '  <span class="v-ic">' + (isTumor ? "\u26A0" : "\u2713") + "</span>" +
      "  <div>" +
      "    <h4>" + (isTumor ? "Possible tumor detected" : "No tumor detected") + "</h4>" +
      "    <p>" +
      "      Accuracy-weighted verdict: <strong class=\"mono\">" + esc(w.label) + "</strong>" +
      "      (score " + pct(w.confidence) + " of confidence mass)." +
      (isTumor ? " Screening signal \u2014 confirm with a specialist." : "") +
      "    </p>" +
      '    <div class="v-chip-row">' +
      '      <span class="v-chip">Weighted score · <span class="num">' + esc(w.label) +
      "        </span> · " + pct(w.confidence) + "</span>" +
      '      <span class="v-chip">Consensus · <span class="num">' + esc(d.majority_vote.label) +
      "        </span> · " + pct(d.majority_vote.confidence) + "</span>" +
      '      <span class="v-chip">Ensemble · <span class="num">' + esc(d.ensemble.label) +
      "        </span> · " + pct(d.ensemble.confidence) + "</span>" +
      "    </div>" +
      "  </div>" +
      "</div>";

    let preview = "";
    if (blob && blob.type && blob.type !== "image/tiff") {
      const url = URL.createObjectURL(blob);
      preview = '<div style="text-align:center; margin-bottom:16px;">' +
        '<img src="' + url + '" alt="analyzed scan" style="max-width:200px; width:100%; ' +
        'border-radius:10px; border:1px solid var(--glass-bd);">' +
        "</div>";
    }

    let bars = '<div class="field-label">Class scores · accuracy-weighted</div>';
    names.forEach(c => {
      const score = w.scores[c] || 0;
      bars += '<div class="prob-row">' +
        '<span class="prob-name">' + esc(c) + "</span>" +
        '<span class="prob-bar"><span class="prob-fill" data-w="' + (score * 100) + '" style="width:0"></span></span>' +
        '<span class="prob-pct">' + pct(score) + "</span>" +
        "</div>";
    });
    bars += '<p class="sec-sub" style="margin-top:12px; font-size:12px;">' +
      "Score = 70% weighted probability + 30% weighted vote, where each model's " +
      "weight is its test accuracy raised to the 4th power \u2014 so top models " +
      "dominate the verdict.</p>";

    let rows = "";
    d.models.forEach(m => {
      const weight = (w.weights[m.name] || 0) * 100;
      const latency = (m.latency_ms != null ? m.latency_ms : 0).toFixed(0) + " ms";
      rows += "<tr><td class=\"mono\">" + esc(m.name) + "</td><td>" + esc(m.label) +
        "</td><td class=\"num\">" + pct(m.confidence) +
        '</td><td class="num">' + weight.toFixed(1) + "%" +
        '</td><td class="num">' + latency + "</td></tr>";
    });

    resultPanel.innerHTML =
      verdict + preview + bars +
      '<details class="model-detail">' +
      "  <summary>Per-model breakdown · " + d.models.length + " predictions</summary>" +
      '  <div class="table-wrap" style="border-top:1px solid var(--glass-bd);">' +
      '    <table class="bench">' +
      "      <thead><tr><th>Model</th><th>Prediction</th><th>Confidence</th><th>Weight</th><th>Latency</th></tr></thead>" +
      "      <tbody>" + rows + "</tbody></table>" +
      "  </div>" +
      "</details>";

    requestAnimationFrame(() => {
      resultPanel.querySelectorAll(".prob-fill").forEach(el => {
        requestAnimationFrame(() => { el.style.width = el.dataset.w + "%"; });
      });
      const verdictEl = resultPanel.querySelector(".verdict");
      if (verdictEl) {
        verdictEl.setAttribute("tabindex", "-1");
        verdictEl.focus({ preventScroll: true });
      }
      resultPanel.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "nearest" });
    });
    hideLoader();
  }

  const setFileInfo = f => {
    if (!f || !drop) return;
    drop.classList.add("dz-has-file");
    if (dzName) dzName.textContent = f.name;
    if (dzSize) dzSize.textContent = fmtSize(f.size);
  };

  if (form && drop && fileInput && loader && resultPanel) {
    form.addEventListener("submit", e => {
      if (!fileInput.files || !fileInput.files.length) return;
      e.preventDefault();
      const f = fileInput.files[0];
      setFileInfo(f);
      runPrediction(f, f.name);
    });

    ["dragover", "dragenter"].forEach(ev =>
      drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("drag-over"); }));
    ["dragleave", "drop"].forEach(ev =>
      drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("drag-over"); }));
    drop.addEventListener("drop", e => {
      if (e.dataTransfer.files && e.dataTransfer.files.length) {
        const f = e.dataTransfer.files[0];
        setFileInfo(f);
        runPrediction(f, f.name);
      }
    });

    document.querySelectorAll(".sample-card").forEach(card => {
      card.addEventListener("click", e => {
        e.preventDefault();
        fetch(card.href)
          .then(r => {
            if (!r.ok) throw new Error("sample unavailable");
            return r.blob();
          })
          .then(blob => runPrediction(blob, "sample.jpg"))
          .catch(() => { window.location = card.href; });
      });
    });
  }

  /* ====================================================== PWA (safe) */
  if ("serviceWorker" in navigator &&
      (location.protocol === "https:" ||
       location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("static/sw.js").catch(() => { /* no-op */ });
    });
  }
});