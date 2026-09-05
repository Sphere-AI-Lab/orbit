/* Tiny JS helpers for the Orbit docs site */

document.addEventListener("DOMContentLoaded", () => {
  // Scroll-reveal: add .is-visible to landing cards / strips when they enter the viewport.
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries, obs) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add("is-visible");
            obs.unobserve(e.target);
          }
        }
      },
      { rootMargin: "0px 0px -10% 0px", threshold: 0.05 }
    );
    document
      .querySelectorAll(".orbit-card, .orbit-update, .orbit-quote, .orbit-strip")
      .forEach((el) => io.observe(el));
  } else {
    // Fallback: just show everything.
    document
      .querySelectorAll(".orbit-card, .orbit-update, .orbit-quote, .orbit-strip")
      .forEach((el) => el.classList.add("is-visible"));
  }

  // Hover micro-interaction on update timeline rows.
  document.querySelectorAll(".orbit-update").forEach((el) => {
    el.addEventListener("mouseenter", () => el.classList.add("is-hover"));
    el.addEventListener("mouseleave", () => el.classList.remove("is-hover"));
  });
});
