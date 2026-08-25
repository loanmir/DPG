/* DPG Interactive Runner -- frontend behaviour.
 *
 * Behaviour
 * ---------
 * 1. User changes any of the four inputs (dataset / perc_var /
 *    decimal_threshold / community_threshold).
 * 2. We debounce 250 ms then re-run the full pipeline.
 * 3. We hide the previous images so the page doesn't show stale renders
 *    while the new run is being computed.
 * 4. On 200 OK we populate each <img data-label=...> with the new URL
 *    returned by /api/run.
 * 5. We render the metrics + any per-stage error messages.
 *
 * Manual submit via the "Run pipeline" button still works; we just bypass
 * the debounce in that case.
 */

(function () {
    "use strict";

    const STAGE_LABELS = [
        "raw",
        "cat_view",
        "grouped",
        "split",
        "conjunction",
    ];

    const form = document.getElementById("run-form");
    const submitBtn = document.getElementById("run-btn");
    const status = document.getElementById("status");

    const dl = document.getElementById("metrics-dl");
    const errorsBox = document.getElementById("errors");
    const errorsPre = document.getElementById("errors-pre");

    let debounceTimer = null;
    let inflightController = null;

    /* --- helpers --------------------------------------------------------- */

    function setStatus(text, cls) {
        status.textContent = text;
        status.className = "status " + (cls || "idle");
    }

    function clearArtifacts() {
        STAGE_LABELS.forEach((label) => {
            const img = document.querySelector(`img[data-label="${label}"]`);
            if (img) {
                img.removeAttribute("src");
                img.alt = label + " (loading…)";
            }
        });
    }

    function renderMetrics(metrics) {
        dl.innerHTML = "";
        const entries = [
            ["accuracy", metrics.accuracy],
            ["f1", metrics.f1],
            ["n_nodes", metrics.n_nodes],
            ["n_edges", metrics.n_edges],
            ["n_communities", metrics.n_communities],
            ["n_features_used", metrics.n_features_used],
        ];
        entries.forEach(([k, v]) => {
            const dt = document.createElement("dt");
            dt.textContent = k;
            const dd = document.createElement("dd");
            dd.textContent = v == null ? "—" : String(v);
            dl.appendChild(dt);
            dl.appendChild(dd);
        });
    }

    function renderErrors(errors) {
        if (!errors || errors.length === 0) {
            errorsBox.hidden = true;
            errorsPre.textContent = "";
            return;
        }
        errorsBox.hidden = false;
        errorsPre.textContent = errors.join("\n\n");
    }

    function renderImages(imageUrls) {
        STAGE_LABELS.forEach((label) => {
            const img = document.querySelector(`img[data-label="${label}"]`);
            if (!img) return;
            const url = imageUrls && imageUrls[label];
            if (url) {
                img.src = url;
                img.alt = label;
            }
        });
    }

    async function runPipeline(payload, opts) {
        opts = opts || {};

        // Cancel any in-flight request so we never display stale results.
        if (inflightController) {
            try {
                inflightController.abort();
            } catch (e) {
                /* no-op */
            }
        }
        inflightController = new AbortController();

        submitBtn.disabled = true;
        setStatus("Running pipeline…", "running");
        if (opts.immediate) clearArtifacts();

        try {
            const res = await fetch("/api/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal: inflightController.signal,
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error("HTTP " + res.status + ": " + text);
            }
            const data = await res.json();
            renderMetrics(data.metrics || {});
            renderErrors(data.errors || []);
            renderImages(data.image_urls || {});
            setStatus(
                "Done in " +
                    (data.elapsed_seconds || "?") +
                    "s. Run id: " +
                    (data.run_id || "?"),
                "ok"
            );
        } catch (err) {
            if (err.name === "AbortError") {
                // Superseded by a newer click; nothing to show.
                return;
            }
            setStatus("Error: " + err.message, "error");
            errorsBox.hidden = false;
            errorsPre.textContent = String(err);
        } finally {
            submitBtn.disabled = false;
        }
    }

    function readForm() {
        return {
            dataset: form.dataset.value,
            perc_var: parseFloat(form.perc_var.value),
            decimal_threshold: parseInt(form.decimal_threshold.value, 10),
            community_threshold: parseFloat(form.community_threshold.value),
        };
    }

    function scheduleRun() {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            debounceTimer = null;
            runPipeline(readForm(), { immediate: true });
        }, 250);
    }

    /* --- wire up --------------------------------------------------------- */

    // Debounce on every input change.
    ["change", "input"].forEach((evt) => {
        form.addEventListener(evt, (e) => {
            if (e.target && e.target.tagName === "BUTTON") return;
            scheduleRun();
        });
    });

    // Click on the submit button: run immediately, no debounce.
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        runPipeline(readForm(), { immediate: true });
    });
})();
