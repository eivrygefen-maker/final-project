/**
 * Load Three.js + addons locally (CSP-safe). Exposes globals for the inline Design Studio script.
 * OrbitControls is required; ArcballControls is optional (GMSH-style trackball when available).
 */
import * as ThreeModule from "three";
import { OrbitControls } from "./OrbitControls.js";
import { GLTFLoader } from "./GLTFLoader.js";

async function loadArcballControls() {
  try {
    const mod = await import("./ArcballControls.js");
    return mod.ArcballControls;
  } catch (err) {
    console.warn(
      "[load-three] ArcballControls unavailable — OrbitControls fallback:",
      err && err.message ? err.message : err
    );
    return null;
  }
}

(async function bootThree() {
  try {
    const THREE = Object.assign(Object.create(null), ThreeModule);
    THREE.OrbitControls = OrbitControls;

    const ArcballControls = await loadArcballControls();
    if (typeof ArcballControls === "function") {
      THREE.ArcballControls = ArcballControls;
      window.ArcballControls = ArcballControls;
    }

    window.THREE = THREE;
    window.OrbitControls = OrbitControls;
    window.GLTFLoader = GLTFLoader;
    window.__threeReady = true;
    window.__arcballAvailable = typeof THREE.ArcballControls === "function";

    window.dispatchEvent(
      new CustomEvent("three-ready", {
        detail: { THREE, OrbitControls, ArcballControls, GLTFLoader },
      })
    );
  } catch (err) {
    window.__threeReady = false;
    window.__arcballAvailable = false;
    window.dispatchEvent(
      new CustomEvent("three-error", {
        detail: { message: err && err.message ? err.message : String(err), error: err },
      })
    );
    throw err;
  }
})();
