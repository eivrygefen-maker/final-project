/**
 * Load Three.js + addons locally (CSP-safe). Exposes globals for the inline Design Studio script.
 */
import * as ThreeModule from "three";
import { OrbitControls } from "./OrbitControls.js";
import { GLTFLoader } from "./GLTFLoader.js";

try {
  // ES module namespace objects are read-only — copy onto a mutable plain object.
  const THREE = Object.assign(Object.create(null), ThreeModule);
  THREE.OrbitControls = OrbitControls;

  window.THREE = THREE;
  window.OrbitControls = OrbitControls;
  window.GLTFLoader = GLTFLoader;
  window.__threeReady = true;

  window.dispatchEvent(
    new CustomEvent("three-ready", {
      detail: { THREE, OrbitControls, GLTFLoader },
    })
  );
} catch (err) {
  window.__threeReady = false;
  window.dispatchEvent(
    new CustomEvent("three-error", {
      detail: { message: err && err.message ? err.message : String(err), error: err },
    })
  );
  throw err;
}
