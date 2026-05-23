/**
 * Load Three.js + addons locally (CSP-safe). Exposes globals for the inline Design Studio script.
 */
import * as ThreeModule from "three";
import { ArcballControls } from "./ArcballControls.js";
import { GLTFLoader } from "./GLTFLoader.js";

try {
  // ES module namespace objects are read-only — copy onto a mutable plain object.
  const THREE = Object.assign(Object.create(null), ThreeModule);
  THREE.ArcballControls = ArcballControls;

  window.THREE = THREE;
  window.ArcballControls = ArcballControls;
  window.GLTFLoader = GLTFLoader;
  window.__threeReady = true;

  window.dispatchEvent(
    new CustomEvent("three-ready", {
      detail: { THREE, ArcballControls, GLTFLoader },
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
