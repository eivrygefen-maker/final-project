/**
 * Load Three.js + addons locally (CSP-safe). Exposes globals for the inline Design Studio script.
 */
import * as THREE from "three";
import { OrbitControls } from "./OrbitControls.js";
import { GLTFLoader } from "./GLTFLoader.js";

THREE.OrbitControls = OrbitControls;
window.THREE = THREE;
window.GLTFLoader = GLTFLoader;
window.dispatchEvent(new Event("three-ready"));
