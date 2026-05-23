console.log("JS BRIDGE: Loaded successfully");

/**
 * Minimal Streamlit custom-component bridge for static HTML (production / _RELEASE).
 * Served from the same directory as index.html — no dev server on port 3001.
 */
(function (global) {
  "use strict";

  var RENDER_EVENT = "streamlit:render";

  var handlers = {};
  var componentReadySent = false;

  function post(type, payload) {
    global.parent.postMessage(Object.assign({ type: type }, payload || {}), "*");
  }

  var Streamlit = {
    RENDER_EVENT: RENDER_EVENT,
    events: {
      addEventListener: function (type, fn) {
        if (!handlers[type]) handlers[type] = [];
        handlers[type].push(fn);
      },
      removeEventListener: function (type, fn) {
        if (!handlers[type]) return;
        handlers[type] = handlers[type].filter(function (h) {
          return h !== fn;
        });
      },
      dispatchEvent: function (event) {
        var list = handlers[event.type] || [];
        for (var i = 0; i < list.length; i++) {
          try {
            list[i](event);
          } catch (err) {
            console.error("[streamlit_bridge]", err);
          }
        }
      },
    },
    setComponentReady: function () {
      if (componentReadySent) return;
      componentReadySent = true;
      console.log("JS BRIDGE: Attempting to notify parent...");
      try {
        post("streamlit:componentReady", { isStreamlitComponent: true });
        console.log("JS BRIDGE: componentReady postMessage sent");
      } catch (err) {
        componentReadySent = false;
        console.error("JS BRIDGE: setComponentReady failed", err);
        if (err && err.stack) console.error(err.stack);
      }
    },
    setFrameHeight: function (height) {
      var h = Math.max(400, Number(height) || 700);
      post("streamlit:setFrameHeight", { height: h });
    },
    setComponentValue: function (value) {
      post("streamlit:setComponentValue", { value: value });
    },
  };

  global.addEventListener("message", function (event) {
    if (!event.data || event.data.type !== "streamlit:render") return;
    Streamlit.events.dispatchEvent({
      type: RENDER_EVENT,
      detail: { args: event.data.args || {} },
    });
  });

  global.Streamlit = Streamlit;
})(window);
