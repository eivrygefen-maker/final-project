console.log("guitar_player bridge loaded");

(function (global) {
  "use strict";

  var RENDER_EVENT = "streamlit:render";
  var API_VERSION = 1;
  var DEFAULT_FRAME_HEIGHT = 520;

  var handlers = {};
  var componentReadySent = false;

  function post(type, payload) {
    global.parent.postMessage(
      Object.assign({ type: type, isStreamlitMessage: true }, payload || {}),
      "*"
    );
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
            console.error("[guitar_player bridge]", err);
          }
        }
      },
    },
    setComponentReady: function () {
      if (componentReadySent) return;
      componentReadySent = true;
      post("streamlit:componentReady", { apiVersion: API_VERSION });
      Streamlit.setFrameHeight(DEFAULT_FRAME_HEIGHT);
    },
    setFrameHeight: function (height) {
      var h = Math.max(320, Number(height) || DEFAULT_FRAME_HEIGHT);
      post("streamlit:setFrameHeight", { height: h });
    },
    setComponentValue: function (value, dataType) {
      post("streamlit:setComponentValue", {
        value: value,
        dataType: dataType || "json",
      });
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
  Streamlit.setComponentReady();
})(window);
