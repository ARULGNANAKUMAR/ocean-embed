/* ═══════════════════════════════════════════════════════════
   layers.js — Multi-variable layer data fetching (Section 8)
   Shared by earth.js and dashboard.js for layer toggling.
   ═══════════════════════════════════════════════════════════ */

const OceanLayers = (() => {
  const cache = {};

  async function fetchLayer(layer, timeIdx = -1, depthIdx = 0) {
    const key = `${layer}_${timeIdx}_${depthIdx}`;
    if (cache[key]) return cache[key];
    try {
      const data = await OceanApp.api(`/api/map/layer?layer=${layer}&time_idx=${timeIdx}&depth_idx=${depthIdx}`);
      cache[key] = data;
      return data;
    } catch (e) {
      return null;
    }
  }

  function clearCache() {
    Object.keys(cache).forEach((k) => delete cache[k]);
  }

  return { fetchLayer, clearCache };
})();
