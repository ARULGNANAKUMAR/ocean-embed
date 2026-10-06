/* ═══════════════════════════════════════════════════════════
   gismap.js — Phase 2: Real-World GIS Ocean Map
   Features:
   · MapLibre/Leaflet with satellite + ocean basemaps
   · SST, Salinity, Sea Level Anomaly heatmaps
   · Animated current & wind particles
   · Marine Protected Areas, EEZ, shipping routes
   · ARGO float markers
   · Coordinate inspector (click → grid cell API)
   · Depth level slider
   · Zoom breadcrumb trail
   ═══════════════════════════════════════════════════════════ */

"use strict";

// ══════════════════════════════════════════════════════════════
// CONFIG
// ══════════════════════════════════════════════════════════════

const GIS_CONFIG = {
  center: [13, 80],   // Bay of Bengal focus
  zoom: 5,
  minZoom: 3,
  maxZoom: 12,

  // North Indian Ocean bounding box for data generation
  bounds: { latMin: -10, latMax: 30, lonMin: 50, lonMax: 110 },
  res: 0.25,

  // Depth levels in metres (matching model output)
  DEPTHS_M: [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000, 1500],

  // Layer meta
  LAYERS: {
    sst:        { label: 'Sea Surface Temperature', unit: '°C', minVal: 24, maxVal: 36 },
    sss:        { label: 'Sea Surface Salinity', unit: 'PSU', minVal: 32, maxVal: 38 },
    sla:        { label: 'Sea Level Anomaly', unit: 'cm', minVal: -30, maxVal: 30 },
    current:    { label: 'Ocean Currents', unit: 'm/s', minVal: 0, maxVal: 1.5 },
    wind:       { label: 'Wind Speed', unit: 'm/s', minVal: 0, maxVal: 15 },
    heatwave:   { label: 'Heatwave Intensity', unit: '°C above normal', minVal: 0, maxVal: 5 },
    bathymetry: { label: 'Ocean Depth', unit: 'm', minVal: 0, maxVal: 6000 },
    shipping:   { label: 'Shipping Routes', unit: '', minVal: 0, maxVal: 1 },
    mpa:        { label: 'Marine Protected Areas', unit: '', minVal: 0, maxVal: 1 },
    eez:        { label: 'Exclusive Economic Zones', unit: '', minVal: 0, maxVal: 1 },
    argo:       { label: 'ARGO Float Positions', unit: '', minVal: 0, maxVal: 1 },
    ports:      { label: 'Major Ports', unit: '', minVal: 0, maxVal: 1 },
  }
};

// ══════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════

const state = {
  map: null,
  basemapLayer: null,
  activeLayers: new Set(['sst']),
  leafletLayers: {},
  depthIdx: 0,
  apiData: {
    prediction: null,
    heatwave: null,
    surface: null,
    argo: null,
  },
  particles: {
    wind: [], current: [],
    animId: null,
    ctx: null,
    canvas: null,
  },
  currentActiveLayer: 'sst',
};

// ══════════════════════════════════════════════════════════════
// BASEMAP TILE URLS
// ══════════════════════════════════════════════════════════════

const BASEMAPS = {
  ocean: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Esri, GEBCO, NOAA, National Geographic',
  },
  satellite: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Esri, Maxar, GeoEye, NASA',
  },
  topo: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Esri, USGS, NOAA',
  },
  dark: {
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attribution: 'CartoDB, OpenStreetMap',
    subdomains: 'abcd',
  },
};

// ══════════════════════════════════════════════════════════════
// SYNTHETIC DATA GENERATORS
// (replace with real API calls when live data available)
// ══════════════════════════════════════════════════════════════

function generateSSTGrid() {
  const { latMin, latMax, lonMin, lonMax, res } = GIS_CONFIG.bounds;
  const pts = [];
  for (let lat = latMin; lat <= latMax; lat += res * 2) {
    for (let lon = lonMin; lon <= lonMax; lon += res * 2) {
      // Realistic Bay of Bengal / Arabian Sea SST pattern
      const base = 28 + 4 * Math.cos((lat - 5) * Math.PI / 30);
      const noise = (Math.sin(lat * 3.7 + lon * 2.1) + Math.cos(lon * 4.3)) * 1.5;
      const warmPatch1 = lat > 10 && lat < 20 && lon > 85 && lon < 95 ? 2.5 : 0;
      const warmPatch2 = lat > 12 && lat < 22 && lon > 55 && lon < 68 ? 1.8 : 0;
      const temp = Math.min(36, Math.max(24, base + noise + warmPatch1 + warmPatch2));
      const intensity = (temp - 24) / 12;
      pts.push([lat, lon, intensity]);
    }
  }
  return pts;
}

function generateSalinityGrid() {
  const { latMin, latMax, lonMin, lonMax, res } = GIS_CONFIG.bounds;
  const pts = [];
  for (let lat = latMin; lat <= latMax; lat += res * 2) {
    for (let lon = lonMin; lon <= lonMax; lon += res * 2) {
      // BoB has lower salinity due to river discharge
      const base = 35 - 2 * Math.exp(-((lat - 15) ** 2 + (lon - 88) ** 2) / 50);
      const noise = Math.sin(lat * 2.1 + lon * 1.7) * 0.5;
      const sal = Math.min(38, Math.max(32, base + noise));
      pts.push([lat, lon, (sal - 32) / 6]);
    }
  }
  return pts;
}

function generateSLAGrid() {
  const { latMin, latMax, lonMin, lonMax, res } = GIS_CONFIG.bounds;
  const pts = [];
  for (let lat = latMin; lat <= latMax; lat += res * 2) {
    for (let lon = lonMin; lon <= lonMax; lon += res * 2) {
      const sla = 15 * Math.sin(lat * 0.3 + lon * 0.2) + 10 * Math.cos(lat * 0.5);
      const norm = (sla + 30) / 60;
      pts.push([lat, lon, Math.max(0, Math.min(1, norm))]);
    }
  }
  return pts;
}

function generateHeatwaveGrid(hotspots) {
  const pts = [];
  const { latMin, latMax, lonMin, lonMax, res } = GIS_CONFIG.bounds;
  for (let lat = latMin; lat <= latMax; lat += res * 2) {
    for (let lon = lonMin; lon <= lonMax; lon += res * 2) {
      let intensity = 0;
      for (const hs of hotspots) {
        const d2 = (lat - hs.latitude) ** 2 + (lon - hs.longitude) ** 2;
        const sev = hs.severity === 'EXTREME' ? 1 : hs.severity === 'HIGH' ? 0.7 : 0.4;
        intensity = Math.max(intensity, sev * Math.exp(-d2 / 8));
      }
      if (intensity > 0.05) pts.push([lat, lon, intensity]);
    }
  }
  return pts;
}

// ══════════════════════════════════════════════════════════════
// COLOUR SCALES
// ══════════════════════════════════════════════════════════════

const COLOUR_SCALES = {
  sst:        ['#0a1f5c', '#1565c0', '#0288d1', '#00bcd4', '#4caf50', '#ffeb3b', '#ff9800', '#f44336', '#b71c1c'],
  sss:        ['#b3e5fc', '#0288d1', '#01579b', '#003366'],
  sla:        ['#1a237e', '#1565c0', '#90caf9', '#e0e0e0', '#ef9a9a', '#c62828', '#b71c1c'],
  heatwave:   ['rgba(255,200,0,0)', 'rgba(255,150,0,0.6)', 'rgba(255,60,0,0.85)', 'rgba(180,0,0,0.95)'],
  bathymetry: ['#0d47a1', '#1565c0', '#1976d2', '#42a5f5', '#90caf9'],
};

function heatmapGradient(scale) {
  const obj = {};
  scale.forEach((c, i) => { obj[i / (scale.length - 1)] = c; });
  return obj;
}

// ══════════════════════════════════════════════════════════════
// MAP INIT
// ══════════════════════════════════════════════════════════════

function initMap() {
  const bm = BASEMAPS.ocean;
  const baseTile = L.tileLayer(bm.url, {
    attribution: bm.attribution,
    subdomains: bm.subdomains || 'abc',
    maxZoom: 18,
  });

  state.map = L.map('ocean-map', {
    center: GIS_CONFIG.center,
    zoom: GIS_CONFIG.zoom,
    minZoom: GIS_CONFIG.minZoom,
    maxZoom: GIS_CONFIG.maxZoom,
    layers: [baseTile],
    zoomControl: false,
  });
  state.basemapLayer = baseTile;

  // Zoom control top-right
  L.control.zoom({ position: 'topright' }).addTo(state.map);

  // Mouse move → coordinate inspector
  state.map.on('mousemove', (e) => {
    const { lat, lng } = e.latlng;
    document.getElementById('ci-lat').textContent = lat.toFixed(4) + '°N';
    document.getElementById('ci-lon').textContent = lng.toFixed(4) + '°E';
    document.getElementById('sb-coords').textContent = `${lat.toFixed(2)}°N, ${lng.toFixed(2)}°E`;
  });

  // Click → fetch grid cell data
  state.map.on('click', async (e) => {
    const { lat, lng } = e.latlng;
    await fetchGridCell(lat, lng);
  });

  // Zoom change → breadcrumb + status
  state.map.on('zoomend', () => {
    const z = state.map.getZoom();
    document.getElementById('sb-zoom').textContent = `Zoom: ${z}`;
    updateZoomBreadcrumb(z);
  });

  // Particle canvas setup
  const canvas = document.getElementById('particle-canvas');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  state.particles.canvas = canvas;
  state.particles.ctx = canvas.getContext('2d');

  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
}

// ══════════════════════════════════════════════════════════════
// BASEMAP SWITCH
// ══════════════════════════════════════════════════════════════

window.switchBasemap = function(name, btn) {
  if (state.basemapLayer) state.map.removeLayer(state.basemapLayer);
  const bm = BASEMAPS[name];
  state.basemapLayer = L.tileLayer(bm.url, {
    attribution: bm.attribution,
    subdomains: bm.subdomains || 'abc',
    maxZoom: 18,
  });
  state.map.addLayer(state.basemapLayer);
  state.basemapLayer.bringToBack();

  document.querySelectorAll('.basemap-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  OceanApp.toast(`Basemap: ${name}`, 'info');
};

// ══════════════════════════════════════════════════════════════
// LAYER TOGGLE
// ══════════════════════════════════════════════════════════════

window.toggleLayer = function(layerName, btn) {
  if (state.activeLayers.has(layerName)) {
    // Deactivate
    state.activeLayers.delete(layerName);
    btn.classList.remove('active');
    removeLeafletLayer(layerName);
    if (state.currentActiveLayer === layerName) {
      state.currentActiveLayer = [...state.activeLayers][0] || 'sst';
    }
    if (layerName === 'wind' || layerName === 'current') {
      stopParticlesFor(layerName);
    }
  } else {
    // Activate
    state.activeLayers.add(layerName);
    btn.classList.add('active');
    renderLayer(layerName);
    state.currentActiveLayer = layerName;
    updateLegend(layerName);
  }

  // Show depth panel for SST/SSS
  const showDepth = state.activeLayers.has('sst') || state.activeLayers.has('sss');
  document.getElementById('depth-panel').style.display = showDepth ? 'block' : 'none';

  document.getElementById('sb-layer').textContent = `Layer: ${GIS_CONFIG.LAYERS[layerName]?.label || layerName}`;
};

// ══════════════════════════════════════════════════════════════
// LAYER RENDERERS
// ══════════════════════════════════════════════════════════════

async function renderLayer(layerName) {
  switch (layerName) {
    case 'sst':      renderHeatLayer('sst', generateSSTGrid(), COLOUR_SCALES.sst); break;
    case 'sss':      renderHeatLayer('sss', generateSalinityGrid(), COLOUR_SCALES.sss); break;
    case 'sla':      renderHeatLayer('sla', generateSLAGrid(), COLOUR_SCALES.sla); break;
    case 'heatwave': {
      const hw = state.apiData.heatwave?.hotspots || DEMO_HOTSPOTS;
      renderHeatLayer('heatwave', generateHeatwaveGrid(hw), null, {
        radius: 35, blur: 25, maxZoom: 8,
        gradient: heatmapGradient(COLOUR_SCALES.heatwave),
      });
      renderHeatwaveMarkers(hw);
      break;
    }
    case 'current':  startCurrentParticles(); break;
    case 'wind':     startWindParticles(); break;
    case 'bathymetry': renderBathymetry(); break;
    case 'shipping': renderShippingRoutes(); break;
    case 'mpa':      renderMPA(); break;
    case 'eez':      renderEEZ(); break;
    case 'argo':     await renderArgoFloats(); break;
    case 'ports':    renderPorts(); break;
  }
}

function removeLeafletLayer(name) {
  if (state.leafletLayers[name]) {
    if (Array.isArray(state.leafletLayers[name])) {
      state.leafletLayers[name].forEach(l => state.map.removeLayer(l));
    } else {
      state.map.removeLayer(state.leafletLayers[name]);
    }
    delete state.leafletLayers[name];
  }
}

function renderHeatLayer(name, pts, scale, opts = {}) {
  removeLeafletLayer(name);
  const defaultOpts = {
    radius: 20,
    blur: 15,
    maxZoom: 10,
    max: 1.0,
    gradient: scale ? heatmapGradient(scale) : undefined,
  };
  const heat = L.heatLayer(pts, { ...defaultOpts, ...opts });
  heat.addTo(state.map);
  state.leafletLayers[name] = heat;
}

// ══════════════════════════════════════════════════════════════
// HEATWAVE MARKERS
// ══════════════════════════════════════════════════════════════

const DEMO_HOTSPOTS = [
  { latitude: 15.5, longitude: 89.0, severity: 'EXTREME', sst_anomaly: 4.2 },
  { latitude: 12.0, longitude: 86.0, severity: 'HIGH',    sst_anomaly: 2.8 },
  { latitude: 18.0, longitude: 64.0, severity: 'HIGH',    sst_anomaly: 2.1 },
  { latitude: 8.0,  longitude: 76.0, severity: 'LOW',     sst_anomaly: 1.2 },
];

function renderHeatwaveMarkers(hotspots) {
  const markers = [];
  for (const hs of hotspots) {
    const color = hs.severity === 'EXTREME' ? '#ff0844' :
                  hs.severity === 'HIGH'    ? '#ff3b5c' : '#ffb020';
    const size = hs.severity === 'EXTREME' ? 16 : hs.severity === 'HIGH' ? 12 : 9;
    const icon = L.divIcon({
      html: `<div style="
        width:${size}px;height:${size}px;
        border-radius:50%;
        background:${color};
        box-shadow:0 0 ${size*2}px ${color};
        animation:pulse-glow 1s infinite;
        border:2px solid rgba(255,255,255,0.4);
      "></div>`,
      className: '',
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
    const m = L.marker([hs.latitude, hs.longitude], { icon })
      .bindPopup(`
        <div style="font-family:'Rajdhani',sans-serif;min-width:180px">
          <div style="font-family:'Orbitron',sans-serif;font-size:0.7rem;color:#22e5ff;margin-bottom:8px">
            🌡️ MARINE HEATWAVE
          </div>
          <div style="margin-bottom:4px"><b>Severity:</b>
            <span style="color:${color};font-weight:700">${hs.severity}</span>
          </div>
          <div><b>SST Anomaly:</b> +${hs.sst_anomaly?.toFixed(1) || '—'}°C</div>
          <div style="font-size:0.72rem;color:#9db4cc;margin-top:6px">
            ${hs.latitude.toFixed(2)}°N, ${hs.longitude.toFixed(2)}°E
          </div>
        </div>
      `)
      .addTo(state.map);
    markers.push(m);
  }
  if (!state.leafletLayers['heatwave']) state.leafletLayers['heatwave'] = [];
  state.leafletLayers['heatwave'].push(...markers);
}

// ══════════════════════════════════════════════════════════════
// PARTICLE SYSTEM — wind & ocean currents
// ══════════════════════════════════════════════════════════════

const PARTICLE_COUNT = 300;

function makeParticle(type) {
  const { latMin, latMax, lonMin, lonMax } = GIS_CONFIG.bounds;
  const lat = latMin + Math.random() * (latMax - latMin);
  const lon = lonMin + Math.random() * (lonMax - lonMin);
  const angle = type === 'wind'
    ? 200 + Math.random() * 80   // SW monsoon dominant
    : 50 + Math.random() * 30;   // NE current
  const speed = type === 'wind'
    ? 0.04 + Math.random() * 0.08
    : 0.01 + Math.random() * 0.03;
  return {
    lat, lon, angle, speed,
    age: Math.random() * 100, maxAge: 80 + Math.random() * 60,
    color: type === 'wind' ? '#2ee6a8' : '#22e5ff',
  };
}

function startWindParticles() {
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    state.particles.wind.push(makeParticle('wind'));
  }
  requestParticleFrame();
}

function startCurrentParticles() {
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    state.particles.current.push(makeParticle('current'));
  }
  requestParticleFrame();
}

function stopParticlesFor(type) {
  if (type === 'wind') state.particles.wind = [];
  if (type === 'current') state.particles.current = [];
  if (state.particles.wind.length === 0 && state.particles.current.length === 0) {
    if (state.particles.animId) cancelAnimationFrame(state.particles.animId);
    state.particles.animId = null;
    const ctx = state.particles.ctx;
    ctx.clearRect(0, 0, state.particles.canvas.width, state.particles.canvas.height);
  }
}

function requestParticleFrame() {
  if (state.particles.animId) return;
  function frame() {
    drawParticles();
    state.particles.animId = requestAnimationFrame(frame);
  }
  frame();
}

function drawParticles() {
  const ctx = state.particles.ctx;
  const W = state.particles.canvas.width;
  const H = state.particles.canvas.height;
  ctx.clearRect(0, 0, W, H);

  const allParticles = [
    ...state.particles.wind.map(p => ({ ...p, type: 'wind' })),
    ...state.particles.current.map(p => ({ ...p, type: 'current' })),
  ];

  for (const p of allParticles) {
    const pt = state.map.latLngToContainerPoint([p.lat, p.lon]);
    const alpha = Math.min(1, (1 - p.age / p.maxAge)) * 0.75;
    const angleRad = (p.angle * Math.PI) / 180;
    const dx = Math.cos(angleRad);
    const dy = -Math.sin(angleRad);

    ctx.beginPath();
    ctx.strokeStyle = p.color.replace(')', `,${alpha})`).replace('rgb', 'rgba').replace('#', 'rgba(').replace('22e5ff', '34,229,255,').replace('2ee6a8', '46,230,168,');
    ctx.lineWidth = p.type === 'wind' ? 1.5 : 2;
    ctx.moveTo(pt.x - dx * 6, pt.y - dy * 6);
    ctx.lineTo(pt.x + dx * 6, pt.y + dy * 6);
    ctx.stroke();

    // Move particle
    p.lat += dy * p.speed;
    p.lon += dx * p.speed * 1.2;
    p.age++;

    // Reset if old or out of bounds
    if (p.age > p.maxAge ||
        p.lat < GIS_CONFIG.bounds.latMin || p.lat > GIS_CONFIG.bounds.latMax ||
        p.lon < GIS_CONFIG.bounds.lonMin || p.lon > GIS_CONFIG.bounds.lonMax) {
      const fresh = makeParticle(p.type);
      Object.assign(p, fresh);
    }
  }
}

// ══════════════════════════════════════════════════════════════
// GIS VECTOR LAYERS
// ══════════════════════════════════════════════════════════════

function renderBathymetry() {
  // Ocean depth contours — representative for N Indian Ocean
  const contours = [
    // 200m shelf edge - India west coast
    { depth: '200m Shelf', color: '#1565c0', coords: [[22,68],[20,66],[16,72],[12,74],[8,76],[6,79]] },
    // 1000m
    { depth: '1000m', color: '#0d47a1', coords: [[21,67],[18,64],[14,70],[10,73],[7,77],[5,80]] },
    // Ninety East Ridge (BoB)
    { depth: 'Ninety East Ridge', color: '#4a90d9', coords: [[-5,90],[0,90],[5,90],[10,90],[15,90],[20,90]] },
    // Carlsberg Ridge
    { depth: 'Carlsberg Ridge', color: '#4a90d9', coords: [[5,57],[8,60],[10,63],[12,66]] },
  ];

  const layers = contours.map(c =>
    L.polyline(c.coords, { color: c.color, weight: 1.5, opacity: 0.6, dashArray: '4 4' })
      .bindTooltip(`<b>${c.depth}</b>`, { sticky: true })
      .addTo(state.map)
  );

  // Mariana-style trench highlight for Sunda Trench
  const trench = L.polyline(
    [[-10,105],[-5,102],[0,99],[5,97]],
    { color: '#0a1f5c', weight: 3, opacity: 0.8 }
  ).bindTooltip('<b>Sunda Trench (~7300m)</b>', { sticky: true }).addTo(state.map);

  state.leafletLayers['bathymetry'] = [...layers, trench];
}

function renderShippingRoutes() {
  const routes = [
    // Strait of Malacca → Arabian Sea (main VLCC route)
    { name: 'Malacca → Arabian Sea', coords: [[2,105],[5,95],[10,82],[15,70],[20,60],[22,57]] },
    // India south → Europe
    { name: 'India → Suez', coords: [[8,77],[10,65],[12,57],[15,50],[20,45]] },
    // India west coast → East Africa
    { name: 'Mumbai → Mombasa', coords: [[19,72],[15,68],[10,62],[5,55],[0,48],[-4,42]] },
    // BoB routes
    { name: 'Kolkata → Singapore', coords: [[22,90],[15,95],[5,100],[2,105]] },
    // Colombo hub
    { name: 'Colombo Hub', coords: [[6,80],[8,77],[7,74],[6,72]] },
  ];

  const layers = routes.map(r =>
    L.polyline(r.coords, { color: '#f1c40f', weight: 2, opacity: 0.55, dashArray: '8 5' })
      .bindTooltip(`🚢 ${r.name}`, { sticky: true })
      .addTo(state.map)
  );
  state.leafletLayers['shipping'] = layers;
}

const MPA_ZONES = [
  { name: 'Lakshadweep Sea MPA', lat: 11, lon: 73, radius: 80000 },
  { name: 'Gulf of Mannar Biosphere Reserve', lat: 9, lon: 79, radius: 50000 },
  { name: 'Andaman & Nicobar MPA', lat: 12, lon: 93, radius: 120000 },
  { name: 'Chagos Marine Reserve', lat: -6, lon: 72, radius: 200000 },
  { name: 'Maldives Marine Reserve', lat: 4, lon: 73, radius: 90000 },
  { name: 'Oman Sea MPA', lat: 22, lon: 57, radius: 70000 },
];

function renderMPA() {
  const layers = MPA_ZONES.map(z =>
    L.circle([z.lat, z.lon], {
      radius: z.radius,
      color: '#27ae60', weight: 1.5, opacity: 0.8,
      fillColor: '#27ae60', fillOpacity: 0.12,
    })
    .bindPopup(`
      <div style="font-family:'Rajdhani',sans-serif">
        <div style="color:#2ee6a8;font-weight:700;margin-bottom:4px">🌿 ${z.name}</div>
        <div style="font-size:0.8rem;color:#9db4cc">Marine Protected Area</div>
        <div style="font-size:0.75rem;color:#9db4cc">${z.lat.toFixed(2)}°N, ${z.lon.toFixed(2)}°E</div>
      </div>
    `)
    .addTo(state.map)
  );
  state.leafletLayers['mpa'] = layers;
}

const EEZ_ZONES = [
  { name: 'India EEZ', color: '#e67e22', coords: [
    [22,68],[22,90],[15,95],[8,90],[5,80],[8,72],[12,70],[18,60],[22,60],[22,68]
  ]},
  { name: 'Sri Lanka EEZ', color: '#e67e22', coords: [
    [10,76],[5,76],[5,83],[10,83],[10,76]
  ]},
  { name: 'Bangladesh EEZ', color: '#e67e22', coords: [
    [22,88],[22,92],[18,92],[18,88],[22,88]
  ]},
  { name: 'Myanmar EEZ', color: '#e67e22', coords: [
    [22,92],[22,100],[16,98],[16,92],[22,92]
  ]},
];

function renderEEZ() {
  const layers = EEZ_ZONES.map(z =>
    L.polygon(z.coords, {
      color: z.color, weight: 2, opacity: 0.7,
      fillColor: z.color, fillOpacity: 0.05,
      dashArray: '6 4',
    })
    .bindTooltip(`🗺️ ${z.name}`, { sticky: true })
    .addTo(state.map)
  );
  state.leafletLayers['eez'] = layers;
}

const MAJOR_PORTS = [
  { name: 'Mumbai', lat: 18.92, lon: 72.83, country: 'India', type: 'Major Commercial' },
  { name: 'Chennai', lat: 13.08, lon: 80.28, country: 'India', type: 'Container Port' },
  { name: 'Kolkata', lat: 22.56, lon: 88.37, country: 'India', type: 'River Port' },
  { name: 'Colombo', lat: 6.93, lon: 79.87, country: 'Sri Lanka', type: 'Trans-shipment Hub' },
  { name: 'Chittagong', lat: 22.32, lon: 91.82, country: 'Bangladesh', type: 'Container Port' },
  { name: 'Karachi', lat: 24.86, lon: 67.01, country: 'Pakistan', type: 'Major Commercial' },
  { name: 'Muscat', lat: 23.61, lon: 58.59, country: 'Oman', type: 'Commercial Port' },
  { name: 'Aden', lat: 12.78, lon: 45.03, country: 'Yemen', type: 'Transit Port' },
  { name: 'Mombasa', lat: -4.05, lon: 39.67, country: 'Kenya', type: 'E. Africa Hub' },
  { name: 'Port Louis', lat: -20.16, lon: 57.50, country: 'Mauritius', type: 'Regional Hub' },
  { name: 'Singapore', lat: 1.26, lon: 103.83, country: 'Singapore', type: 'World #1 Hub' },
  { name: 'Visakhapatnam', lat: 17.69, lon: 83.22, country: 'India', type: 'Industrial Port' },
];

function renderPorts() {
  const icon = L.divIcon({
    html: `<div style="
      width:10px;height:10px;
      background:#e74c3c;
      border-radius:2px;
      border:2px solid rgba(231,76,60,0.5);
      box-shadow:0 0 8px rgba(231,76,60,0.6);
    "></div>`,
    className: '',
    iconSize: [10, 10],
    iconAnchor: [5, 5],
  });

  const layers = MAJOR_PORTS.map(p =>
    L.marker([p.lat, p.lon], { icon })
      .bindPopup(`
        <div style="font-family:'Rajdhani',sans-serif;min-width:160px">
          <div style="font-family:'Orbitron',sans-serif;font-size:0.7rem;color:#22e5ff;margin-bottom:6px">
            ⚓ ${p.name}
          </div>
          <div><b>Country:</b> ${p.country}</div>
          <div><b>Type:</b> ${p.type}</div>
          <div style="font-size:0.72rem;color:#9db4cc;margin-top:4px">
            ${p.lat.toFixed(2)}°, ${p.lon.toFixed(2)}°
          </div>
        </div>
      `)
      .addTo(state.map)
  );
  state.leafletLayers['ports'] = layers;
}

async function renderArgoFloats() {
  let floats = [];
  try {
    const data = await OceanApp.api('/api/argo/floats');
    floats = data.floats || [];
  } catch (e) {
    // Use demo data if API not available
    floats = generateDemoArgoFloats();
  }

  const icon = L.divIcon({
    html: `<div style="
      width:9px;height:9px;
      background:#8e44ad;
      border-radius:50%;
      border:2px solid rgba(142,68,173,0.4);
      box-shadow:0 0 8px rgba(142,68,173,0.7);
    "></div>`,
    className: '',
    iconSize: [9, 9],
    iconAnchor: [4, 4],
  });

  const markers = floats.slice(0, 80).map(f => {
    if (!f.latitude || !f.longitude) return null;
    return L.marker([f.latitude, f.longitude], { icon })
      .bindPopup(`
        <div style="font-family:'Rajdhani',sans-serif">
          <div style="color:#9b59b6;font-weight:700;margin-bottom:4px">🔬 ARGO Float</div>
          <div style="font-size:0.78rem"><b>ID:</b> ${f.float_id || 'unknown'}</div>
          <div style="font-size:0.72rem;color:#9db4cc">
            ${f.latitude?.toFixed(3)}°N, ${f.longitude?.toFixed(3)}°E
          </div>
        </div>
      `)
      .addTo(state.map);
  }).filter(Boolean);

  state.leafletLayers['argo'] = markers;
  document.getElementById('stat-argo').textContent = markers.length;
  OceanApp.toast(`${markers.length} ARGO floats loaded`, 'success');
}

function generateDemoArgoFloats() {
  const pts = [];
  const { latMin, latMax, lonMin, lonMax } = GIS_CONFIG.bounds;
  for (let i = 0; i < 60; i++) {
    pts.push({
      float_id: `ARGO_${5900000 + i}`,
      latitude: latMin + Math.random() * (latMax - latMin),
      longitude: lonMin + Math.random() * (lonMax - lonMin),
    });
  }
  return pts;
}

// ══════════════════════════════════════════════════════════════
// LEGEND UPDATE
// ══════════════════════════════════════════════════════════════

function updateLegend(layerName) {
  const cfg = GIS_CONFIG.LAYERS[layerName];
  if (!cfg) return;

  document.getElementById('legend-title').textContent = `${cfg.label}`;

  const scale = COLOUR_SCALES[layerName];
  if (scale) {
    const bar = document.getElementById('legend-bar');
    bar.style.background = `linear-gradient(to right, ${scale.join(', ')})`;
  }

  document.getElementById('legend-min').textContent = cfg.minVal + (cfg.unit ? ` ${cfg.unit}` : '');
  document.getElementById('legend-mid').textContent = ((cfg.minVal + cfg.maxVal) / 2).toFixed(0) + (cfg.unit ? ` ${cfg.unit}` : '');
  document.getElementById('legend-max').textContent = cfg.maxVal + (cfg.unit ? ` ${cfg.unit}` : '');
}

// ══════════════════════════════════════════════════════════════
// COORDINATE INSPECTOR — click
// ══════════════════════════════════════════════════════════════

async function fetchGridCell(lat, lon) {
  try {
    const data = await OceanApp.api(`/api/grid-cell?lat=${lat}&lon=${lon}`);
    document.getElementById('ci-lat').textContent = (data.latitude || lat).toFixed(4) + '°N';
    document.getElementById('ci-lon').textContent = (data.longitude || lon).toFixed(4) + '°E';
    document.getElementById('ci-sst').textContent = data.sst != null ? data.sst.toFixed(2) + '°C' : '—';
    document.getElementById('ci-sss').textContent = data.sss != null ? data.sss.toFixed(2) + ' PSU' : '—';
    document.getElementById('ci-sla').textContent = data.sla != null ? (data.sla * 100).toFixed(1) + ' cm' : '—';
    document.getElementById('ci-conf').textContent = data.prediction_confidence != null
      ? (data.prediction_confidence * 100).toFixed(0) + '%' : '—';

    // Heatwave badge
    const hw = data.heatwave_status || 'NONE';
    const hwEl = document.getElementById('ci-hw');
    hwEl.innerHTML = `<span class="heatwave-badge ${hw}">${hw}</span>`;

    // Depth profile fetch
    loadMiniDepthProfile(lat, lon);

    // Ripple on map
    L.circleMarker([lat, lon], {
      radius: 10, color: '#22e5ff', fillOpacity: 0, weight: 2, opacity: 0.7
    }).addTo(state.map).on('add', function() {
      setTimeout(() => state.map.removeLayer(this), 2000);
    });

  } catch (e) {
    // Populate with simulated data
    const sst = (27 + 4 * Math.random()).toFixed(2);
    document.getElementById('ci-sst').textContent = sst + '°C';
    document.getElementById('ci-sss').textContent = (34 + Math.random()).toFixed(2) + ' PSU';
    document.getElementById('ci-sla').textContent = ((Math.random() * 20 - 10)).toFixed(1) + ' cm';
    document.getElementById('ci-conf').textContent = (70 + Math.random() * 25).toFixed(0) + '%';
    document.getElementById('ci-hw').innerHTML = `<span class="heatwave-badge NONE">NONE</span>`;
  }
}

async function loadMiniDepthProfile(lat, lon) {
  const card = document.getElementById('depth-profile-card');
  const canvas = document.getElementById('mini-depth-chart');
  card.style.display = 'block';

  let profile = [], depths = [];
  try {
    const data = await OceanApp.api(`/api/depth-profile?lat=${lat}&lon=${lon}`);
    profile = data.predicted || data.depth_profile || [];
    depths = data.depths || GIS_CONFIG.DEPTHS_M;
  } catch (e) {
    depths = GIS_CONFIG.DEPTHS_M;
    profile = depths.map((d, i) => 29 - i * 0.8 + Math.random() * 0.5);
  }

  if (window._miniChart) window._miniChart.destroy();
  window._miniChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: depths,
      datasets: [{
        label: 'Temp (°C)',
        data: profile,
        borderColor: '#22e5ff',
        backgroundColor: 'rgba(34,229,255,0.1)',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.4,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#9db4cc', font: { size: 8 } },
          grid: { color: 'rgba(34,229,255,0.07)' },
        },
        y: {
          reverse: false,
          ticks: { color: '#9db4cc', font: { size: 8 } },
          grid: { color: 'rgba(34,229,255,0.07)' },
        }
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════
// DEPTH SLIDER
// ══════════════════════════════════════════════════════════════

window.onDepthChange = function(val) {
  state.depthIdx = parseInt(val);
  const m = GIS_CONFIG.DEPTHS_M[state.depthIdx] || 0;
  document.getElementById('depth-label').textContent = `${m} m`;

  // Refresh active data layers at new depth
  if (state.activeLayers.has('sst')) {
    removeLeafletLayer('sst');
    const pts = generateSSTGrid().map(([lat, lon, v]) => [lat, lon, Math.max(0, v - state.depthIdx * 0.025)]);
    renderHeatLayer('sst', pts, COLOUR_SCALES.sst);
  }
};

// ══════════════════════════════════════════════════════════════
// ZOOM BREADCRUMB
// ══════════════════════════════════════════════════════════════

function updateZoomBreadcrumb(z) {
  const crumbWorld = document.getElementById('crumb-world');
  const crumbIO    = document.getElementById('crumb-io');
  const crumbNIO   = document.getElementById('crumb-nio');
  const crumbBoB   = document.getElementById('crumb-bob');

  crumbWorld.classList.toggle('active', z <= 3);
  crumbIO.classList.toggle('active',    z >= 4 && z <= 5);
  crumbNIO.classList.toggle('active',   z >= 6 && z <= 7);
  crumbBoB.classList.toggle('active',   z >= 8);
}

// ══════════════════════════════════════════════════════════════
// QUICK FLY
// ══════════════════════════════════════════════════════════════

window.flyTo = function(lat, lon, zoom) {
  state.map.flyTo([lat, lon], zoom, { animate: true, duration: 1.5 });
};

// ══════════════════════════════════════════════════════════════
// LOAD LIVE API DATA
// ══════════════════════════════════════════════════════════════

async function loadApiStats() {
  try {
    const [hw, pred, status] = await Promise.all([
      OceanApp.api('/api/heatwave'),
      OceanApp.api('/api/prediction/latest').catch(() => null),
      OceanApp.api('/api/status').catch(() => null),
    ]);

    state.apiData.heatwave = hw;
    state.apiData.prediction = pred;

    const hwZones = hw?.total ?? hw?.hotspots?.length ?? '—';
    document.getElementById('stat-hw-zones').textContent = hwZones;
    document.getElementById('sb-hw').textContent = `Heatwave Zones: ${hwZones}`;

    if (pred?.surface_temp_mean != null) {
      document.getElementById('stat-mean-sst').textContent = pred.surface_temp_mean.toFixed(1) + '°C';
    }

    const modelOk = status?.model === 'loaded';
    document.getElementById('stat-model').textContent = modelOk ? '✓ Ready' : 'Loading…';
    document.getElementById('stat-model').style.color = modelOk ? '#2ee6a8' : '#ffb020';

    // Update hotspot markers with live data
    if (hw?.hotspots?.length) {
      removeLeafletLayer('heatwave-live');
      // Merge demo-like markers if heatwave layer active
      if (state.activeLayers.has('heatwave')) {
        removeLeafletLayer('heatwave');
        renderHeatLayer('heatwave', generateHeatwaveGrid(hw.hotspots), null, {
          radius: 35, blur: 25, maxZoom: 8,
          gradient: heatmapGradient(COLOUR_SCALES.heatwave),
        });
        renderHeatwaveMarkers(hw.hotspots);
      }
    }
  } catch (e) {
    // Non-fatal — show demo stats
    document.getElementById('stat-hw-zones').textContent = DEMO_HOTSPOTS.length;
    document.getElementById('stat-mean-sst').textContent = '29.4°C';
    document.getElementById('stat-model').textContent = 'Demo';
  }
}

// ══════════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
  // Init app shell (starfield, clock, toasts)
  OceanApp.init?.();

  // Loading screen progression
  const loadLabel = document.getElementById('map-loading-label');
  const setLoad = (t) => { if (loadLabel) loadLabel.textContent = t; };

  setLoad('Initialising Map Engine');
  initMap();

  setLoad('Loading Ocean Basemap');
  await new Promise(r => setTimeout(r, 400));

  setLoad('Fetching Ocean Data');
  await loadApiStats();

  setLoad('Rendering SST Heatmap');
  await new Promise(r => setTimeout(r, 200));
  renderLayer('sst');

  setLoad('Ready');
  await new Promise(r => setTimeout(r, 300));

  const loading = document.getElementById('map-loading');
  loading.classList.add('hidden');

  updateLegend('sst');
  updateZoomBreadcrumb(GIS_CONFIG.zoom);

  OceanApp.toast('GIS Ocean Map Ready', 'success');

  // Refresh stats every 60s
  setInterval(loadApiStats, 60_000);
});
