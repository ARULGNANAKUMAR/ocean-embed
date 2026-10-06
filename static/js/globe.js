/* ═══════════════════════════════════════════════════════════
   globe.js — Three.js holographic Earth (Section 2)
   Auto-rotate, drag rotate, wheel zoom, double-click zoom,
   ocean glow highlight, cloud layer, atmosphere, starfield.
   ═══════════════════════════════════════════════════════════ */

const OceanGlobe = (() => {
  let scene, camera, renderer, controls;
  let earthMesh, cloudMesh, atmosphereMesh, glowMesh;
  let raycaster, mouse;
  let autoRotate = true;
  let container;
  let animationId;
  let onCellClickCallback = null;
  let gridGroup = null;
  let argoGroup = null;

  const EARTH_RADIUS = 5;

  function init(containerId) {
    container = document.getElementById(containerId);
    if (!container) return null;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(
      45, container.clientWidth / container.clientHeight, 0.1, 1000
    );
    camera.position.set(0, 0, 14);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // ── Lighting ──
    const ambient = new THREE.AmbientLight(0x445577, 1.4);
    scene.add(ambient);
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(6, 3, 8);
    scene.add(sun);
    const rim = new THREE.DirectionalLight(0x22e5ff, 0.6);
    rim.position.set(-8, -2, -6);
    scene.add(rim);

    buildEarth();
    buildClouds();
    buildAtmosphere();
    buildStarfield();

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // ── Controls: OrbitControls-style manual drag/zoom (no external dep) ──
    initManualControls();

    window.addEventListener("resize", onResize);
    container.addEventListener("dblclick", onDoubleClick);
    container.addEventListener("mousemove", onMouseMove);
    container.addEventListener("click", onClick);

    animate();
    return { scene, camera, renderer };
  }

  function buildEarth() {
    const geo = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);

    // Procedural ocean/land material — ocean glowing blue, land darker.
    // Uses a simple gradient shader keyed on a land/ocean noise texture
    // baked at runtime (no external texture dependency required).
    const canvas = document.createElement("canvas");
    canvas.width = 1024; canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Base ocean gradient
    const grad = ctx.createLinearGradient(0, 0, 0, 512);
    grad.addColorStop(0, "#04203f");
    grad.addColorStop(0.5, "#063868");
    grad.addColorStop(1, "#04203f");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 1024, 512);

    // Simple procedural "continents" — darker blobs (approx real coastlines is
    // out of scope without texture assets; this gives visual land/ocean contrast)
    ctx.fillStyle = "#0a1420";
    const continents = [
      // Rough Africa/Eurasia/India band placement (equirectangular)
      [520, 150, 90, 130], [560, 230, 60, 90], // Africa-ish
      [620, 120, 140, 70],                       // Asia-ish
      [640, 190, 50, 40],                        // India-ish
      [120, 140, 100, 90], [100, 250, 70, 110],  // Americas-ish
      [780, 300, 90, 60],                        // Australia-ish
      [480, 60, 160, 50],                        // Europe/N.Asia-ish
    ];
    continents.forEach(([x, y, w, h]) => {
      ctx.beginPath();
      ctx.ellipse(x, y, w / 2, h / 2, 0, 0, Math.PI * 2);
      ctx.fill();
    });

    // Highlight North Indian Ocean region with cyan glow patch
    const nioGrad = ctx.createRadialGradient(660, 230, 5, 660, 230, 110);
    nioGrad.addColorStop(0, "rgba(34,229,255,0.55)");
    nioGrad.addColorStop(1, "rgba(34,229,255,0)");
    ctx.fillStyle = nioGrad;
    ctx.beginPath();
    ctx.ellipse(660, 230, 110, 90, 0, 0, Math.PI * 2);
    ctx.fill();

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;

    const mat = new THREE.MeshPhongMaterial({
      map: texture,
      emissive: new THREE.Color(0x0a3a5c),
      emissiveIntensity: 0.35,
      shininess: 18,
      specular: new THREE.Color(0x1a5c8c),
    });

    earthMesh = new THREE.Mesh(geo, mat);
    scene.add(earthMesh);

    // Wireframe grid overlay (subtle lat/lon lines)
    const wireGeo = new THREE.SphereGeometry(EARTH_RADIUS + 0.01, 24, 16);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x22e5ff, wireframe: true, transparent: true, opacity: 0.06,
    });
    scene.add(new THREE.Mesh(wireGeo, wireMat));
  }

  function buildClouds() {
    const geo = new THREE.SphereGeometry(EARTH_RADIUS + 0.06, 48, 48);
    const canvas = document.createElement("canvas");
    canvas.width = 512; canvas.height = 256;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 512, 256);
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    for (let i = 0; i < 60; i++) {
      const x = Math.random() * 512, y = Math.random() * 256;
      const r = Math.random() * 18 + 6;
      ctx.beginPath();
      ctx.ellipse(x, y, r, r * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.MeshPhongMaterial({
      map: texture, transparent: true, opacity: 0.35, depthWrite: false,
    });
    cloudMesh = new THREE.Mesh(geo, mat);
    scene.add(cloudMesh);
  }

  function buildAtmosphere() {
    const geo = new THREE.SphereGeometry(EARTH_RADIUS + 0.35, 48, 48);
    const mat = new THREE.ShaderMaterial({
      uniforms: {},
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.65 - dot(vNormal, vec3(0,0,1.0)), 3.0);
          gl_FragColor = vec4(0.13, 0.9, 1.0, 1.0) * intensity;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });
    atmosphereMesh = new THREE.Mesh(geo, mat);
    scene.add(atmosphereMesh);
  }

  function buildStarfield() {
    const geo = new THREE.BufferGeometry();
    const count = 2500;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 80 + Math.random() * 120;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0xaaccff, size: 0.4, sizeAttenuation: true });
    scene.add(new THREE.Points(geo, mat));
  }

  // ── Manual orbit-style controls (drag rotate, wheel zoom) ──
  let isDragging = false, prevX = 0, prevY = 0;
  let rotX = 0, rotY = 0;

  function initManualControls() {
    container.addEventListener("mousedown", (e) => {
      isDragging = true; autoRotate = false;
      prevX = e.clientX; prevY = e.clientY;
    });
    window.addEventListener("mouseup", () => { isDragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevX, dy = e.clientY - prevY;
      rotY += dx * 0.005;
      rotX += dy * 0.005;
      rotX = Math.max(-1.3, Math.min(1.3, rotX));
      prevX = e.clientX; prevY = e.clientY;
    });
    container.addEventListener("wheel", (e) => {
      e.preventDefault();
      const dist = camera.position.length();
      const newDist = THREE.MathUtils.clamp(dist + e.deltaY * 0.01, 6.5, 40);
      camera.position.setLength(newDist);
    }, { passive: false });

    // Touch support
    let touchStartDist = null;
    container.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) {
        isDragging = true; autoRotate = false;
        prevX = e.touches[0].clientX; prevY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        touchStartDist = getTouchDist(e);
      }
    });
    container.addEventListener("touchmove", (e) => {
      if (e.touches.length === 1 && isDragging) {
        const dx = e.touches[0].clientX - prevX, dy = e.touches[0].clientY - prevY;
        rotY += dx * 0.005; rotX += dy * 0.005;
        rotX = Math.max(-1.3, Math.min(1.3, rotX));
        prevX = e.touches[0].clientX; prevY = e.touches[0].clientY;
      } else if (e.touches.length === 2 && touchStartDist) {
        const newDist = getTouchDist(e);
        const scale = touchStartDist / newDist;
        const dist = THREE.MathUtils.clamp(camera.position.length() * scale, 6.5, 40);
        camera.position.setLength(dist);
        touchStartDist = newDist;
      }
    }, { passive: true });
    container.addEventListener("touchend", () => { isDragging = false; touchStartDist = null; });
  }
  function getTouchDist(e) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function onDoubleClick(e) {
    // Zoom toward clicked point (Section 2: double click zooms)
    const dist = camera.position.length();
    const newDist = Math.max(6.5, dist * 0.55);
    camera.position.setLength(newDist);
  }

  function onMouseMove(e) {
    const rect = container.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    // Update live cursor lat/lon readout (Section 4)
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(earthMesh);
    const coordEl = document.getElementById("cursor-coords");
    if (hits.length > 0 && coordEl) {
      const p = hits[0].point.clone();
      earthMesh.worldToLocal(p);
      const { lat, lon } = vectorToLatLon(p);
      coordEl.textContent = `LAT ${lat.toFixed(2)}° · LON ${lon.toFixed(2)}°`;
    }
  }

  function onClick(e) {
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(earthMesh);
    if (hits.length > 0 && onCellClickCallback) {
      const p = hits[0].point.clone();
      earthMesh.worldToLocal(p);
      const { lat, lon } = vectorToLatLon(p);
      onCellClickCallback(lat, lon);
    }
  }

  function vectorToLatLon(vec) {
    const r = vec.length();
    const lat = 90 - (Math.acos(vec.y / r) * 180) / Math.PI;
    let lon = (Math.atan2(vec.z, -vec.x) * 180) / Math.PI - 90;
    if (lon < -180) lon += 360;
    if (lon > 180) lon -= 360;
    return { lat, lon };
  }

  function latLonToVector3(lat, lon, radius) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 90) * (Math.PI / 180);
    return new THREE.Vector3(
      -radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta)
    );
  }

  // ── Draw 0.25° grid overlay for North Indian Ocean region (Section 5) ──
  function drawOceanGrid(latMin, latMax, lonMin, lonMax, step = 1.0) {
    if (gridGroup) scene.remove(gridGroup);
    gridGroup = new THREE.Group();
    const mat = new THREE.LineBasicMaterial({ color: 0x22e5ff, transparent: true, opacity: 0.35 });

    for (let lat = latMin; lat <= latMax; lat += step) {
      const pts = [];
      for (let lon = lonMin; lon <= lonMax; lon += 0.5) {
        pts.push(latLonToVector3(lat, lon, EARTH_RADIUS + 0.02));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      gridGroup.add(new THREE.Line(geo, mat));
    }
    for (let lon = lonMin; lon <= lonMax; lon += step) {
      const pts = [];
      for (let lat = latMin; lat <= latMax; lat += 0.5) {
        pts.push(latLonToVector3(lat, lon, EARTH_RADIUS + 0.02));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      gridGroup.add(new THREE.Line(geo, mat));
    }
    scene.add(gridGroup);
  }

  function clearOceanGrid() {
    if (gridGroup) { scene.remove(gridGroup); gridGroup = null; }
  }

  // ── ARGO float markers (Section 9) ──
  function drawArgoFloats(floats) {
    if (argoGroup) scene.remove(argoGroup);
    argoGroup = new THREE.Group();
    const geo = new THREE.SphereGeometry(0.035, 8, 8);
    const mat = new THREE.MeshBasicMaterial({ color: 0x22e5ff });

    floats.forEach((f) => {
      if (f.latitude == null || f.longitude == null) return;
      const pos = latLonToVector3(f.latitude, f.longitude, EARTH_RADIUS + 0.03);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(pos);
      mesh.userData = f;
      argoGroup.add(mesh);
    });
    scene.add(argoGroup);
  }

  function getArgoGroup() { return argoGroup; }

  // ── Camera fly-to (Section 3 — cinematic zoom sequence) ──
  function flyTo(lat, lon, distance, duration = 1800, onComplete) {
    const targetPos = latLonToVector3(lat, lon, distance);
    const startPos = camera.position.clone();
    const startTime = performance.now();

    function step(now) {
      const t = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      camera.position.lerpVectors(startPos, targetPos, eased);
      camera.lookAt(0, 0, 0);
      if (t < 1) requestAnimationFrame(step);
      else if (onComplete) onComplete();
    }
    autoRotate = false;
    requestAnimationFrame(step);
  }

  function resetView(duration = 1500) {
    flyTo(0, 0, 14, duration);
    rotX = 0; rotY = 0;
    setTimeout(() => { autoRotate = true; }, duration + 100);
  }

  function onResize() {
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    if (autoRotate) {
      rotY += 0.0011;
    }
    if (earthMesh) {
      earthMesh.rotation.y = rotY;
      earthMesh.rotation.x = rotX;
    }
    if (cloudMesh) {
      cloudMesh.rotation.y = rotY * 1.15;
      cloudMesh.rotation.x = rotX;
    }
    if (gridGroup) {
      gridGroup.rotation.y = rotY;
      gridGroup.rotation.x = rotX;
    }
    if (argoGroup) {
      argoGroup.rotation.y = rotY;
      argoGroup.rotation.x = rotX;
    }
    renderer.render(scene, camera);
  }

  function setAutoRotate(v) { autoRotate = v; }
  function onCellClick(cb) { onCellClickCallback = cb; }

  function destroy() {
    if (animationId) cancelAnimationFrame(animationId);
    window.removeEventListener("resize", onResize);
    if (renderer && container) container.removeChild(renderer.domElement);
  }

  return {
    init, flyTo, resetView, setAutoRotate, onCellClick,
    drawOceanGrid, clearOceanGrid, drawArgoFloats, getArgoGroup,
    latLonToVector3, vectorToLatLon, destroy,
    getCamera: () => camera, getScene: () => scene, getRenderer: () => renderer,
    getRaycaster: () => raycaster, getMouse: () => mouse, getEarthMesh: () => earthMesh,
  };
})();
