# main.py - GEM ENGINE v26.0 Master Output
def build_map(output_html="index.html"):
  html_content = """<!DOCTYPE html>
<html lang="zh-TW" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>GEM Engine v26.0 | 草嶺古道水牛動態預判與空間戰情互動控制台</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: { orbitron: ['Orbitron', 'sans-serif'], rajdhani: ['Rajdhani', 'sans-serif'], mono: ['"JetBrains Mono"', 'monospace'] },
                    colors: { cyberDark: '#040711', cyberPanel: 'rgba(8, 14, 28, 0.94)', cyberCard: 'rgba(13, 24, 46, 0.88)', cyberBorder: '#162b4d', neonCyan: '#00f0ff', neonPurple: '#b026ff', neonAmber: '#ffaa00', neonRed: '#ff2a5f', neonGreen: '#00ff88' },
                    boxShadow: { 'neon-cyan': '0 0 15px rgba(0, 240, 255, 0.45)', 'neon-purple': '0 0 18px rgba(176, 38, 255, 0.45)', 'neon-red': '0 0 20px rgba(255, 42, 95, 0.55)', 'neon-green': '0 0 15px rgba(0, 255, 136, 0.45)' }
                }
            }
        }
    </script>
    <style>
        body { background: #040711; color: #f1f5f9; font-family: 'Rajdhani', sans-serif; -webkit-tap-highlight-color: transparent; user-select: none; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: #040711; }
        ::-webkit-scrollbar-thumb { background: #162b4d; border-radius: 2px; }
        .hud-corner-bracket { position: relative; }
        .hud-corner-bracket::before { content: ''; position: absolute; top: -1px; left: -1px; width: 8px; height: 8px; border-top: 2px solid #00f0ff; border-left: 2px solid #00f0ff; pointer-events: none; z-index: 30; }
        .hud-corner-bracket::after { content: ''; position: absolute; bottom: -1px; right: -1px; width: 8px; height: 8px; border-bottom: 2px solid #00f0ff; border-right: 2px solid #00f0ff; pointer-events: none; z-index: 30; }
        .leaflet-container { background: #040711 !important; font-family: 'Rajdhani', sans-serif !important; width: 100% !important; height: 100% !important; }
        body.force-mobile #desktop-root { display: none !important; }
        body.force-mobile #mobile-root { display: flex !important; }
        body.force-desktop #desktop-root { display: grid !important; }
        body.force-desktop #mobile-root { display: none !important; }
        #leafletMapDesk, #leafletMapMobile { position: absolute !important; top: 0; left: 0; right: 0; bottom: 0; width: 100% !important; height: 100% !important; z-index: 10; }
    </style>
</head>
<body class="bg-cyberDark text-slate-100 font-rajdhani min-h-screen flex flex-col overflow-x-hidden">

    <header class="border-b border-cyberBorder bg-cyberPanel backdrop-blur-md px-3 py-2 flex items-center justify-between sticky top-0 z-50 shrink-0">
        <div class="flex items-center gap-2">
            <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-600 via-indigo-600 to-neonPurple flex items-center justify-center shadow-neon-cyan text-white text-xs">
                <i class="fa-solid fa-radar fa-spin"></i>
            </div>
            <div>
                <h1 class="font-orbitron font-extrabold text-xs md:text-sm tracking-wider text-white flex items-center gap-1">
                    GEM<span class="text-neonCyan">ENGINE</span> <span class="text-[8px] px-1 py-0.2 rounded bg-purple-950 text-purple-300 border border-neonPurple/50 font-mono">v26.0 DUAL-UI</span>
                </h1>
                <div class="flex items-center gap-1 text-[9px] text-slate-400 font-mono">
                    <span class="w-1.5 h-1.5 rounded-full bg-neonGreen animate-ping"></span>
                    <span>草嶺古道埡口 348m 實體山脈動態全域對齊</span>
                </div>
            </div>
        </div>
        <div class="flex items-center bg-cyberDark/90 p-0.5 rounded-lg border border-cyberBorder text-[10px] font-mono">
            <button id="btn-device-auto" class="px-2 py-0.5 rounded bg-cyan-950 text-neonCyan font-bold transition flex items-center gap-1"><i class="fa-solid fa-wand-magic-sparkles"></i><span class="hidden sm:inline">自適應</span></button>
            <button id="btn-device-desktop" class="px-2 py-0.5 rounded text-slate-400 hover:text-white transition flex items-center gap-1"><i class="fa-solid fa-desktop"></i><span class="hidden sm:inline">桌面版</span></button>
            <button id="btn-device-mobile" class="px-2 py-0.5 rounded text-slate-400 hover:text-white transition flex items-center gap-1"><i class="fa-solid fa-mobile-screen"></i><span class="hidden sm:inline">手機版</span></button>
        </div>
        <div class="flex items-center gap-1.5 text-xs font-mono">
            <div id="header-yi-badge" class="px-2 py-0.5 rounded text-[10px] bg-rose-950 text-rose-300 border border-neonRed/50 font-bold">老陽 (過載)</div>
        </div>
    </header>

    <div id="desktop-root" class="flex-1 hidden lg:grid lg:grid-cols-12 gap-2 p-2 md:p-3 max-w-[1920px] w-full mx-auto overflow-hidden">
        <div class="col-span-8 flex flex-col gap-2 relative min-h-[580px]">
            <div class="bg-cyberPanel rounded-xl border border-cyberBorder p-2 shadow-2xl relative flex flex-col flex-1 hud-corner-bracket">
                <div class="flex items-center justify-between pb-1.5 border-b border-cyberBorder/80 text-xs font-mono text-[11px]">
                    <span class="font-bold text-slate-200 tracking-wider font-orbitron">草嶺古道稜線 GIS 戰情台 (DESKTOP TACTICAL HUD)</span>
                    <select id="select-tile-layer-desk" class="bg-cyberCard text-slate-200 border border-cyberBorder rounded px-1.5 py-0.5 text-[10px] font-mono">
                        <option value="nlsc">國土測繪 (NLSC 航照)</option>
                        <option value="emap">國土測繪 (EMAP 地形)</option>
                        <option value="esri">Esri 全球高清衛星</option>
                    </select>
                </div>
                <div class="relative w-full flex-1 min-h-[440px] rounded-lg overflow-hidden mt-1.5 bg-cyberDark border border-cyberBorder/80">
                    <div id="leafletMapDesk" class="w-full h-full z-10"></div>
                    <div class="absolute top-2 left-2 bg-cyberDark/90 backdrop-blur-md border border-neonPurple/50 rounded-lg p-2 pointer-events-none text-xs font-mono space-y-0.5 z-30 shadow-neon-purple max-w-[260px]">
                        <div class="text-purple-300 font-bold text-[10px]">🧭 步道實體向量外推 (CosSim 91.5%)</div>
                        <div class="text-slate-200 text-[10px]">向量：<span class="text-neonPurple font-bold">南南東 165°</span> @ <span class="text-neonCyan">0.8 m/s</span></div>
                        <div class="text-[9px] text-slate-300 bg-cyberCard/80 p-1 rounded border border-cyberBorder">🎯 <b>終點：</b>護管所泥塘 (+25m)</div>
                    </div>
                    <div id="desk-alert-banner" class="absolute bottom-2 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-lg backdrop-blur-md border font-mono font-bold text-xs flex items-center gap-2 shadow-neon-red z-30 bg-rose-950/95 border-neonRed text-rose-200">
                        🔴 RED_ALERT: 侵入10m硬防線！水牛 7 頭 @埡口 (強制啟動低碳 E-bike 導流)
                    </div>
                </div>
                <div class="grid grid-cols-4 gap-1.5 mt-2">
                    <div class="bg-cyberCard p-2 rounded-lg border border-cyberBorder">
                        <div class="text-[10px] text-slate-400 font-mono">THI 熱應力指數</div>
                        <div class="text-2xl font-bold font-orbitron text-neonRed">81.1</div>
                    </div>
                    <div class="bg-cyberCard p-2 rounded-lg border border-cyberBorder">
                        <div class="text-[10px] text-slate-400 font-mono">衝突風險 R (Conflict)</div>
                        <div class="text-2xl font-bold font-orbitron text-neonRed">8.75</div>
                    </div>
                    <div class="bg-cyberCard p-2 rounded-lg border border-cyberBorder">
                        <div class="text-[10px] text-slate-400 font-mono">水牛頭數 (FLIR)</div>
                        <div class="text-2xl font-bold font-orbitron text-neonCyan">7 <span class="text-xs">頭</span></div>
                    </div>
                    <div class="bg-cyberCard p-2 rounded-lg border border-cyberBorder">
                        <div class="text-[10px] text-slate-400 font-mono">LinUCB 策略</div>
                        <div class="text-sm font-bold font-mono text-purple-300">Arm 1 (E-Bike)</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-span-4 flex flex-col gap-2 overflow-y-auto pr-0.5 max-h-[calc(100vh-65px)]">
            <div class="bg-cyberPanel rounded-xl border border-neonPurple/40 p-2.5 shadow-xl hud-corner-bracket">
                <div class="text-purple-300 font-bold text-[11px] font-mono mb-1.5">⏱️ 前瞻 t+1h ~ t+3h 步道橫越風險預報</div>
                <div class="grid grid-cols-3 gap-1.5 text-center text-xs font-mono">
                    <div class="bg-rose-950/40 border border-neonRed/50 p-1.5 rounded">
                        <div class="text-slate-400 text-[9px]" id="desk-t1-time">--:-- (+1h)</div>
                        <div class="text-neonRed font-bold text-sm font-orbitron">85% (高)</div>
                    </div>
                    <div class="bg-amber-950/40 border border-neonAmber/50 p-1.5 rounded">
                        <div class="text-slate-400 text-[9px]" id="desk-t2-time">--:-- (+2h)</div>
                        <div class="text-neonAmber font-bold text-sm font-orbitron">60% (中)</div>
                    </div>
                    <div class="bg-emerald-950/40 border border-neonGreen/50 p-1.5 rounded">
                        <div class="text-slate-400 text-[9px]" id="desk-t3-time">--:-- (+3h)</div>
                        <div class="text-neonGreen font-bold text-sm font-orbitron">20% (低)</div>
                    </div>
                </div>
            </div>

            <div class="bg-gradient-to-br from-cyberCard to-indigo-950/40 p-2.5 rounded-xl border border-cyan-900/60 shadow-lg hud-corner-bracket">
                <div class="font-bold text-xs text-cyan-200 font-orbitron mb-1">GEMINI 三才戰略導言</div>
                <div class="text-[11px] text-slate-300 leading-relaxed font-sans bg-cyberDark/80 p-2 rounded border border-cyberBorder/80 space-y-1">
                    <div><span class="font-mono font-bold text-neonCyan">【天時・恆卦】</span> THI 達 83.6，突破無汗腺體熱閾值。水牛沿 165° 谷線往護管所泥塘散熱。</div>
                    <div><span class="font-mono font-bold text-neonAmber">【地利・艮山】</span> 水牛距步道僅 10m，途經埡口南側，登道遭遇概率達 85% 爆發點。</div>
                    <div><span class="font-mono font-bold text-neonGreen">【人和・離火】</span> 啟動 LBS 圍欄推播與 E-bike 分流，人均減碳 8.9 kg CO₂e，名實對齊完畢。</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const mapDesk = L.map('leafletMapDesk', { zoomControl: false }).setView([24.9780, 121.9242], 16);
        L.tileLayer('https://wmts.nlsc.gov.tw/wmts/PHOTO2/default/GoogleMapsCompatible/{z}/{y}/{x}', { maxZoom: 19 }).addTo(mapDesk);

        const trail = [[25.0034, 121.9318], [24.9960, 121.9285], [24.9886, 121.9250], [24.9785, 121.9240], [24.9780, 121.9242], [24.9762, 121.9245], [24.9745, 121.9248], [24.9696, 121.9242], [24.9691, 121.9246]];
        L.polyline(trail, { color: '#2563eb', weight: 6, opacity: 0.85 }).addTo(mapDesk);
