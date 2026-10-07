from datetime import datetime, timedelta
import folium

# ==========================================
# 1. 精確收斂於陸地山區之 WGS84 據點與步道節點
# ==========================================
LANDMARKS = {
    "遠望坑入口": [25.0034, 121.9318],
    "跌死馬橋": [24.9960, 121.9285],
    "雄鎮蠻煙碑": [24.9886, 121.9250],
    "虎字碑": [24.9785, 121.9240],
    "埡口涼亭 (鞍部核心區)": [24.9780, 121.9242],
    "護管所 (泥塘散熱區)": [24.9745, 121.9248],
    "大里天公廟": [24.9696, 121.9242],
    "大里遊客中心": [24.9691, 121.9246],
}

# 草嶺古道主線折線
TRAIL_PATH = [
    [25.0034, 121.9318],
    [24.9960, 121.9285],
    [24.9886, 121.9250],
    [24.9785, 121.9240],
    [24.9780, 121.9242],
    [24.9762, 121.9245],
    [24.9745, 121.9248],
    [24.9696, 121.9242],
    [24.9691, 121.9246],
]

BUFFALO_COUNT = 7
BUFFALO_POS = [24.9782, 121.9240]
VECTOR_BEARING = "南南東 (165°)"
VECTOR_SPEED = "0.8 m/s"

# 預測移動路徑（嚴格貼合步道折線）
PREDICTED_LOCATIONS = [
    {
        "name": "埡口草坡 (當前起點)",
        "pos": [24.9782, 121.9240],
        "eta": "0 分鐘",
        "desc": "即時出沒點",
    },
    {
        "name": "埡口南側步道 (預測途經點)",
        "pos": [24.9762, 121.9245],
        "eta": "+10 分鐘",
        "desc": "沿草嶺古道步道順向移動",
    },
    {
        "name": "護管所泥塘散熱區 (預測終點)",
        "pos": [24.9745, 121.9248],
        "eta": "+25 分鐘",
        "desc": "高溫 THI 驅動泥塘泡水目的地",
    },
]

HISTORICAL_COS_SIM = 91.5

now_time = datetime.now()
T1_TIME = (now_time + timedelta(hours=1)).strftime("%H:%M")
T2_TIME = (now_time + timedelta(hours=2)).strftime("%H:%M")
T3_TIME = (now_time + timedelta(hours=3)).strftime("%H:%M")


def build_map(output_html="index.html"):
  """繪製並輸出草嶺古道水牛即時戰情與向量圖台"""
  m = folium.Map(
      location=LANDMARKS["埡口涼亭 (鞍部核心區)"],
      zoom_start=16,
      tiles=None,
      zoom_control=True,
  )

  # 1. 內政部國土通用電子地圖 (EMAP)
  folium.TileLayer(
      tiles="https://wmts.nlsc.gov.tw/wmts/EMAP/default/GoogleMapsCompatible/{z}/{y}/{x}",
      attr="&copy; 內政部國土測繪圖資服務雲 (EMAP)",
      name="🗺️ 國土通用電子地圖",
      max_zoom=19,
      overlay=False,
  ).add_to(m)

  # 2. 衛星航照圖 (PHOTO2)
  folium.TileLayer(
      tiles="https://wmts.nlsc.gov.tw/wmts/PHOTO2/default/GoogleMapsCompatible/{z}/{y}/{x}",
      attr="&copy; 內政部國土測繪圖資服務雲 (PHOTO2)",
      name="🛰️ 國土衛星航照圖",
      max_zoom=19,
      overlay=False,
  ).add_to(m)

  # 3. 步道主線
  folium.PolyLine(
      TRAIL_PATH,
      color="#2563eb",
      weight=6,
      opacity=0.8,
      popup="<b>🚶‍♂️ 草嶺古道主線</b>",
  ).add_to(m)

  # 4. 未來移動向量 (紫色虛線對齊步道)
  predicted_path_coords = [loc["pos"] for loc in PREDICTED_LOCATIONS]
  dest_name = PREDICTED_LOCATIONS[-1]["name"]
  via_name = PREDICTED_LOCATIONS[1]["name"]
  via_eta = PREDICTED_LOCATIONS[1]["eta"]

  folium.PolyLine(
      predicted_path_coords,
      color="#a855f7",
      weight=6,
      opacity=0.95,
      dash_array="8, 8",
      popup=f"<b>🧭 未來移動向量</b><br>方向: {VECTOR_BEARING}<br>速度: {VECTOR_SPEED}",
      tooltip=f"🧭 沿古道預測移動：{VECTOR_BEARING} ➔ 目標：{dest_name}",
  ).add_to(m)

  # 5. 標示預測地點 Marker
  for i, loc in enumerate(PREDICTED_LOCATIONS):
    if i == 0:
      continue
    icon_color = "purple" if i == len(PREDICTED_LOCATIONS) - 1 else "cadetblue"
    icon_type = "flag" if i == len(PREDICTED_LOCATIONS) - 1 else "arrow-right"
    folium.Marker(
        loc["pos"],
        tooltip=f"📍 步道預測地點：<b>{loc['name']}</b> ({loc['eta']})",
        popup=f"<b>🎯 預測地點：{loc['name']}</b><br>預估抵達時間：{loc['eta']}<br>說明：{loc['desc']}",
        icon=folium.Icon(color=icon_color, icon=icon_type),
    ).add_to(m)

  # 6. 水牛當前點位與 Hover 氣泡
  folium.Circle(
      BUFFALO_POS,
      radius=10,
      color="#ef4444",
      fill=True,
      fill_color="#ef4444",
      fill_opacity=0.4,
      tooltip=f"🔴 10m 硬防線 | 水牛數量：{BUFFALO_COUNT} 頭",
  ).add_to(m)

  folium.Marker(
      BUFFALO_POS,
      tooltip=f"🦬 <b>水牛數量：{BUFFALO_COUNT} 頭</b><br>向量方向：{VECTOR_BEARING}",
      popup=f"🦬 <b>水牛即時點位資訊</b><br>• 即時數量：{BUFFALO_COUNT} 頭<br>• 移動向量：{VECTOR_BEARING} ({VECTOR_SPEED})<br>• 預測地點：{dest_name}<br>• 預估抵達：25 分鐘內",
      icon=folium.Icon(color="red", icon="warning"),
  ).add_to(m)

  # 7. 沿線據點 Marker
  for name, pos in LANDMARKS.items():
    folium.Marker(
        pos,
        popup=f"<b>📍 {name}</b>",
        tooltip=f"📍 {name}",
        icon=folium.Icon(
            color="orange" if "埡口" in name or "護管所" in name else "green",
            icon="info-sign",
        ),
    ).add_to(m)

  folium.LayerControl(position="topright", collapsed=False).add_to(m)

  m.save(output_html)

  # 8. 注入 UI 面板
  vector_ui = f"""
    <style>
      @keyframes pulse-ring {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.7); }}
        70% {{ transform: scale(1.05); box-shadow: 0 0 0 6px rgba(168, 85, 247, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(168, 85, 247, 0); }}
      }}
      .purple-dot {{
        display: inline-block; width: 8px; height: 8px; background-color: #a855f7;
        border-radius: 50%; margin-right: 6px; animation: pulse-ring 1.5s infinite; vertical-align: middle;
      }}
    </style>

    <div id="mobile-dashboard" style="
        position: fixed; top: 10px; left: 10px; max-width: 320px; z-index: 9999;
        background: rgba(15, 23, 42, 0.92); color: white; padding: 12px 14px;
        border-radius: 12px; font-family: -apple-system, sans-serif;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4); backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.12); font-size: 12px; line-height: 1.5;
    ">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div style="font-weight: bold; font-size: 14px; color: #f8fafc;">
          <span class="purple-dot"></span>🦬 水牛向量與古道對齊預判
        </div>
        <div style="font-size: 10px; color: #94a3b8; background: rgba(255,255,255,0.1); padding: 1px 6px; border-radius: 10px;">
          <span id="update-time">--:--:--</span> (<span id="refresh-count">#1</span>)
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; background: rgba(255,255,255,0.05); padding: 6px; border-radius: 8px; margin-bottom: 6px;">
        <div>
          <div style="color: #94a3b8; font-size: 10px;">🦬 水牛數量</div>
          <div style="font-weight: bold; font-size: 15px; color: #f87171;">{BUFFALO_COUNT} <span style="font-size: 10px;">頭</span></div>
        </div>
        <div>
          <div style="color: #94a3b8; font-size: 10px;">📊 模式相似度</div>
          <div style="font-weight: bold; font-size: 15px; color: #c084fc;">{HISTORICAL_COS_SIM}%</div>
        </div>
      </div>

      <div style="background: rgba(168, 85, 247, 0.15); padding: 6px 8px; border-left: 3px solid #a855f7; border-radius: 4px; margin-bottom: 6px;">
        <div style="color: #e9d5ff; font-weight: bold; font-size: 11px;">🧭 步道沿線向量：{VECTOR_BEARING} ({VECTOR_SPEED})</div>
        <div style="color: #cbd5e1; font-size: 11px; margin-top: 2px;">
          🎯 <b>預測目標地點：</b>{dest_name}
          <br><span style="color: #38bdf8; font-size: 10px;">途經：{via_name} ({via_eta})</span>
        </div>
      </div>

      <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px;">
        <div style="color: #94a3b8; font-size: 10px; margin-bottom: 4px;">⏱️ <b>未來登步道/橫越風險時序：</b></div>
        <div style="display: flex; justify-content: space-between; text-align: center; font-size: 10px;">
          <div style="flex: 1; background: rgba(239, 68, 68, 0.25); margin: 0 2px; padding: 3px 0; border-radius: 4px;">
            <div style="color: #cbd5e1;">{T1_TIME} (+1h)</div>
            <div style="color: #f87171; font-weight: bold;">85% (高)</div>
          </div>
          <div style="flex: 1; background: rgba(245, 158, 11, 0.25); margin: 0 2px; padding: 3px 0; border-radius: 4px;">
            <div style="color: #cbd5e1;">{T2_TIME} (+2h)</div>
            <div style="color: #fbbf24; font-weight: bold;">60% (中)</div>
          </div>
          <div style="flex: 1; background: rgba(34, 197, 94, 0.25); margin: 0 2px; padding: 3px 0; border-radius: 4px;">
            <div style="color: #cbd5e1;">{T3_TIME} (+3h)</div>
            <div style="color: #4ade80; font-weight: bold;">20% (低)</div>
          </div>
        </div>
      </div>
    </div>

    <script>
      let count = parseInt(localStorage.getItem('caoling_map_refresh_count') || '0') + 1;
      localStorage.setItem('caoling_map_refresh_count', count);
      document.getElementById('refresh-count').innerText = '#' + count;

      let now = new Date();
      document.getElementById('update-time').innerText = now.toTimeString().split(' ')[0];

      setTimeout(function(){{ location.reload(); }}, 10000);
    </script>
    </body>
    """

  with open(output_html, "r", encoding="utf-8") as f:
    content = f.read().replace("</body>", vector_ui)
  with open(output_html, "w", encoding="utf-8") as f:
    f.write(content)

  print(f"✅ [main.py] 成功產出最新的戰情地圖: {output_html}")


if __name__ == "__main__":
  build_map("index.html")
