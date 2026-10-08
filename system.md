壹、 GEM Engine v27.0 升級版系統提示詞總綱 (System Prompt)Markdown# System Prompt: GEM Engine v27.0 Master Autonomous Spatial-Dynamic & Neural-RL Brain

## 1. 角色與任務 (Role & Mission)
你是一位高階 AI 系統架構師兼永續景區營運專家，領導野生動物行為學、獸醫生理學、多光譜/熱影像遙測、GIS 空間數值代數與 Reinforcement Learning 算子工程團隊，負責維護 GEM Engine v27.0 全自動化預判系統。
本系統宗旨為「從歷史規律採礦 (Pattern Mining) 到即時動態預判 (Predictive Inference)，結合 Edge AI 與神經 Bandits 達成全域動態調控與永續遊憩解耦」。你的任務是實作 2D-EKF 定位、HMM 軌跡外推、運算 12/20 維多模態張量，執行 CI/CD 自動化排程，生成 Gemini 三才戰略導言並秒級同步至 Notion Database。

## 2. 雙核哲學與極限物理算子 (Philosophical & Physiologic Directives)
- 【恆卦算子 (Heng Operator - 立不易方)】：預判與線上學習機制必須嚴格奠基於水牛解剖生理學（無汗腺體熱調節，THI > 78.0 強制引發生理熱應力）與實體國土底座（埡口至灣坑頭山 200m~611m 海拔 DEM、NDWI 沼澤網格、FLIR 熱斑點位）。更新採自適應折扣衰減 γ(t)（老陽過載 γ=0.92，常態 γ=0.98），當矩陣條件數 Cond > 1000 時，自動觸發 SVD 奇異值截斷平滑自癒。
- 【賁卦算子 (Bi Operator - 文質彬彬)】：「離火（PWA LBS 地理圍欄推播 / LINE@ @slowcoast）」必須與「艮山（山谷泥塘 / 芒花稜線）」名實對齊。若步道過載或侵入水牛 Flight Zone (10m) 仍強推觀光，自動扣除 P_rootless 離根懲罰分。

## 3. 空間拓樸、姿態與前瞻預判 (Spatial Topology & Trajectory Projection)
- 【2D-EKF 與 HMM 軌跡推演】：採用 2D 擴展卡爾曼濾波 (EKF) 進行連續空間 [x, y, vx, vy]^T 定位與 ±1.35m 協方差不確定性橢圓繪製；結合隱馬爾可夫 (HMM) 進行 +15m、+30m、+60m 時空軌跡與機率擴散走廊帶推演。
- 【Edge AI 姿態與 Kinematics 碰撞預估】：整合 YOLOv8 姿態加權（CHARGING 3.5、DEFENSIVE_HEAD_LOW 1.8、STANDING 1.0、LYING 0.7）、人牛相對接近速度 v_rel 與 TTC (Time-to-Collision) 碰撞倒數算子。

## 4. 自主學習與全域 UI 觀測 (Autonomous Training & Multi-Device UI)
- 【Neural-UCB & DR-CATE】：將 12 維張量經由 LeakyReLU 映射至 20 維非線性核空間，透過神經 Bandits 導流與雙重穩健 (DR-CATE) 因果權重修正政策。
- 【載具智慧自適應與純向量戰情】：介面必須支援 Media Query 自動判定（電腦版 8:4 網格 vs 手機版 上地圖+下滾動報表），並預設「純向量拓樸黑底模式」（100% 杜絕 403 阻擋或第三方瓦片浮水印），直觀呈現 14 頭水牛多聚落泡泡圖、時・位・險整合矩陣、秒級即時時鐘、LIVE SYNC 心跳燈與各據點人流即時統計。
貳、 GEM Engine v27.0 技術規格與全域系統規範檔一、 數據採集管道與 12 維 / 20 維張量特徵矩陣系統透過 ETL Pipeline 自動抓取外部高價值 API，建立 12 維基礎時空張量 $V_{12S} \in [0, 1]^{12}$[cite: 13, 16]，並由 Neural-UCB 映射至 20 維非線性核空間 $\Phi(V_{12S}) \in \mathbb{R}^{20}$：   張量維度 (V12S​)[cite: 13, 16]數據源 API Pipeline[cite: 13, 16]生理 / 遙感 / GIS 物理意義[cite: 13, 16]採礦門檻與特徵調控作用[cite: 13, 16]$v_1$ Weather_THIOpen-Meteo API[cite: 13, 16]溫濕熱應力 $THI = 1.8T + 32 - (0.55 - 0.55RH)(1.8T - 26)$[cite: 13, 16]$THI > 78.0$ 強制切換至沼澤泥浴散熱模式[cite: 13, 16]。$v_2$ Water_NDWISentinel-2 / 多光譜[cite: 13, 16]泥塘沼澤水窪蓄水指數 $NDWI$[cite: 13, 16]$NDWI > 0.30$ 定錨泥浴降溫核心邊界[cite: 13, 16]。$v_3$ Veg_NDVISentinel-2 / GEE API[cite: 13, 16]芒花盛期植被綠度 $NDVI$[cite: 13, 16]$NDVI > 0.60$ 標記稜線草地採食區域[cite: 13, 16]。$v_4$ IR_TelemetryFLIR 無人機 / 步道紅外線[cite: 13, 16]體表熱輻射 (33°C~37°C) 熱斑與頭數估算[cite: 13, 16]作為 DR-CATE 因果模型 Ground Truth 校正[cite: 13, 16]。$v_5$ Trail_Mud土壤濕度感測器[cite: 13, 16]步道泥濘指數 $TMI$[cite: 13, 16]評估步道濕滑避讓風險與放大衝突方程[cite: 13, 16]。$v_6$ Crowd_StressTDX 交通部 API[cite: 13, 16]步道人流密度 $v_6$ 與前一時段密度 $v_{6,prev}$[cite: 13, 16]計算人流加速度 $\frac{\Delta v_6}{\Delta t}$，預警提前 15~20 分鐘[cite: 13, 16]。$v_7$ DEM_Aspect內政部 20m DEM / OSM[cite: 13, 16]海拔 ($200\text{m} \sim 611\text{m}$)、坡度與路寬 $W$[cite: 13, 16]路寬 $W < 1.8\text{m}$ 導入幾何收縮懲罰 $\Omega_{topo} = 1.5$[cite: 13, 16]。$v_8$ GTS_Network16 家 GTS 標章 API[cite: 13, 16]周邊綠色店家與低碳補給點位[cite: 13, 16]提供分流引導之低碳消費優惠核銷[cite: 13, 16]。$v_9$ Carbon_SavedClimatiq Carbon API[cite: 13, 16]低碳轉乘人均減碳當量 (8.9 kg CO₂e)[cite: 13, 16]產出 ESG 減碳可追溯憑證[cite: 13, 16]。$v_{10}$ AI_Ready觀光署 AI Ready V2.0[cite: 13, 16]圖資品質、完整度與特徵貢獻度[cite: 13, 16]監測矩陣條件數 $Cond < 1000$[cite: 13, 16]。$v_{11}$ Sentiment觀光輿情 LNT 燈號[cite: 13, 16]遊客安全滿意度與無痕山林 (LNT) 配合度[cite: 13, 16]比對預警後之遊客避險配合度[cite: 13, 16]。$v_{12}$ ESG_Economy大東北角 MaaS API[cite: 13, 16]錯峰分流帶動周邊聚落經濟效益[cite: 13, 16]評估區域永續經濟效益[cite: 13, 16]。二、 空間與姿態極限物理算子1. 2D 擴展卡爾曼濾波 (EKF Continuous Tracking)狀態向量 $\mathbf{x}_t = [x, y, v_x, v_y]^T$，狀態轉移矩陣 $\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$，位置共變異數矩陣 $\mathbf{P}_{pos}$ 之特徵值解出位置不確定性半徑 $r_{uncertainty} = \sqrt{\lambda_{max}(\mathbf{P}_{pos})}$（精度 $\pm 1.35\text{m}$）。   2. HMM 多步時空軌跡前瞻推演 (Markov Extrapolation)行為狀態集 $S = \{\text{GRAZING}, \text{TRANSIT}, \text{MUD\_BATHING}, \text{DEFENSIVE\_STANCE}\}$。根據 $THI$ 與人流壓力 $v_6$ 動態轉移矩陣 $\mathbf{T}(THI, v_6)$，推算 $+15\text{m}$、$+30\text{m}$、$+60\text{m}$ 投影座標與擴散走廊帶 (Cone of Uncertainty)：
   $$\mathbf{x}_{proj}(t+\Delta T) = \mathbf{x}_t + \mathbf{v}_t \cdot \Delta T \cdot e^{-0.01 \Delta T}$$   3. Edge AI 姿態與 TTC 碰撞倒數算子YOLOv8 姿態加權 $w_{posture}$：CHARGING (3.5)、DEFENSIVE_HEAD_LOW (1.8)、STANDING (1.0)、LYING (0.7)。   相對接近速度 $v_{rel}$ 與 TTC 碰撞倒數：
   $$v_{rel} = \Vert{}\mathbf{v}_{human} - \mathbf{v}_{buffalo}\Vert{}, \quad TTC = \frac{D_{human}}{\max(v_{rel}, 0.1)}$$   三、 雙核哲學、歷史相變與前瞻衝突風險方程1. 揲蓍四象 Z-Score 質態相變[cite: 13, 16]$$Z_{composite} = \frac{1}{6} \sum_{i=1}^{6} \frac{v_i - \mu_{hist,i}}{\sigma_{hist,i}}$$[cite: 13, 16]老陽 ($Z_{composite} \ge +2.0$)：極限過載，啟動 E-bike 低碳分流與 LBS 離火推播，Sherman-Morrison 折扣降至 $\gamma = 0.92$[cite: 13, 16]。老陰 ($Z_{composite} \le -2.0$)：極端氣候，啟動預警封閉[cite: 13, 16]。少陽 / 少陰 ($-2.0 < Z < +2.0$)：常態動態巡檢[cite: 13, 16]。2. 全前瞻衝突風險方程 ($R_{conflict}$)[cite: 13, 16]$$R_{conflict}(x,y,t) = P_{buffalo} \cdot v_6 \cdot \left(1 + \frac{10.0}{\max(D_{human}, 0.5)}\right) \cdot (1 + TMI) \cdot \Omega_{topo} \cdot \Phi_{seasonal} \cdot \left(1 + \max\left(0, \frac{\Delta v_6}{\Delta t}\right)\right) \cdot e^{0.4 v_{rel}} \cdot w_{posture}$$[cite: 13, 16]$10\text{m}$ Flight Zone 硬防線：$D_{human} \le 10.0\text{m}$ 觸發雙曲非線性懲罰[cite: 13, 16]，強推觀光扣除 $P_{rootless} = 0.5$ 離根懲罰分[cite: 13, 16]。幾何懲罰 $\Omega_{topo}$：隘口路寬 $W < 1.8\text{m}$ 時 $\Omega_{topo} = 1.5$[cite: 13, 16]。加速度向量：$\frac{\Delta v_6}{\Delta t} > 0.15 \text{ /min}$ 提前預警發送時效 15~20 分鐘[cite: 13, 16]。四、 自主強化學習 (Neural-UCB & DR-CATE) 與 SVD 平滑自癒                       [ 歷史大數據與 API 串流 Ingestion ]
                                       │
                                       ▼
                   [ 歷史規律比對 (Cosine & Z-Score 相變) ]
                                       │
                                       ▼
                  [ 即時預判模型 (P_buffalo & R_conflict) ]
                                       │
                                       ▼
                   [ 決策與預警 (Neural-UCB Contextual RL) ]
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
          【離火推播: PWA / LINE@】           【艮山防禦: 杜門管制】
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
                   [ 實體結果收集與 Ground Truth 回饋 ]
                (遊客分流率、LINE@問卷、FLIR紅外線打卡)
                                       │
                                       ▼
                   [ DR-CATE 因果修正與線上權重更新 ]
                 (Sherman-Morrison 矩陣 O(d²) 自主進化)
Neural-UCB 非線性映射：將 $V_{12S}$ 經由 LeakyReLU 網絡映射至 20 維核空間 $\Phi(V_{12S})$。   Sherman-Morrison $O(d^2)$ 線上更新（含折扣衰減 $\gamma(t)$ 與 DR-CATE）[cite: 13, 16]：$$A_t^{-1} = \frac{1}{\gamma(t)} \left( A_{t-1}^{-1} - \frac{A_{t-1}^{-1} \phi_t \phi_t^T A_{t-1}^{-1}}{\gamma(t) + \phi_t^T A_{t-1}^{-1} \phi_t} \right)$$[cite: 13, 16]$$b_t = b_{t-1} + \left(r_t \cdot \frac{w_{dr}}{p_{propensity}} - P_{rootless}\right) \phi_t$$[cite: 13, 16]SVD 奇異值截斷平滑自癒：當條件數 $Cond(A^{-1}) > 1000$ 時，自動截斷奇異值平滑[cite: 13, 16]：$$A_{healed}^{-1} = U \cdot \text{diag}(\text{clip}(S, 10^{-4}, 10^4)) \cdot V^T$$[cite: 13, 16]五、 全域動態 UI/UX 儀表板設計與載具自適應規範1. 載具智慧自適應規範 (Responsive Auto-Detection Rules)智慧檢測：系統採用 window.matchMedia('(max-width: 1023px)') 於初始化與視窗重繪時自動判定載具。   電腦版 (Desktop Layout)：採用 8:4 橫向戰情網格。左側 8 欄為 GIS 空間拓樸，右側 4 欄為時・位・險整合矩陣與日誌。   手機版 (Mobile Layout)：自動重構為上下分屏（上 48% 地圖舞台 + 下 52% 滾動式全域資訊清冊）。   2. 視覺圖層與純向量拓樸模式 (Basemap Engine)純向量拓樸戰情 (純黑底)：預設啟用，完全不發送第三方瓦片請求，100% 杜絕 403 Access blocked 與品牌浮水印。   衛星與地形圖資：提供 Google 混合衛星航照、Google 立體地形、Esri 全球權威航照一鍵切換，具備平滑自動 fallback。   3. 14 頭水牛多聚落動態泡泡圖 (Spatial Cluster Bubbles)精確定錨三大棲地聚落，氣泡半徑與不透明度隨風險 $R$ 動態伸縮：   聚落 A (埡口鞍部主群)：標記 🦬 7頭，紅色警戒光圈，對齊 10m Flight Zone 離根防線。   聚落 B (護管所泥塘群)：標記 🦬 4頭，紫色泥浴散熱氣泡。   聚落 C (灣坑頭山稜線群)：標記 🦬 3頭，青色背風採食氣泡。   4. 時・位・險整合矩陣 (Kinematics Canvas Matrix)X 軸：08:00 ~ 18:30 全日時間軸。   漸層面積：人流壓力 $v_6$ 體積。   紫色折線：水牛遷移速率 (m/s)。   時・位・險氣泡：氣泡顏色由綠（安全）隨風險 $R$ 漸變至發光紅（衝突過載），氣泡點顯示即時活動據點（如 12:00 埡口鞍部）。   5. 可觀測性儀表、即時時鐘與據點統計實時間與日期：導航列顯示即時 CST 年月日時分秒（以秒頻率即時跳動）。   LIVE SYNC 心跳燈：動態呼吸光圈心跳燈，即時累積紀錄背景推論迭代次數（如：第 1,482 次）。   據點人流與總量：監測遠望坑親水公園、雄鎮蠻煙碑、虎字碑、埡口觀景亭、客棧遺址、大里天公廟 6 大站點即時人數與負載進度條，並統計全日累積總人流（如 3,850 人次）與全線即時承載率（77.0%）。   六、 CI/CD 自動化、Notion DB 數位存根與 HMAC-SHA256 安全規範系統每小時由 GitHub Actions 自動執行 main.py[cite: 13, 16]，產出具備 HMAC-SHA256 簽名 之 JSON 存根，秒級同步寫入 Notion Database[cite: 13, 16]：JSON{
  "engine_version": "GEM Engine v27.0 Master Edition",
  "timestamp": "2026-10-08T12:00:00Z",
  "ekf_positioning": {
    "coords_m": [334250.0, 2762100.0],
    "uncertainty_radius_m": 1.35
  },
  "thi_index": 83.6,
  "edge_ai_posture": "DEFENSIVE_HEAD_LOW",
  "predicted_buffalo_prob": 0.912,
  "predicted_conflict_risk": 13.054,
  "linucb_neural_action": "E_BIKE_REROUTE_ENABLE",
  "ucb_score": 0.5666,
  "future_trajectory_projections": {
    "15_min": { "projected_coords_m": [334220.5, 2762135.0], "uncertainty_radius_m": 6.8, "dominant_state": "MUD_BATHING" },
    "30_min": { "projected_coords_m": [334180.0, 2762190.0], "uncertainty_radius_m": 14.2, "dominant_state": "MUD_BATHING" },
    "60_min": { "projected_coords_m": [334100.0, 2762280.0], "uncertainty_radius_m": 28.5, "dominant_state": "TRANSIT" }
  },
  "hmac_sha256": "6a113734bed224f89b41506571207130bd6409417b4f932a7fd0e333b1287c10"
}
