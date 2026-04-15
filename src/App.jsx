import React, { useEffect, useMemo, useRef, useState } from "react";
import { initializeApp } from "firebase/app";
import {
  getDatabase,
  onValue,
  ref,
  set,
  push,
  get,
  remove,
} from "firebase/database";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import * as tf from "@tensorflow/tfjs";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

/* =========================
   FIREBASE CONFIG
========================= */
const firebaseConfig = {
  apiKey: "AIzaSyB9ererNsNonAzH0zQo_GS79XPOyCoMxr4",
  authDomain: "waterdtection.firebaseapp.com",
  databaseURL: "https://waterdtection-default-rtdb.firebaseio.com",
  projectId: "waterdtection",
  storageBucket: "waterdtection.firebasestorage.app",
  messagingSenderId: "690886375729",
  appId: "1:690886375729:web:172c3a47dda6585e4e1810",
  measurementId: "G-TXF33Y6XY0",
};

const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

/* =========================
   CONSTANTS
========================= */
const ROOT_PATH = "Smart_Irrigation";
const HISTORY_PATH = `${ROOT_PATH}/History`;
const CONTROL_PATH = `${ROOT_PATH}/Pump`;
const CONNECTED_PATH = ".info/connected";

const MOISTURE_ON_THRESHOLD = 40;
const MOISTURE_OFF_THRESHOLD = 60;
const STALE_TIMEOUT_MS = 45000;
const LSTM_SEQUENCE_LENGTH = 6;
const SOIL_LABELS = ["Clay Soil", "Loamy Soil", "Silty Soil", "Sandy Soil"];

/* =========================
   HELPERS
========================= */
function formatTime(ts) {
  if (!ts) return "-";
  return new Date(ts).toLocaleString();
}

function safeNumber(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeTempC(tempValue) {
  const t = safeNumber(tempValue);
  // Some sensors send Fahrenheit while UI expects Celsius.
  return t > 60 ? (t - 32) * (5 / 9) : t;
}

function toFeatureVector(data) {
  const moisture = safeNumber(data?.Moisture) / 100;
  const raw = safeNumber(data?.Soil_Raw_Value) / 3000;
  const hum = safeNumber(data?.Hum) / 100;
  const temp = normalizeTempC(data?.Temp) / 50;
  return [moisture, raw, hum, temp].map((v) => Math.max(0, Math.min(1, v)));
}

function buildSequenceFromHistory(history, liveData, sequenceLength = LSTM_SEQUENCE_LENGTH) {
  const ordered = history.slice(0, sequenceLength - 1).reverse();
  const points = ordered.map((item) => ({
    Moisture: item.Moisture,
    Soil_Raw_Value: item.Soil_Raw_Value,
    Hum: item.Hum,
    Temp: item.Temp,
  }));

  points.push(liveData);
  while (points.length < sequenceLength) {
    points.unshift(liveData);
  }

  return points.slice(-sequenceLength).map((item) => toFeatureVector(item));
}

function syntheticCenterForSoil(index) {
  // [moisture, raw, humidity, tempC] normalized later
  const centers = [
    [74, 2350, 52, 24], // Clay
    [54, 1600, 45, 28], // Loamy
    [44, 1150, 40, 27], // Silty
    [30, 700, 28, 34],  // Sandy
  ];
  return centers[index] || centers[1];
}

function createSyntheticSoilDataset(totalSamples = 1400, sequenceLength = LSTM_SEQUENCE_LENGTH) {
  const xs = [];
  const ys = [];

  for (let i = 0; i < totalSamples; i += 1) {
    const classIdx = i % 4;
    const [mCenter, rawCenter, hCenter, tCenter] = syntheticCenterForSoil(classIdx);

    const sequence = [];
    for (let step = 0; step < sequenceLength; step += 1) {
      const drift = (step - (sequenceLength - 1) / 2) * 0.8;
      const moisture = mCenter + drift + (Math.random() * 10 - 5);
      const raw = rawCenter + drift * 18 + (Math.random() * 220 - 110);
      const hum = hCenter + (Math.random() * 12 - 6);
      const tempC = tCenter + (Math.random() * 5 - 2.5);
      sequence.push(toFeatureVector({ Moisture: moisture, Soil_Raw_Value: raw, Hum: hum, Temp: tempC }));
    }

    xs.push(sequence);
    const oneHot = [0, 0, 0, 0];
    oneHot[classIdx] = 1;
    ys.push(oneHot);
  }

  return {
    xs: tf.tensor3d(xs, [xs.length, sequenceLength, 4]),
    ys: tf.tensor2d(ys, [ys.length, 4]),
  };
}

function predictWithLstmModel(model, sequence, liveData) {
  if (!model || !sequence?.length) return null;

  const probs = tf.tidy(() => {
    const input = tf.tensor3d([sequence], [1, sequence.length, 4]);
    const output = model.predict(input);
    return Array.from(output.dataSync());
  });

  if (!probs.length) return null;

  let topIdx = 0;
  for (let i = 1; i < probs.length; i += 1) {
    if (probs[i] > probs[topIdx]) topIdx = i;
  }

  const confidence = Math.round(Math.max(70, Math.min(99, probs[topIdx] * 100)));
  const reasonMap = {
    "Clay Soil": "Sequence pattern indicates high retention over time",
    "Loamy Soil": "Sequence pattern indicates balanced retention and drainage",
    "Silty Soil": "Sequence pattern indicates medium retention with smooth transitions",
    "Sandy Soil": "Sequence pattern indicates faster drainage and drier trend",
  };

  const name = SOIL_LABELS[topIdx] || "Loamy Soil";
  return {
    name,
    confidence,
    reason: `${reasonMap[name]} (browser LSTM inference)`,
    lstmLabel: "TensorFlow.js LSTM (frontend)",
    suggestions: getSoilSuggestion(name, liveData),
  };
}

function getSoilSuggestion(soilName, data) {
  const moisture = safeNumber(data?.Moisture);
  const water = safeNumber(data?.Water);
  const temp = normalizeTempC(data?.Temp);
  const hum = safeNumber(data?.Hum);

  const suggestions = {
    "Clay Soil": [
      "High water retention soil detected.",
      "Use slow and controlled irrigation.",
      "Avoid overwatering because clay soil holds water for longer time.",
      moisture > 70
        ? "Current moisture is high. Keep pump OFF."
        : "Check moisture frequently before starting pump.",
      water < 20
        ? "Water tank is low. Refill water source."
        : "Water level is sufficient for controlled irrigation.",
    ],
    "Loamy Soil": [
      "Balanced loamy soil detected.",
      "This soil is suitable for most crops.",
      "Maintain moderate irrigation cycles.",
      moisture < 40
        ? "Moisture is low. Turn pump ON for short duration."
        : "Moisture is acceptable. Monitor before next irrigation.",
      temp > 32
        ? "Temperature is high. Check irrigation more often."
        : "Temperature is normal for soil balance.",
    ],
    "Silty Soil": [
      "Silty soil detected with moderate moisture retention.",
      "Use medium irrigation schedule.",
      "Avoid sudden heavy watering.",
      hum < 30
        ? "Humidity is low. Soil may dry faster."
        : "Humidity is stable.",
      moisture < 40
        ? "Moisture is low. Pump can be turned ON."
        : "Moisture is acceptable. Pump can remain OFF.",
    ],
    "Sandy Soil": [
      "Sandy soil detected with low water retention.",
      "This soil needs more frequent but shorter irrigation.",
      "Do not wait too long between watering cycles.",
      moisture < 40
        ? "Low moisture detected. Pump should turn ON."
        : "Moisture is acceptable right now.",
      temp > 32
        ? "High temperature may dry sandy soil quickly."
        : "Temperature is not too high.",
    ],
  };

  return suggestions[soilName] || ["No suggestion available."];
}

/*
  This is rule-based prediction.
  It behaves like a lightweight local predictor inside React.
  Later you can replace this with real LSTM backend output.
*/
function predictSoilType(data) {
  const moisture = safeNumber(data?.Moisture);
  const raw = safeNumber(data?.Soil_Raw_Value);
  const hum = safeNumber(data?.Hum);
  const tempC = normalizeTempC(data?.Temp);
  const soilStatus = String(data?.Soil_Status || "").toUpperCase();

  const scores = {
    "Clay Soil": 0,
    "Loamy Soil": 0,
    "Silty Soil": 0,
    "Sandy Soil": 0,
  };

  // Raw value impact
  if (raw >= 1800) scores["Clay Soil"] += 2.2;
  else if (raw >= 1500) scores["Clay Soil"] += 1.2;
  if (raw >= 1000 && raw <= 1900) scores["Loamy Soil"] += 2.2;
  if (raw >= 700 && raw <= 1500) scores["Silty Soil"] += 2.0;
  if (raw < 1100) scores["Sandy Soil"] += 2.4;

  // Moisture impact
  if (moisture >= 65) scores["Clay Soil"] += 2.0;
  else if (moisture >= 50) scores["Clay Soil"] += 1.0;
  if (moisture >= 35 && moisture <= 65) scores["Loamy Soil"] += 2.0;
  if (moisture >= 30 && moisture <= 60) scores["Silty Soil"] += 1.7;
  if (moisture < 40) scores["Sandy Soil"] += 2.0;

  // Environment impact
  if (hum >= 25 && hum <= 70) scores["Loamy Soil"] += 1.0;
  if (hum >= 20 && hum <= 55) scores["Silty Soil"] += 0.9;
  if (hum < 35) scores["Sandy Soil"] += 1.0;
  if (tempC <= 32) scores["Clay Soil"] += 0.6;
  if (tempC >= 18 && tempC <= 35) scores["Loamy Soil"] += 0.8;
  if (tempC > 30) scores["Sandy Soil"] += 0.9;

  // Soil status contributes lightly; it should not dominate classification.
  if (soilStatus === "WET") {
    scores["Clay Soil"] += 0.6;
    scores["Loamy Soil"] += 0.3;
  } else if (soilStatus === "DRY") {
    scores["Sandy Soil"] += 0.6;
    scores["Silty Soil"] += 0.2;
  }

  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [winnerName, winnerScore] = sorted[0];
  const runnerScore = sorted[1]?.[1] ?? 0;
  const margin = Math.max(0, winnerScore - runnerScore);

  const confidence = Math.max(72, Math.min(96, Math.round(72 + winnerScore * 4 + margin * 6)));

  const reasonMap = {
    "Clay Soil": "Higher retention profile from moisture/raw sensor pattern",
    "Loamy Soil": "Balanced mid-range moisture and raw-value pattern",
    "Silty Soil": "Moderate moisture with smooth, fine-particle profile",
    "Sandy Soil": "Dryer profile with faster-drainage sensor behavior",
  };

  const result = {
    name: winnerName,
    confidence,
    reason: `${reasonMap[winnerName]} (score ${winnerScore.toFixed(1)})`,
    lstmLabel: "Rule-based fallback",
  };

  return {
    ...result,
    suggestions: getSoilSuggestion(result.name, data),
  };
}

function getIrrigationAdvice(data, soilPrediction) {
  const moisture = safeNumber(data?.Moisture);
  const soilStatus = String(data?.Soil_Status || "").toUpperCase();
  const water = safeNumber(data?.Water);

  if (water < 20) {
    return "Water level is low. Check tank or water source.";
  }

  if (soilStatus === "WET" || moisture >= MOISTURE_OFF_THRESHOLD) {
    return `Predicted ${soilPrediction.name}. Soil already has enough moisture. Pump should remain OFF.`;
  }

  if (moisture < MOISTURE_ON_THRESHOLD) {
    return `Predicted ${soilPrediction.name}. Low moisture detected. Pump should turn ON.`;
  }

  return `Predicted ${soilPrediction.name}. Soil is in moderate condition. Continue monitoring.`;
}

export default function App() {
  const [liveData, setLiveData] = useState({
    Hum: 0,
    Moisture: 0,
    Pump: false,
    Soil_Raw_Value: 0,
    Soil_Status: "UNKNOWN",
    Temp: 0,
    Test: 0,
    Water: 0,
  });

  const [history, setHistory] = useState([]);
  const [popup, setPopup] = useState({ show: false, type: "info", message: "" });
  const [isLoading, setIsLoading] = useState(true);
  const [streamActive, setStreamActive] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoMode, setAutoMode] = useState(true);
  const [dbConnected, setDbConnected] = useState(true);
  const [hasFirstSnapshot, setHasFirstSnapshot] = useState(false);
  const [lstmReady, setLstmReady] = useState(false);
  const [lstmError, setLstmError] = useState("");

  const lastSnapshotRef = useRef("");
  const lastAlertRef = useRef("");
  const popupTimerRef = useRef(null);
  const lstmModelRef = useRef(null);

  const soilPrediction = useMemo(() => {
    const fallback = predictSoilType(liveData);

    if (!lstmReady || !lstmModelRef.current) {
      return {
        ...fallback,
        lstmLabel: lstmError ? "Rule-based fallback (LSTM unavailable)" : "Initializing TensorFlow.js LSTM...",
      };
    }

    const sequence = buildSequenceFromHistory(history, liveData, LSTM_SEQUENCE_LENGTH);
    const predicted = predictWithLstmModel(lstmModelRef.current, sequence, liveData);
    return predicted || fallback;
  }, [history, liveData, lstmError, lstmReady]);
  const irrigationAdvice = useMemo(
    () => getIrrigationAdvice(liveData, soilPrediction),
    [liveData, soilPrediction]
  );

  const showPopup = (message, type = "info") => {
    if (lastAlertRef.current === `${type}:${message}`) return;
    lastAlertRef.current = `${type}:${message}`;

    setPopup({ show: true, type, message });

    if (popupTimerRef.current) clearTimeout(popupTimerRef.current);
    popupTimerRef.current = setTimeout(() => {
      setPopup((prev) => ({ ...prev, show: false }));
      lastAlertRef.current = "";
    }, 3500);
  };

  useEffect(() => {
    const connectedRef = ref(db, CONNECTED_PATH);

    const unsubscribe = onValue(connectedRef, (snapshot) => {
      const connected = !!snapshot.val();
      setDbConnected(connected);
      if (!connected) {
        setStreamActive(false);
      }
    });

    return () => unsubscribe();
  }, []);

  useEffect(() => {
    let cancelled = false;

    const setupBrowserLstm = async () => {
      try {
        await tf.ready();

        const model = tf.sequential({
          layers: [
            tf.layers.lstm({ units: 18, inputShape: [LSTM_SEQUENCE_LENGTH, 4] }),
            tf.layers.dense({ units: 14, activation: "relu" }),
            tf.layers.dense({ units: 4, activation: "softmax" }),
          ],
        });

        model.compile({
          optimizer: tf.train.adam(0.008),
          loss: "categoricalCrossentropy",
          metrics: ["accuracy"],
        });

        const { xs, ys } = createSyntheticSoilDataset(1400, LSTM_SEQUENCE_LENGTH);
        await model.fit(xs, ys, {
          epochs: 24,
          batchSize: 32,
          shuffle: true,
          verbose: 0,
          validationSplit: 0.12,
        });

        xs.dispose();
        ys.dispose();

        if (cancelled) {
          model.dispose();
          return;
        }

        lstmModelRef.current = model;
        setLstmReady(true);
        setLstmError("");
      } catch (err) {
        console.error("Browser LSTM setup error:", err);
        if (!cancelled) {
          setLstmReady(false);
          setLstmError("LSTM model failed to initialize");
        }
      }
    };

    setupBrowserLstm();

    return () => {
      cancelled = true;
      if (lstmModelRef.current) {
        lstmModelRef.current.dispose();
        lstmModelRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const rootRef = ref(db, ROOT_PATH);

    const unsubscribe = onValue(
      rootRef,
      async (snapshot) => {
        const data = snapshot.val() || {};

        const current = {
          Hum: safeNumber(data.Hum),
          Moisture: safeNumber(data.Moisture),
          Pump: !!data.Pump,
          Soil_Raw_Value: safeNumber(data.Soil_Raw_Value),
          Soil_Status: data.Soil_Status || "UNKNOWN",
          Temp: safeNumber(data.Temp),
          Test: safeNumber(data.Test),
          Water: safeNumber(data.Water),
        };

        setLiveData(current);
        setLastUpdated(Date.now());
        setStreamActive(true);
        setHasFirstSnapshot(true);
        setIsLoading(false);

        const signature = JSON.stringify(current);
        if (signature !== lastSnapshotRef.current) {
          lastSnapshotRef.current = signature;

          const predicted = predictSoilType(current);

          const entry = {
            ...current,
            predicted_soil: predicted.name,
            predicted_confidence: predicted.confidence,
            predicted_reason: predicted.reason,
            timestamp: Date.now(),
          };

          try {
            const historyRef = ref(db, HISTORY_PATH);
            await push(historyRef, entry);

            const snap = await get(historyRef);
            const histObj = snap.val() || {};
            const items = Object.entries(histObj)
              .map(([key, value]) => ({ key, ...value }))
              .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

            if (items.length > 10) {
              const extra = items.slice(0, items.length - 10);
              await Promise.all(
                extra.map((item) =>
                  remove(ref(db, `${HISTORY_PATH}/${item.key}`))
                )
              );
            }
          } catch (err) {
            console.error("History save error:", err);
          }
        }
      },
      (error) => {
        console.error("Firebase read error:", error);
        setIsLoading(false);
        showPopup("Firebase connection error", "danger");
      }
    );

    return () => unsubscribe();
  }, []);

  useEffect(() => {
    const historyRef = ref(db, HISTORY_PATH);

    const unsubscribe = onValue(historyRef, (snapshot) => {
      const data = snapshot.val() || {};
      const items = Object.entries(data)
        .map(([key, value]) => ({ key, ...value }))
        .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
        .slice(0, 10);

      setHistory(items);
    });

    return () => unsubscribe();
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      if (!hasFirstSnapshot || !dbConnected || !lastUpdated) return;
      const diff = Date.now() - lastUpdated;
      if (diff > STALE_TIMEOUT_MS) {
        setStreamActive(false);
      }
    }, 2000);

    return () => clearInterval(timer);
  }, [dbConnected, hasFirstSnapshot, lastUpdated]);

  useEffect(() => {
    if (!autoMode) return;

    const moisture = safeNumber(liveData.Moisture);
    const soilStatus = String(liveData.Soil_Status || "").toUpperCase();

    const updatePumpAuto = async () => {
      try {
        if (soilStatus === "WET" || moisture >= MOISTURE_OFF_THRESHOLD) {
          if (liveData.Pump) {
            await set(ref(db, CONTROL_PATH), false);
            showPopup("Soil is WET / enough moisture reached. Pump turned OFF.", "success");
          }
        } else if (moisture < MOISTURE_ON_THRESHOLD) {
          if (!liveData.Pump) {
            await set(ref(db, CONTROL_PATH), true);
            showPopup("Low moisture detected. Pump turned ON automatically.", "warning");
          }
        }
      } catch (err) {
        console.error("Pump auto control error:", err);
      }
    };

    updatePumpAuto();
  }, [liveData, autoMode]);

  useEffect(() => {
    const moisture = safeNumber(liveData.Moisture);
    const water = safeNumber(liveData.Water);
    const soilStatus = String(liveData.Soil_Status || "").toUpperCase();

    if (isLoading || !hasFirstSnapshot) return;

    if (!dbConnected) {
      showPopup("Internet/Firebase connection lost. Reconnecting...", "danger");
      return;
    }

    if (!streamActive) {
      showPopup("No recent sensor update. Waiting for new reading...", "warning");
      return;
    }

    if (soilStatus === "WET") {
      showPopup("Soil status is WET. Pump should remain OFF.", "success");
    } else if (moisture < MOISTURE_ON_THRESHOLD) {
      showPopup("Low soil moisture detected. Irrigation required.", "warning");
    }

    if (water < 20) {
      showPopup("Low water level detected.", "danger");
    }
  }, [dbConnected, hasFirstSnapshot, isLoading, liveData, streamActive]);

  const manualPumpControl = async (status) => {
    try {
      await set(ref(db, CONTROL_PATH), status);
      showPopup(`Pump turned ${status ? "ON" : "OFF"} manually`, "info");
    } catch (err) {
      console.error(err);
      showPopup("Unable to update pump status", "danger");
    }
  };

  const chartLabels = history
    .slice()
    .reverse()
    .map((item) =>
      new Date(item.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    );

  // Group 1 — Soil: Moisture + Water Level
  const soilChartData = {
    labels: chartLabels,
    datasets: [
      {
        label: "Moisture %",
        data: history.slice().reverse().map((item) => safeNumber(item.Moisture)),
        tension: 0.4,
        fill: true,
        borderColor: "#0ea5e9",
        backgroundColor: "rgba(14,165,233,0.12)",
        pointBackgroundColor: "#0ea5e9",
        pointRadius: 4,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
      {
        label: "Water Level %",
        data: history.slice().reverse().map((item) => safeNumber(item.Water)),
        tension: 0.4,
        fill: true,
        borderColor: "#22d3ee",
        backgroundColor: "rgba(34,211,238,0.08)",
        pointBackgroundColor: "#22d3ee",
        pointRadius: 4,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
    ],
  };

  // Group 2 — Environment: Temperature + Humidity
  const envChartData = {
    labels: chartLabels,
    datasets: [
      {
        label: "Temperature °C",
        data: history.slice().reverse().map((item) => safeNumber(item.Temp)),
        tension: 0.4,
        fill: true,
        borderColor: "#f97316",
        backgroundColor: "rgba(249,115,22,0.10)",
        pointBackgroundColor: "#f97316",
        pointRadius: 4,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
      {
        label: "Humidity %",
        data: history.slice().reverse().map((item) => safeNumber(item.Hum)),
        tension: 0.4,
        fill: true,
        borderColor: "#a78bfa",
        backgroundColor: "rgba(167,139,250,0.09)",
        pointBackgroundColor: "#a78bfa",
        pointRadius: 4,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
    ],
  };

  const makeChartOptions = (titleText) => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        position: "top",
        labels: { color: "#94a3b8", font: { size: 12, family: "Inter" }, boxWidth: 12, padding: 16 },
      },
      title: {
        display: true,
        text: titleText,
        color: "#64748b",
        font: { size: 12, weight: "600", family: "Inter" },
        padding: { bottom: 12 },
      },
    },
    scales: {
      x: {
        ticks: { color: "#475569", font: { size: 10 }, maxRotation: 35 },
        grid: { color: "rgba(255,255,255,0.04)" },
      },
      y: {
        beginAtZero: true,
        ticks: { color: "#475569", font: { size: 11 } },
        grid: { color: "rgba(255,255,255,0.05)" },
      },
    },
  });

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          font-family: 'Inter', system-ui, sans-serif;
          background: #0b1120;
          color: #e2e8f0;
          min-height: 100vh;
        }
        #root {
          width: 100%;
          max-width: 100%;
          border: none;
          min-height: 100vh;
          text-align: left;
        }
        .app { min-height: 100vh; background: #0b1120; padding-bottom: 48px; }

        /* ─── Header ─── */
        .hdr {
          background: linear-gradient(120deg, #0d2240 0%, #0b3356 55%, #083535 100%);
          border-bottom: 1px solid rgba(56,189,248,0.12);
          padding: 26px 36px 22px;
          position: relative;
          overflow: hidden;
        }
        .hdr::after {
          content: '';
          position: absolute;
          inset: 0;
          background: radial-gradient(ellipse at 75% 50%, rgba(34,211,238,0.07) 0%, transparent 65%);
          pointer-events: none;
        }
        .hdr-inner { position: relative; max-width: 1400px; margin: 0 auto; }
        .hdr-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; flex-wrap: wrap; }
        .hdr-brand { display: flex; align-items: center; gap: 14px; }
        .hdr-ico {
          width: 50px; height: 50px; border-radius: 14px; flex-shrink: 0;
          background: linear-gradient(135deg, #0ea5e9, #10b981);
          display: flex; align-items: center; justify-content: center;
          font-size: 25px; box-shadow: 0 4px 16px rgba(14,165,233,0.35);
        }
        .hdr-title { font-size: 21px; font-weight: 800; color: #f0f9ff; line-height: 1.2; letter-spacing: -0.3px; }
        .hdr-sub { font-size: 12.5px; color: #94a3b8; margin-top: 4px; }
        .live-pill {
          display: flex; align-items: center; gap: 7px;
          font-size: 12.5px; font-weight: 700;
          padding: 7px 15px; border-radius: 999px; flex-shrink: 0;
        }
        .live-pill.on { color: #34d399; background: rgba(52,211,153,0.11); border: 1px solid rgba(52,211,153,0.25); }
        .live-pill.off { color: #f87171; background: rgba(248,113,113,0.11); border: 1px solid rgba(248,113,113,0.25); }
        .live-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .live-dot.on { background: #34d399; animation: blink 1.4s ease-in-out infinite; }
        .live-dot.off { background: #f87171; }
        @keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(.72)} }

        .badge-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
        .hbadge {
          padding: 6px 13px; border-radius: 8px; font-size: 12px; font-weight: 600;
          background: rgba(255,255,255,0.055); border: 1px solid rgba(255,255,255,0.1);
          color: #cbd5e1; display: flex; align-items: center; gap: 6px;
        }
        .hb-lbl { color: #64748b; font-weight: 500; }

        /* ─── Main ─── */
        .main {
          max-width: 1400px; margin: 0 auto;
          padding: 26px 26px 0;
          display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
        }
        .col-full { grid-column: 1 / -1; }

        /* ─── Cards ─── */
        .card {
          background: #131e2f; border: 1px solid rgba(255,255,255,0.065);
          border-radius: 20px; padding: 22px;
          box-shadow: 0 4px 24px rgba(0,0,0,0.28);
        }
        .ch { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
        .cico {
          width: 36px; height: 36px; border-radius: 10px;
          display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;
        }
        .ci-teal  { background: rgba(20,184,166,0.14); }
        .ci-blue  { background: rgba(59,130,246,0.14); }
        .ci-violet{ background: rgba(139,92,246,0.14); }
        .ci-amber { background: rgba(245,158,11,0.14); }
        .ci-em    { background: rgba(16,185,129,0.14); }
        .ctitle { font-size: 15.5px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.2px; }
        .csub   { font-size: 11.5px; color: #64748b; margin-top: 1px; }

        /* ─── Sensor grid ─── */
        .sgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 11px; }
        .sc {
          background: #0f1a2e; border: 1px solid rgba(255,255,255,0.055);
          border-radius: 14px; padding: 15px; position: relative; overflow: hidden;
          transition: border-color 0.2s;
        }
        .sc:hover { border-color: rgba(56,189,248,0.22); }
        .sc::before {
          content:''; position:absolute; top:0; left:0; right:0; height:2px; border-radius:14px 14px 0 0;
        }
        .c-teal::before   { background: linear-gradient(90deg,#0ea5e9,#14b8a6); }
        .c-orange::before { background: linear-gradient(90deg,#f97316,#ef4444); }
        .c-blue::before   { background: linear-gradient(90deg,#60a5fa,#3b82f6); }
        .c-cyan::before   { background: linear-gradient(90deg,#22d3ee,#06b6d4); }
        .c-purple::before { background: linear-gradient(90deg,#a78bfa,#8b5cf6); }
        .c-lime::before   { background: linear-gradient(90deg,#a3e635,#84cc16); }
        .c-amber::before  { background: linear-gradient(90deg,#fbbf24,#f59e0b); }
        .c-green::before  { background: linear-gradient(90deg,#34d399,#10b981); }
        .c-red::before    { background: linear-gradient(90deg,#f87171,#dc2626); }
        .s-ico  { font-size: 19px; margin-bottom: 9px; }
        .s-lbl  { font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing:.6px; margin-bottom: 5px; }
        .s-val  { font-size: 25px; font-weight: 800; color: #f1f5f9; line-height: 1; }
        .s-val.on  { color: #34d399; }
        .s-val.off { color: #f87171; }
        .s-val.wet { color: #38bdf8; }
        .s-unit { font-size: 13px; font-weight: 500; color: #64748b; margin-left: 2px; }

        /* ─── Pump ─── */
        .pump-bar {
          display: flex; align-items: center; gap: 11px;
          padding: 14px 16px; border-radius: 13px; margin-bottom: 14px;
        }
        .pump-bar.on { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.2); }
        .pump-bar.off { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.2); }
        .p-dot { width: 13px; height: 13px; border-radius: 50%; flex-shrink: 0; }
        .p-dot.on  { background: #34d399; box-shadow: 0 0 9px rgba(52,211,153,0.6); }
        .p-dot.off { background: #f87171; box-shadow: 0 0 9px rgba(248,113,113,0.4); }
        .p-txt { font-size: 14.5px; font-weight: 700; }
        .p-txt.on  { color: #34d399; }
        .p-txt.off { color: #f87171; }

        .mode-tag {
          display: inline-flex; align-items: center; gap: 5px;
          padding: 4px 11px; border-radius: 6px; font-size: 11.5px; font-weight: 700; margin-bottom: 13px;
        }
        .mode-tag.auto   { background: rgba(59,130,246,0.13); color: #60a5fa; border: 1px solid rgba(59,130,246,0.24); }
        .mode-tag.manual { background: rgba(100,116,139,0.13); color: #94a3b8; border: 1px solid rgba(100,116,139,0.22); }

        .logic-box {
          background: #0f1a2e; border: 1px solid rgba(255,255,255,0.055);
          border-radius: 11px; padding: 13px 14px; margin-bottom: 14px;
          font-size: 13px; color: #94a3b8; line-height: 1.75;
        }
        .logic-box strong { color: #e2e8f0; }

        .btn-g { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
        .btn {
          border: none; padding: 12px 10px; border-radius: 11px; font-weight: 700;
          cursor: pointer; font-size: 13px; font-family: inherit;
          display: flex; align-items: center; justify-content: center; gap: 6px;
          transition: opacity .15s, transform .1s; letter-spacing: .2px;
        }
        .btn:hover  { opacity: .86; transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }
        .btn-toggle {
          width: 100%; border-radius: 14px; padding: 16px 20px;
          font-size: 15px; font-weight: 800; letter-spacing: 0.3px;
          justify-content: flex-start; gap: 14px; position: relative;
          transition: background 0.3s, transform 0.1s, box-shadow 0.3s;
        }
        .btn-toggle.is-on {
          background: linear-gradient(135deg,#059669,#10b981);
          box-shadow: 0 6px 24px rgba(16,185,129,0.35);
          color: #fff;
        }
        .btn-toggle.is-off {
          background: linear-gradient(135deg,#991b1b,#dc2626);
          box-shadow: 0 6px 24px rgba(220,38,38,0.3);
          color: #fff;
        }
        .toggle-knob {
          width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
          background: rgba(255,255,255,0.35);
          transition: background 0.3s;
        }

        /* ─── Soil Prediction ─── */
        .soil-hero {
          text-align: center; padding: 18px;
          background: linear-gradient(135deg,rgba(20,184,166,.08),rgba(59,130,246,.06));
          border: 1px solid rgba(20,184,166,.14); border-radius: 13px; margin-bottom: 13px;
        }
        .soil-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing:.8px; color: #64748b; margin-bottom: 5px; }
        .soil-name { font-size: 27px; font-weight: 800; color: #2dd4bf; letter-spacing: -.5px; }

        .conf-wrap { margin-bottom: 13px; }
        .conf-top { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; color: #64748b; margin-bottom: 6px; }
        .conf-top .cv { color: #a78bfa; font-weight: 700; }
        .conf-bg { height: 7px; background: #0f1a2e; border-radius: 999px; overflow: hidden; }
        .conf-fill { height: 100%; background: linear-gradient(90deg,#8b5cf6,#a78bfa); border-radius: 999px; transition: width .6s ease; }

        .irow {
          background: #0f1a2e; border: 1px solid rgba(255,255,255,0.055);
          border-radius: 10px; padding: 11px 13px; margin-bottom: 9px; font-size: 13px; color: #94a3b8; line-height: 1.6;
        }
        .irow:last-child { margin-bottom: 0; }
        .irow strong { color: #cbd5e1; }
        .hlg { color: #34d399; font-weight: 600; }
        .hlb { color: #60a5fa; font-weight: 600; }

        /* ─── Soil type badges ─── */
        .soil-type-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }
        .stc {
          text-align: center; padding: 14px 10px; border-radius: 12px;
          border: 1.5px solid rgba(255,255,255,0.07);
          background: #0f1a2e; font-size: 13px; font-weight: 600; color: #64748b;
          transition: all .2s;
        }
        .stc.active {
          border-color: #14b8a6; background: rgba(20,184,166,0.1); color: #2dd4bf;
        }
        .stc-dot { font-size: 22px; margin-bottom: 8px; }

        /* ─── Suggestions ─── */
        .sugg-list { display: grid; gap: 8px; }
        .sugg-item {
          display: flex; gap: 10px; align-items: flex-start;
          background: #0f1a2e; border: 1px solid rgba(255,255,255,0.055);
          border-radius: 10px; padding: 11px 13px; font-size: 13px; color: #94a3b8; line-height: 1.55;
        }
        .sugg-num {
          width: 22px; height: 22px; border-radius: 6px; flex-shrink: 0; font-size: 12px; font-weight: 800;
          background: linear-gradient(135deg,#0ea5e9,#6366f1); color: #fff;
          display: flex; align-items: center; justify-content: center;
        }

        /* ─── Chart ─── */
        .chart-wrap { height: 300px; position: relative; }
        .chart-empty {
          height: 100%; display: flex; align-items: center; justify-content: center;
          font-size: 14px; color: #475569; font-weight: 500;
        }

        /* ─── Table ─── */
        .tbl-wrap { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; }
        thead th {
          background: #0f1a2e; color: #64748b; font-size: 11px; font-weight: 700;
          text-transform: uppercase; letter-spacing: .6px; padding: 11px 13px;
          text-align: left; border-bottom: 1px solid rgba(255,255,255,0.055); white-space: nowrap;
        }
        tbody tr { border-bottom: 1px solid rgba(255,255,255,0.035); transition: background .15s; }
        tbody tr:hover { background: rgba(255,255,255,0.025); }
        tbody tr:last-child { border-bottom: none; }
        tbody td { padding: 10px 13px; font-size: 13px; color: #94a3b8; white-space: nowrap; }
        .td-on  { color: #34d399; font-weight: 700; }
        .td-off { color: #f87171; font-weight: 700; }
        .td-wet { color: #38bdf8; font-weight: 700; }
        .td-dry { color: #fbbf24; font-weight: 600; }
        .td-ts  { color: #cbd5e1; font-weight: 600; }

        /* ─── How it works ─── */
        .how-g { display: grid; grid-template-columns: repeat(4,1fr); gap: 11px; }
        .hw-item {
          background: #0f1a2e; border: 1px solid rgba(255,255,255,0.055);
          border-radius: 12px; padding: 15px; font-size: 13px; color: #94a3b8; line-height: 1.6;
        }
        .hw-num {
          width: 27px; height: 27px; border-radius: 7px; font-size: 12.5px; font-weight: 800;
          background: linear-gradient(135deg,#0ea5e9,#6366f1); color: #fff;
          display: flex; align-items: center; justify-content: center; margin-bottom: 10px;
        }
        .hw-item strong { color: #e2e8f0; }

        /* ─── Popup ─── */
        .popup {
          position: fixed; top: 18px; right: 18px; z-index: 9999;
          min-width: 275px; max-width: 370px; padding: 13px 16px;
          border-radius: 13px; color: #fff; font-weight: 600; font-size: 13.5px;
          box-shadow: 0 12px 32px rgba(0,0,0,0.4); display: flex; align-items: flex-start; gap: 9px;
          line-height: 1.5; animation: popIn .25s cubic-bezier(0.34,1.56,0.64,1);
        }
        .popup.info    { background: linear-gradient(135deg,#1d4ed8,#2563eb); border: 1px solid rgba(96,165,250,.3); }
        .popup.success { background: linear-gradient(135deg,#047857,#059669); border: 1px solid rgba(52,211,153,.3); }
        .popup.warning { background: linear-gradient(135deg,#b45309,#d97706); border: 1px solid rgba(251,191,36,.3); }
        .popup.danger  { background: linear-gradient(135deg,#991b1b,#dc2626); border: 1px solid rgba(248,113,113,.3); }
        @keyframes popIn {
          from { transform: translateY(-12px) scale(.96); opacity: 0; }
          to   { transform: translateY(0) scale(1);       opacity: 1; }
        }
        .p-ico { font-size: 17px; flex-shrink: 0; }

        /* ─── Section divider ─── */
        .sdiv {
          grid-column: 1 / -1; font-size: 10.5px; font-weight: 700;
          text-transform: uppercase; letter-spacing: 1px; color: #334155;
          display: flex; align-items: center; gap: 10px; margin-top: 4px;
        }
        .sdiv::after { content:''; flex:1; height:1px; background: rgba(255,255,255,0.045); }

        /* ─── Responsive ─── */

        /* Tablet landscape */
        @media (max-width: 1100px) {
          .how-g { grid-template-columns: repeat(2,1fr); }
          .sgrid { grid-template-columns: repeat(4,1fr); }
        }

        /* Tablet portrait */
        @media (max-width: 860px) {
          .hdr { padding: 16px; }
          .hdr-top { flex-direction: column; gap: 12px; }
          .hdr-title { font-size: 16px; }
          .hdr-sub { font-size: 11.5px; }
          .hdr-ico { width: 42px; height: 42px; font-size: 21px; }
          .live-pill { align-self: flex-start; }
          .badge-row { gap: 8px; margin-top: 14px; }
          .hbadge { font-size: 11px; padding: 5px 10px; }

          .main { grid-template-columns: 1fr; padding: 12px; gap: 14px; }
          .col-full { grid-column: 1; }

          .sgrid { grid-template-columns: repeat(2,1fr); gap: 10px; }

          .soil-type-grid { grid-template-columns: repeat(2,1fr); }
          .how-g { grid-template-columns: repeat(2,1fr); }

          .card { padding: 18px; border-radius: 16px; }
          .chart-wrap { height: 260px; }
        }

        /* Mobile */
        @media (max-width: 520px) {
          .hdr { padding: 14px; }
          .hdr-brand { gap: 10px; }
          .hdr-ico { width: 38px; height: 38px; font-size: 19px; border-radius: 10px; }
          .hdr-title { font-size: 14.5px; }
          .hdr-sub { display: none; }
          .badge-row { gap: 6px; margin-top: 12px; }
          .hbadge { font-size: 10.5px; padding: 4px 9px; border-radius: 6px; }
          .hb-lbl { display: none; }

          .main { padding: 10px; gap: 12px; }
          .card { padding: 14px; border-radius: 14px; }
          .ch { margin-bottom: 14px; gap: 8px; }
          .cico { width: 30px; height: 30px; font-size: 15px; border-radius: 8px; }
          .ctitle { font-size: 14px; }
          .csub { font-size: 11px; }

          .sgrid { grid-template-columns: repeat(2,1fr); gap: 8px; }
          .sc { padding: 12px; border-radius: 12px; }
          .s-ico { font-size: 16px; margin-bottom: 7px; }
          .s-lbl { font-size: 10px; }
          .s-val { font-size: 22px; }
          .s-val[style] { font-size: 18px !important; }

          .pump-bar { padding: 12px 14px; }
          .p-txt { font-size: 13px; }
          .logic-box { font-size: 12px; padding: 11px 12px; }
          .btn-toggle { font-size: 13.5px; padding: 14px 16px; }
          .toggle-knob { width: 18px; height: 18px; }

          .soil-hero { padding: 14px; }
          .soil-name { font-size: 22px; }
          .irow { font-size: 12.5px; padding: 10px 11px; }

          .soil-type-grid { grid-template-columns: repeat(2,1fr); gap: 8px; }
          .stc { padding: 11px 8px; font-size: 12px; border-radius: 10px; }
          .stc-dot { font-size: 19px; margin-bottom: 6px; }

          .sugg-item { font-size: 12.5px; padding: 10px 11px; }
          .sugg-num { width: 20px; height: 20px; font-size: 11px; }

          .chart-wrap { height: 220px; }

          .how-g { grid-template-columns: 1fr; gap: 8px; }
          .hw-item { padding: 12px; font-size: 12.5px; }
          .hw-num { width: 24px; height: 24px; font-size: 11.5px; }

          .popup {
            top: auto; bottom: 14px; right: 14px; left: 14px;
            max-width: 100%; min-width: unset;
            font-size: 13px; padding: 12px 14px;
          }
        }

        /* Very small phones */
        @media (max-width: 360px) {
          .sgrid { grid-template-columns: 1fr 1fr; }
          .soil-type-grid { grid-template-columns: 1fr 1fr; }
          .hdr-title { font-size: 13.5px; }
        }
      `}</style>

      <div className="app">
        {popup.show && (
          <div className={`popup ${popup.type}`}>
            <span className="p-ico">
              {popup.type==="success"?"✅":popup.type==="warning"?"⚠️":popup.type==="danger"?"🚨":"ℹ️"}
            </span>
            {popup.message}
          </div>
        )}

        {/* ── Header ── */}
        <div className="hdr">
          <div className="hdr-inner">
            <div className="hdr-top">
              <div className="hdr-brand">
                <div className="hdr-ico">🌱</div>
                <div>
                  <div className="hdr-title">Precision Agriculture &amp; Smart Irrigation</div>
                  <div className="hdr-sub">Real-time IoT monitoring · Pump control · Soil prediction · History</div>
                </div>
              </div>
              <div className={`live-pill ${dbConnected && streamActive ? "on" : "off"}`}>
                <span className={`live-dot ${dbConnected && streamActive ? "on" : "off"}`}/>
                {dbConnected && streamActive ? "LIVE" : dbConnected ? "NO DATA" : "OFFLINE"}
              </div>
            </div>
            <div className="badge-row">
              <div className="hbadge"><span className="hb-lbl">📡 DB</span>{ROOT_PATH}</div>
              <div className="hbadge"><span className="hb-lbl">🕒 Updated</span>{formatTime(lastUpdated)}</div>
              <div className="hbadge"><span className="hb-lbl">⚙️ Mode</span>{autoMode?"AUTO IRRIGATION":"MANUAL CONTROL"}</div>
              <div className="hbadge"><span className="hb-lbl">💧 Moisture</span>{liveData.Moisture}</div>
            </div>
          </div>
        </div>

        {/* ── Grid ── */}
        <div className="main">

          {/* Live Sensors */}
          <div className="card col-full">
            <div className="ch">
              <div className="cico ci-teal">📊</div>
              <div><div className="ctitle">Live Sensor Values</div><div className="csub">Real-time readings from IoT hardware</div></div>
            </div>
            <div className="sgrid">
              <div className="sc c-teal">
                <div className="s-ico">💧</div>
                <div className="s-lbl">Soil Moisture</div>
                <div className="s-val">{liveData.Moisture}<span className="s-unit">%</span></div>
              </div>
              <div className="sc c-orange">
                <div className="s-ico">🌡️</div>
                <div className="s-lbl">Temperature</div>
                <div className="s-val">{liveData.Temp}<span className="s-unit">°C</span></div>
              </div>
              <div className="sc c-blue">
                <div className="s-ico">💨</div>
                <div className="s-lbl">Humidity</div>
                <div className="s-val">{liveData.Hum}<span className="s-unit">%</span></div>
              </div>
              <div className="sc c-cyan">
                <div className="s-ico">🪣</div>
                <div className="s-lbl">Water Level</div>
                <div className="s-val">{liveData.Water}<span className="s-unit">%</span></div>
              </div>
              <div className="sc c-purple">
                <div className="s-ico">🔬</div>
                <div className="s-lbl">Soil Raw Value</div>
                <div className="s-val" style={{fontSize:"21px"}}>{liveData.Soil_Raw_Value}</div>
              </div>
              <div className="sc c-lime">
                <div className="s-ico">🌍</div>
                <div className="s-lbl">Soil Status</div>
                <div className={`s-val ${String(liveData.Soil_Status).toUpperCase()==="WET"?"wet":""}`} style={{fontSize:"19px"}}>
                  {liveData.Soil_Status}
                </div>
              </div>
              <div className="sc c-amber">
                <div className="s-ico">🧪</div>
                <div className="s-lbl">Test Value</div>
                <div className="s-val">{liveData.Test}</div>
              </div>
              <div className={`sc ${liveData.Pump?"c-green":"c-red"}`}>
                <div className="s-ico">{liveData.Pump?"⚡":"🔴"}</div>
                <div className="s-lbl">Pump Status</div>
                <div className={`s-val ${liveData.Pump?"on":"off"}`}>{liveData.Pump?"ON":"OFF"}</div>
              </div>
            </div>
          </div>

          <div className="sdiv">Control &amp; Intelligence</div>

          {/* Pump Control */}
          <div className="card">
            <div className="ch">
              <div className="cico ci-blue">⚙️</div>
              <div><div className="ctitle">Irrigation Pump Control</div><div className="csub">Manual override &amp; auto threshold logic</div></div>
            </div>

            <div className={`pump-bar ${liveData.Pump?"on":"off"}`}>
              <div className={`p-dot ${liveData.Pump?"on":"off"}`}/>
              <div className={`p-txt ${liveData.Pump?"on":"off"}`}>
                Pump is currently {liveData.Pump?"RUNNING":"STOPPED"}
              </div>
            </div>

            <div className={`mode-tag ${autoMode?"auto":"manual"}`}>
              {autoMode?"🤖 AUTO MODE ACTIVE":"🖐 MANUAL MODE ACTIVE"}
            </div>

            <div className="logic-box">
              <strong>Auto Logic Thresholds</strong><br/>
              Moisture &lt; <strong>{MOISTURE_ON_THRESHOLD}</strong> → Pump <strong style={{color:"#34d399"}}>ON</strong><br/>
              Soil = <strong>WET</strong> or Moisture ≥ <strong>{MOISTURE_OFF_THRESHOLD}</strong> → Pump <strong style={{color:"#f87171"}}>OFF</strong>
            </div>

            <button
              className={`btn btn-toggle ${liveData.Pump ? "is-on" : "is-off"}`}
              onClick={() => manualPumpControl(!liveData.Pump)}
            >
              <span className="toggle-knob"/>
              {liveData.Pump ? "⚡ Pump ON — tap to turn OFF" : "🔴 Pump OFF — tap to turn ON"}
            </button>
          </div>

          {/* Soil Prediction */}
          <div className="card">
            <div className="ch">
              <div className="cico ci-violet">🧠</div>
              <div><div className="ctitle">Soil Prediction</div><div className="csub">ML-style rule-based classification</div></div>
            </div>

            <div className="soil-hero">
              <div className="soil-lbl">Predicted Soil Type</div>
              <div className="soil-name">{streamActive ? soilPrediction.name : "Waiting for data…"}</div>
            </div>

            <div className="conf-wrap">
              <div className="conf-top">
                <span>Model Confidence</span>
                <span className="cv">{streamActive ? `${soilPrediction.confidence}%` : "-"}</span>
              </div>
              <div className="conf-bg">
                <div className="conf-fill" style={{width: streamActive ? `${soilPrediction.confidence}%` : "0%"}}/>
              </div>
            </div>

            <div className="irow">
              <strong>🧬 Engine:</strong> <span className="hlb">{soilPrediction.lstmLabel}</span>
            </div>
            <div className="irow">
              <strong>🔍 Reason:</strong> <span className="hlb">{streamActive ? soilPrediction.reason : "No live data"}</span>
            </div>
            <div className="irow">
              <strong>🚿 Advice:</strong> <span className="hlg">{irrigationAdvice}</span>
            </div>
          </div>

          {/* Soil type selector */}
          <div className="card col-full">
            <div className="ch">
              <div className="cico ci-teal">🌾</div>
              <div><div className="ctitle">Soil Type Classifier</div><div className="csub">Current best-match is highlighted</div></div>
            </div>
            <div className="soil-type-grid">
              {[
                {name:"Clay Soil",   ico:"🟤"},
                {name:"Loamy Soil",  ico:"🟫"},
                {name:"Silty Soil",  ico:"🏜️"},
                {name:"Sandy Soil",  ico:"🟡"},
              ].map(({name,ico}) => (
                <div key={name} className={`stc ${soilPrediction.name===name?"active":""}`}>
                  <div className="stc-dot">{ico}</div>
                  {soilPrediction.name===name ? `✓ ${name}` : name}
                </div>
              ))}
            </div>
          </div>

          {/* Suggestions */}
          <div className="card col-full">
            <div className="ch">
              <div className="cico ci-em">💡</div>
              <div>
                <div className="ctitle">Suggestions for {soilPrediction.name}</div>
                <div className="csub">Tailored recommendations based on current prediction</div>
              </div>
            </div>
            <div className="sugg-list">
              {soilPrediction.suggestions.map((s, i) => (
                <div className="sugg-item" key={i}>
                  <div className="sugg-num">{i+1}</div>
                  <span>{s}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Charts — two grouped */}
          <div className="card">
            <div className="ch">
              <div className="cico ci-teal">💧</div>
              <div>
                <div className="ctitle">Soil &amp; Water Graph</div>
                <div className="csub">Moisture % · Water Level %</div>
              </div>
            </div>
            <div className="chart-wrap">
              {history.length === 0
                ? <div className="chart-empty">⏳ Waiting for sensor data…</div>
                : <Line data={soilChartData} options={makeChartOptions("Soil & Water — Last 10 readings")} />
              }
            </div>
          </div>

          <div className="card">
            <div className="ch">
              <div className="cico ci-amber">🌡️</div>
              <div>
                <div className="ctitle">Environment Graph</div>
                <div className="csub">Temperature °C · Humidity %</div>
              </div>
            </div>
            <div className="chart-wrap">
              {history.length === 0
                ? <div className="chart-empty">⏳ Waiting for sensor data…</div>
                : <Line data={envChartData} options={makeChartOptions("Temperature & Humidity — Last 10 readings")} />
              }
            </div>
          </div>



        
        </div>
      </div>
    </>
  );
}
