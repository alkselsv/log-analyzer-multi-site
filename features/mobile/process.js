require('../../js/load-env').loadEnv();
const fs = require('fs');
const readline = require('readline');

/** Дождаться сброса буфера Writable (без stream/promises — совместимость со старым Node). */
function waitWritableFinish(stream) {
  return new Promise((resolve, reject) => {
    if (stream.writableFinished) {
      resolve();
      return;
    }
    stream.once('error', reject);
    stream.once('finish', resolve);
  });
}

// Название сайта (через SITE_NAME) добавляется в имена выходного и промежуточного файлов.
// Например: SITE_NAME=vedita node process_tmv_clk_enhanced.js
const siteName = (process.env.SITE_NAME || '').trim().replace(/[^\w.-]/g, '_') || null;

const inputFile = process.env.INPUT_FILE || 'mobile.vedita+zwilling.ndjson';
const intermediateFile = siteName
  ? `intermediate_${siteName}.json`
  : 'intermediate_mobile.vedita+zwilling.json';
const checkpointFile = process.env.TMV_CHECKPOINT_FILE || 'tmv_process_checkpoint.json';
const changedSessionsFile = process.env.CHANGED_SESSIONS_FILE || 'changed_sessions.json';
const forceFullRebuild = process.env.FORCE_FULL_REBUILD === '1';

// Количество последних tmv записей для сбора (добавляется в имя выходного файла)
const MAX_TMV_RECORDS = parseInt(process.env.MAX_TMV_RECORDS, 10) || 5;
const outputFile = process.env.TMV_STATISTICS_JSON
  || process.env.OUTPUT_FILE
  || (siteName
  ? `${siteName}_tmv_clk_statistics_max${MAX_TMV_RECORDS}.json`
  : `mobile_tmv_clk_statistics_max${MAX_TMV_RECORDS}.json`);

let parsed = 0;
let processed = 0;
let sclAggregated = 0;
let isFirst = true;

const mode = process.env.MODE || 'extract';

function getFileIdentity(stats) {
  return {
    ino: stats.ino || null,
    size: stats.size,
    mtimeMs: stats.mtimeMs,
  };
}

function loadJsonFile(path, fallbackValue) {
  if (!fs.existsSync(path)) {
    return fallbackValue;
  }

  try {
    return JSON.parse(fs.readFileSync(path, 'utf8'));
  } catch (error) {
    console.warn(`Предупреждение: не удалось прочитать JSON из '${path}': ${error.message}`);
    return fallbackValue;
  }
}

function writeJsonAtomic(path, value) {
  const tmpPath = `${path}.tmp`;
  fs.writeFileSync(tmpPath, JSON.stringify(value));
  fs.renameSync(tmpPath, path);
}

function writeChangedSessions(path, sessionIds, fullRebuild) {
  writeJsonAtomic(path, {
    full_rebuild: fullRebuild,
    session_ids: Array.from(sessionIds).sort(),
  });
}

function loadCheckpoint() {
  return loadJsonFile(checkpointFile, null);
}

function restoreStateFromCheckpoint(checkpoint) {
  for (const key of Object.keys(tmvHistoryBySession)) {
    delete tmvHistoryBySession[key];
  }
  for (const key of Object.keys(sclHistoryBySession)) {
    delete sclHistoryBySession[key];
  }
  for (const key of Object.keys(lastTmvBySession)) {
    delete lastTmvBySession[key];
  }

  if (!checkpoint) {
    return;
  }

  const tmvState = checkpoint.tmvHistoryBySession || {};
  for (const [sessionKey, history] of Object.entries(tmvState)) {
    tmvHistoryBySession[sessionKey] = Array.isArray(history) ? history : [];
    const lastRecord = tmvHistoryBySession[sessionKey][tmvHistoryBySession[sessionKey].length - 1];
    if (lastRecord) {
      lastTmvBySession[sessionKey] = lastRecord;
    }
  }

  const sclState = checkpoint.sclHistoryBySession || {};
  for (const [sessionKey, history] of Object.entries(sclState)) {
    sclHistoryBySession[sessionKey] = Array.isArray(history) ? history : [];
  }
}

function buildCheckpoint(stats, offset) {
  return {
    source: getFileIdentity(stats),
    offset,
    tmvHistoryBySession,
    sclHistoryBySession,
  };
}

function shouldRebuild(checkpoint, stats) {
  if (forceFullRebuild) {
    return true;
  }

  if (!checkpoint || !checkpoint.source) {
    return true;
  }

  const previous = checkpoint.source;
  if (stats.size < (checkpoint.offset || 0)) {
    return true;
  }

  if (previous.ino !== null && stats.ino !== undefined && previous.ino !== stats.ino) {
    console.log(
      `Checkpoint: inode изменился (${previous.ino} -> ${stats.ino}), продолжаем инкрементально.`
    );
  }

  return false;
}

async function readNewCompleteLines(startOffset) {
  const lines = [];
  let buffer = '';
  let nextOffset = startOffset;
  const stream = fs.createReadStream(inputFile, {
    start: startOffset,
    encoding: 'utf8',
  });

  for await (const chunk of stream) {
    buffer += chunk;

    let newlineIndex = buffer.indexOf('\n');
    while (newlineIndex !== -1) {
      const completeChunk = buffer.slice(0, newlineIndex + 1);
      const line = completeChunk.slice(0, -1).replace(/\r$/, '');
      lines.push(line);
      nextOffset += Buffer.byteLength(completeChunk, 'utf8');
      buffer = buffer.slice(newlineIndex + 1);
      newlineIndex = buffer.indexOf('\n');
    }
  }

  return {
    lines,
    nextOffset,
    hasPartialTail: buffer.length > 0,
  };
}

// Класс для вычисления статистик движения мыши с фокусом на детекцию ботов
class MouseMoveStatistics {
    constructor() {
        this.points = [];
        this.maxPoints = 1000; // Ограничиваем количество точек для анализа
        this.minPointsForAnalysis = 3;
    }

    reset() {
        this.points = [];
    }

    addPoint(x, y, timestamp) {
        this.points.push({ x, y, timestamp });

        if (this.points.length > this.maxPoints) {
            this.points = this.points.slice(-this.maxPoints);
        }
    }

    run(event) {
        const { clientX: x, clientY: y, tm: timestamp } = event;

        if (this.points.length > 0) {
            const lastPoint = this.points[this.points.length - 1];
            const distance = Math.sqrt((x - lastPoint.x) ** 2 + (y - lastPoint.y) ** 2);

            if (distance < 1) return;

            if (timestamp - lastPoint.timestamp > 15000) {
                this.reset();
            }
        }

        this.addPoint(x, y, timestamp);
    }

    calculateDirectionEntropy() {
        if (this.points.length < 2) return 0;

        const directions = [];
        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i-1].x;
            const dy = this.points[i].y - this.points[i-1].y;
            const angle = Math.atan2(dy, dx);
            directions.push(angle);
        }

        const bins = 16;
        const binSize = (2 * Math.PI) / bins;
        const counts = new Array(bins).fill(0);

        directions.forEach(angle => {
            let normalizedAngle = angle + Math.PI;
            const binIndex = Math.floor(normalizedAngle / binSize);
            counts[binIndex % bins]++;
        });

        const total = directions.length;
        let entropy = 0;
        counts.forEach(count => {
            if (count > 0) {
                const p = count / total;
                entropy -= p * Math.log2(p);
            }
        });

        return entropy;
    }

    calculateJerk() {
        if (this.points.length < 4) return 0;

        const jerks = [];
        for (let i = 3; i < this.points.length; i++) {
            const p1 = this.points[i-3];
            const p2 = this.points[i-2];
            const p3 = this.points[i-1];
            const p4 = this.points[i];

            const dt1 = p2.timestamp - p1.timestamp;
            const dt2 = p3.timestamp - p2.timestamp;
            const dt3 = p4.timestamp - p3.timestamp;

            if (dt1 > 0 && dt2 > 0 && dt3 > 0) {
                const v1x = (p2.x - p1.x) / dt1;
                const v1y = (p2.y - p1.y) / dt1;
                const v2x = (p3.x - p2.x) / dt2;
                const v2y = (p3.y - p2.y) / dt2;
                const v3x = (p4.x - p3.x) / dt3;
                const v3y = (p4.y - p3.y) / dt3;

                const a1x = (v2x - v1x) / dt2;
                const a1y = (v2y - v1y) / dt2;
                const a2x = (v3x - v2x) / dt3;
                const a2y = (v3y - v2y) / dt3;

                const jerkx = (a2x - a1x) / dt3;
                const jerky = (a2y - a1y) / dt3;
                const jerk = Math.sqrt(jerkx * jerkx + jerky * jerky);

                jerks.push(jerk);
            }
        }

        return jerks.length > 0 ? jerks.reduce((a, b) => a + b, 0) / jerks.length : 0;
    }

    calculateLinearity() {
        if (this.points.length < 2) return 1;

        const start = this.points[0];
        const end = this.points[this.points.length - 1];
        const straightLineDistance = Math.sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2);

        if (straightLineDistance === 0) return 1;

        let actualDistance = 0;
        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i-1].x;
            const dy = this.points[i].y - this.points[i-1].y;
            actualDistance += Math.sqrt(dx * dx + dy * dy);
        }

        return straightLineDistance / actualDistance;
    }

    calculateJitter() {
        if (this.points.length < 2) return 0;

        let jitterCount = 0;
        let totalMovements = 0;

        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i-1].x;
            const dy = this.points[i].y - this.points[i-1].y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            totalMovements++;
            if (distance < 3) {
                jitterCount++;
            }
        }

        return totalMovements > 0 ? jitterCount / totalMovements : 0;
    }

    calculateSpeedVariability() {
        if (this.points.length < 2) return 0;

        const speeds = [];
        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i-1].x;
            const dy = this.points[i].y - this.points[i-1].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const dt = this.points[i].timestamp - this.points[i-1].timestamp;

            if (dt > 0) {
                speeds.push(distance / dt);
            }
        }

        if (speeds.length < 2) return 0;

        const meanSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length;
        const variance = speeds.reduce((sum, speed) => sum + (speed - meanSpeed) ** 2, 0) / speeds.length;

        return Math.sqrt(variance) / meanSpeed;
    }

    calculateCurvature() {
        if (this.points.length < 3) return 0;

        const curvatures = [];
        for (let i = 1; i < this.points.length - 1; i++) {
            const p1 = this.points[i-1];
            const p2 = this.points[i];
            const p3 = this.points[i+1];

            const dx1 = p2.x - p1.x;
            const dy1 = p2.y - p1.y;
            const dx2 = p3.x - p2.x;
            const dy2 = p3.y - p2.y;

            const crossProduct = dx1 * dy2 - dy1 * dx2;
            const magnitude1 = Math.sqrt(dx1 * dx1 + dy1 * dy1);
            const magnitude2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);

            if (magnitude1 > 0 && magnitude2 > 0) {
                const curvature = Math.abs(crossProduct) / (magnitude1 * magnitude2);
                curvatures.push(curvature);
            }
        }

        return curvatures.length > 0 ? curvatures.reduce((a, b) => a + b, 0) / curvatures.length : 0;
    }

    calculateRepetitionPatterns() {
        if (this.points.length < 6) return 0;

        const segmentLength = 3;
        let repetitions = 0;
        let totalComparisons = 0;

        for (let i = 0; i <= this.points.length - segmentLength * 2; i++) {
            for (let j = i + segmentLength; j <= this.points.length - segmentLength; j++) {
                const segment1 = this.points.slice(i, i + segmentLength);
                const segment2 = this.points.slice(j, j + segmentLength);

                const normalized1 = this.normalizeSegment(segment1);
                const normalized2 = this.normalizeSegment(segment2);

                const similarity = this.calculateSegmentSimilarity(normalized1, normalized2);
                totalComparisons++;

                if (similarity > 0.8) {
                    repetitions++;
                }
            }
        }

        return totalComparisons > 0 ? repetitions / totalComparisons : 0;
    }

    normalizeSegment(segment) {
        const start = segment[0];
        return segment.map(point => ({
            x: point.x - start.x,
            y: point.y - start.y
        }));
    }

    calculateSegmentSimilarity(seg1, seg2) {
        if (seg1.length !== seg2.length) return 0;

        let totalDistance = 0;
        for (let i = 0; i < seg1.length; i++) {
            const dx = seg1[i].x - seg2[i].x;
            const dy = seg1[i].y - seg2[i].y;
            totalDistance += Math.sqrt(dx * dx + dy * dy);
        }

        const avgDistance = totalDistance / seg1.length;
        return Math.max(0, 1 - avgDistance / 50);
    }

    calculateTimeIntervals() {
        if (this.points.length < 2) return { mean: 0, std: 0, irregularity: 0 };

        const intervals = [];
        for (let i = 1; i < this.points.length; i++) {
            intervals.push(this.points[i].timestamp - this.points[i-1].timestamp);
        }

        const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const variance = intervals.reduce((sum, interval) => sum + (interval - mean) ** 2, 0) / intervals.length;
        const std = Math.sqrt(variance);
        const irregularity = std / mean;

        return { mean, std, irregularity };
    }

    calculateTotalDistance() {
        if (this.points.length < 2) return 0;

        let totalDistance = 0;
        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i-1].x;
            const dy = this.points[i].y - this.points[i-1].y;
            totalDistance += Math.sqrt(dx * dx + dy * dy);
        }

        return totalDistance;
    }

    calculateAverageSpeed() {
        const totalDistance = this.calculateTotalDistance();
        const totalDuration = this.points[this.points.length - 1].timestamp - this.points[0].timestamp;

        return totalDuration > 0 ? totalDistance / totalDuration : 0;
    }

    calculateAccelerationVariability() {
        if (this.points.length < 3) return 0;

        const accelerations = [];
        for (let i = 2; i < this.points.length; i++) {
            const p1 = this.points[i-2];
            const p2 = this.points[i-1];
            const p3 = this.points[i];

            const dt1 = p2.timestamp - p1.timestamp;
            const dt2 = p3.timestamp - p2.timestamp;

            if (dt1 > 0 && dt2 > 0) {
                const v1x = (p2.x - p1.x) / dt1;
                const v1y = (p2.y - p1.y) / dt1;
                const v2x = (p3.x - p2.x) / dt2;
                const v2y = (p3.y - p2.y) / dt2;

                const ax = (v2x - v1x) / dt2;
                const ay = (v2y - v1y) / dt2;
                const acceleration = Math.sqrt(ax * ax + ay * ay);

                accelerations.push(acceleration);
            }
        }

        if (accelerations.length < 2) return 0;

        const meanAcc = accelerations.reduce((a, b) => a + b, 0) / accelerations.length;
        const variance = accelerations.reduce((sum, acc) => sum + (acc - meanAcc) ** 2, 0) / accelerations.length;

        return Math.sqrt(variance) / meanAcc;
    }

    calculateBotScore(features) {
        let score = 0;

        if (features.directionEntropy < 2.0) score += 0.3;
        else if (features.directionEntropy < 2.5) score += 0.1;

        if (features.linearity > 0.9) score += 0.2;
        else if (features.linearity > 0.8) score += 0.1;

        if (features.jerk < 0.001) score += 0.2;
        else if (features.jerk < 0.01) score += 0.1;

        if (features.jitter < 0.1) score += 0.15;
        else if (features.jitter < 0.2) score += 0.05;

        if (features.speedVariability < 0.3) score += 0.15;
        else if (features.speedVariability < 0.5) score += 0.05;

        if (features.repetitionPatterns > 0.3) score += 0.2;
        else if (features.repetitionPatterns > 0.1) score += 0.1;

        if (features.timeIntervals.irregularity < 0.3) score += 0.1;

        return Math.min(1.0, score);
    }

    getBotDetectionFeatures() {
        if (this.points.length < this.minPointsForAnalysis) {
            return null;
        }

        const features = {
            pointCount: this.points.length,
            totalDuration: this.points[this.points.length - 1].timestamp - this.points[0].timestamp,
            directionEntropy: this.calculateDirectionEntropy(),
            jerk: this.calculateJerk(),
            linearity: this.calculateLinearity(),
            jitter: this.calculateJitter(),
            speedVariability: this.calculateSpeedVariability(),
            curvature: this.calculateCurvature(),
            repetitionPatterns: this.calculateRepetitionPatterns(),
            timeIntervals: this.calculateTimeIntervals(),
            totalDistance: this.calculateTotalDistance(),
            averageSpeed: this.calculateAverageSpeed(),
            accelerationVariability: this.calculateAccelerationVariability()
        };

        features.botScore = this.calculateBotScore(features);
        return features;
    }

    send() {
        const features = this.getBotDetectionFeatures();
        if (!features) return null;

        return {
            '_b': {
                'n': features.pointCount,
                'tm': this.points[this.points.length - 1].timestamp,
                'ms': features.totalDuration,
                'cxy2': [[this.points[0].x, this.points[0].y], [this.points[this.points.length - 1].x, this.points[this.points.length - 1].y]],
                'l': features.totalDistance
            },
            'bot_detection': {
                'direction_entropy': features.directionEntropy,
                'jerk': features.jerk,
                'linearity': features.linearity,
                'jitter': features.jitter,
                'speed_variability': features.speedVariability,
                'curvature': features.curvature,
                'repetition_patterns': features.repetitionPatterns,
                'time_irregularity': features.timeIntervals.irregularity,
                'acceleration_variability': features.accelerationVariability,
                'bot_score': features.botScore,
                'is_likely_bot': features.botScore > 0.5
            }
        };
    }
}

const lastTmvBySession = {};
const tmvHistoryBySession = {};
const sclHistoryBySession = {};
const lastEventTmBySession = {};

function getSessionKey(sessA) {
  return `${sessA}`;
}

function parseTmvCoordinates(lstr, tm) {
  let absTm = tm;
  return lstr.split(';').map(el => {
    const parts = el.split(',');
    if (!parts.length) return null;
    const t = parseInt(parts[0], 36);
    const x = parts[2] !== undefined ? parseInt(parts[2], 36) : 0;
    const y = parts[3] !== undefined ? parseInt(parts[3], 36) : 0;
    absTm += t;
    return { clientX: x, clientY: y, tm: absTm };
  }).filter(Boolean);
}

function hasTmv(obj) {
  return obj?.prm?.data?.e?.tmv?.l;
}

function hasClk(obj) {
  return obj?.prm?.data?.e?.clk;
}

function hasScl(obj) {
  return obj?.prm?.data?.e?.scl?.l;
}

function isQmlReady(obj) {
  return obj?.act === 'qml.ready';
}

function parseSclCoordinates(lstr, tm) {
  let absTm = tm;
  return lstr.split(';').map(el => {
    const [t, x, y] = el.split(',').map(val => parseInt(val, 36));
    absTm += t;
    return { clientX: x, clientY: y, tm: absTm };
  });
}

function getAllTmvCoordinatesUpToQmlReady(sessionKey, currentTm) {
  const history = tmvHistoryBySession[sessionKey] || [];
  let allCoordinates = [];
  for (const tmvRecord of history) {
    if (tmvRecord.tm < currentTm) {
      allCoordinates = [...allCoordinates, ...tmvRecord.coordinates];
    }
  }
  return allCoordinates;
}

function getAllSclCoordinatesUpToQmlReady(sessionKey, currentTm) {
  const sclHistory = sclHistoryBySession[sessionKey] || [];
  let allSclCoordinates = [];
  for (const sclRecord of sclHistory) {
    if (sclRecord.tm < currentTm) {
      allSclCoordinates = [...allSclCoordinates, ...sclRecord.coordinates];
    }
  }
  return allSclCoordinates;
}

function buildSessionSummaryAtTime(sessionKey, sessA, cutoffTm) {
  const previousTmvCoordinates = getAllTmvCoordinatesUpToQmlReady(sessionKey, cutoffTm);
  const sclCoordinates = getAllSclCoordinatesUpToQmlReady(sessionKey, cutoffTm);
  const resultCoordinates = [...previousTmvCoordinates, ...sclCoordinates];
  const sm = new MouseMoveStatistics();

  for (const point of resultCoordinates) {
    sm.run(point);
  }

  const statistics = sm.send();
  if (!statistics) {
    return null;
  }

  let scrollData = [];
  if (sclCoordinates.length > 0) {
    const sclHistory = sclHistoryBySession[sessionKey] || [];
    for (const sclRecord of sclHistory) {
      if (sclRecord.tm < cutoffTm) {
        const coords = sclRecord.coordinates;
        if (coords.length > 1) {
          const scrollDelta = coords[coords.length - 1].clientY - coords[0].clientY;
          scrollData.push(scrollDelta);
        }
      }
    }
  }

  return {
    sessA,
    tm: cutoffTm,
    ts: null,
    ip: null,
    ua: '',
    type: 'qml_ready',
    synthetic: true,
    syntheticReason: 'session_end_or_no_next_ready',
    statistics,
    sclCount: sclCoordinates.length,
    scrollData,
  };
}

function extractIntermediateData(obj) {
  const sessA = obj?.sess?.a;
  const tm = obj?.prm?.tm;
  const ts = obj?.ts;
  const ip = obj?.ip;

  if (!sessA || !tm) {
    return null;
  }

  const sessionKey = getSessionKey(sessA);
  const hasTmvRecord = hasTmv(obj);
  const hasClkRecord = hasClk(obj);
  const hasSclRecord = hasScl(obj);

  if (hasSclRecord) {
    const sclL = obj.prm.data.e.scl.l;
    const sclRecord = {
      sessA,
      tm,
      coordinates: parseSclCoordinates(sclL, tm),
      type: 'scl'
    };

    if (!sclHistoryBySession[sessionKey]) {
      sclHistoryBySession[sessionKey] = [];
    }
    sclHistoryBySession[sessionKey].push(sclRecord);

    if (sclHistoryBySession[sessionKey].length > MAX_TMV_RECORDS) {
      sclHistoryBySession[sessionKey] = sclHistoryBySession[sessionKey].slice(-MAX_TMV_RECORDS);
    }
  }

  if (hasTmvRecord) {
    const tmvL = obj.prm.data.e.tmv.l;
    const tmvRecord = {
      sessA,
      tm,
      coordinates: parseTmvCoordinates(tmvL, tm),
      type: hasClkRecord ? 'tmv_clk' : 'tmv'
    };

    if (!tmvHistoryBySession[sessionKey]) {
      tmvHistoryBySession[sessionKey] = [];
    }
    tmvHistoryBySession[sessionKey].push(tmvRecord);

    if (tmvHistoryBySession[sessionKey].length > MAX_TMV_RECORDS) {
      tmvHistoryBySession[sessionKey] = tmvHistoryBySession[sessionKey].slice(-MAX_TMV_RECORDS);
    }

    lastTmvBySession[sessionKey] = tmvRecord;
  }

  if (isQmlReady(obj)) {
    const previousTmvCoordinates = getAllTmvCoordinatesUpToQmlReady(sessionKey, tm);
    const sclCoordinates = getAllSclCoordinatesUpToQmlReady(sessionKey, tm);
    const ua = (obj?.ua && typeof obj.ua === 'object' && obj.ua.v != null) ? String(obj.ua.v) : '';
    return {
      sessA,
      tm,
      ts,
      ip,
      ua,
      type: 'qml_ready',
      previousTmvCoordinates,
      sclCoordinates,
      sclCount: sclCoordinates.length,
      currentCoordinates: []
    };
  }

  return null;
}

function processIntermediateData(intermediateRecord) {
  const { currentCoordinates = [], previousTmvCoordinates, sclCoordinates, ...recordData } = intermediateRecord;
  const resultCoordinates = [...previousTmvCoordinates, ...sclCoordinates, ...currentCoordinates];
  const sm = new MouseMoveStatistics();

  for (const point of resultCoordinates) {
    sm.run(point);
  }

  const statistics = sm.send();

  if (statistics) {
    let scrollData = [];
    if (sclCoordinates.length > 0) {
      const sclHistory = sclHistoryBySession[getSessionKey(recordData.sessA)] || [];
      for (const sclRecord of sclHistory) {
        if (sclRecord.tm < recordData.tm) {
          const coords = sclRecord.coordinates;
          if (coords.length > 1) {
            const scrollDelta = coords[coords.length - 1].clientY - coords[0].clientY;
            scrollData.push(scrollDelta);
          }
        }
      }
    }

    return {
      ...recordData,
      statistics,
      scrollData
    };
  }

  return null;
}

function processRecord(obj) {
  const sessA = obj?.sess?.a;
  const tm = obj?.prm?.tm;
  const ts = obj?.ts;
  const ip = obj?.ip;

  if (!sessA || !tm) {
    return null;
  }

  const sessionKey = getSessionKey(sessA);
  lastEventTmBySession[sessionKey] = tm;
  const hasTmvRecord = hasTmv(obj);
  const hasClkRecord = hasClk(obj);
  const hasSclRecord = hasScl(obj);

  if (hasSclRecord) {
    const sclL = obj.prm.data.e.scl.l;
    const sclRecord = {
      sessA,
      tm,
      coordinates: parseSclCoordinates(sclL, tm),
      type: 'scl'
    };

    if (!sclHistoryBySession[sessionKey]) {
      sclHistoryBySession[sessionKey] = [];
    }
    sclHistoryBySession[sessionKey].push(sclRecord);

    if (sclHistoryBySession[sessionKey].length > MAX_TMV_RECORDS) {
      sclHistoryBySession[sessionKey] = sclHistoryBySession[sessionKey].slice(-MAX_TMV_RECORDS);
    }
  }

  if (hasTmvRecord) {
    const tmvL = obj.prm.data.e.tmv.l;
    const tmvRecord = {
      sessA,
      tm,
      coordinates: parseTmvCoordinates(tmvL, tm),
      type: hasClkRecord ? 'tmv_clk' : 'tmv'
    };

    if (!tmvHistoryBySession[sessionKey]) {
      tmvHistoryBySession[sessionKey] = [];
    }
    tmvHistoryBySession[sessionKey].push(tmvRecord);

    if (tmvHistoryBySession[sessionKey].length > MAX_TMV_RECORDS) {
      tmvHistoryBySession[sessionKey] = tmvHistoryBySession[sessionKey].slice(-MAX_TMV_RECORDS);
    }

    lastTmvBySession[sessionKey] = tmvRecord;
  }

  if (isQmlReady(obj)) {
    const previousTmvCoordinates = getAllTmvCoordinatesUpToQmlReady(sessionKey, tm);
    const sclCoordinates = getAllSclCoordinatesUpToQmlReady(sessionKey, tm);
    const resultCoordinates = [...previousTmvCoordinates, ...sclCoordinates];
    const sm = new MouseMoveStatistics();

    for (const point of resultCoordinates) {
      sm.run(point);
    }

    const statistics = sm.send();

    if (statistics) {
      sclAggregated += sclCoordinates.length;

      let scrollData = [];
      if (sclCoordinates.length > 0) {
        const sclHistory = sclHistoryBySession[sessionKey] || [];
        for (const sclRecord of sclHistory) {
          if (sclRecord.tm < tm) {
            const coords = sclRecord.coordinates;
            if (coords.length > 1) {
              const scrollDelta = coords[coords.length - 1].clientY - coords[0].clientY;
              scrollData.push(scrollDelta);
            }
          }
        }
      }

      const ua = (obj?.ua && typeof obj.ua === 'object' && obj.ua.v != null) ? String(obj.ua.v) : '';
      return {
        sessA,
        tm,
        ts,
        ip,
        ua,
        type: 'qml_ready',
        statistics: statistics,
        sclCount: sclCoordinates.length,
        scrollData: scrollData
      };
    }
  }

  return null;
}

async function extractData() {
  console.log('=== ЭТАП 1: Извлечение данных ===');
  const intermediateOutput = fs.createWriteStream(intermediateFile, { flags: 'w' });
  intermediateOutput.write('[\n');

  const rl = readline.createInterface({
    input: fs.createReadStream(inputFile),
    crlfDelay: Infinity
  });

  let isFirst = true;
  let extracted = 0;

  for await (const line of rl) {
    if (!line.trim()) continue;
    let obj;
    try {
      obj = JSON.parse(line);
      parsed++;
    } catch (e) {
      console.error('Ошибка парсинга:', e, line.slice(0, 100));
      continue;
    }

    const result = extractIntermediateData(obj);
    if (result) {
      intermediateOutput.write((isFirst ? '' : ',\n') + JSON.stringify(result));
      isFirst = false;
      extracted++;
    }
  }

  intermediateOutput.write('\n]\n');
  intermediateOutput.end();
  await waitWritableFinish(intermediateOutput);

  console.log('Извлечение завершено!');
  console.log('Успешно распарсено объектов:', parsed);
  console.log('Извлечено записей с qml.ready:', extracted);
  console.log('Промежуточные данные сохранены в:', intermediateFile);

  return extracted;
}

async function processData() {
  console.log('=== ЭТАП 2: Обработка данных ===');

  if (!fs.existsSync(intermediateFile)) {
    console.error('Промежуточный файл не найден! Сначала запустите извлечение данных.');
    return;
  }

  const output = fs.createWriteStream(outputFile, { flags: 'w' });
  output.write('[\n');

  const rl = readline.createInterface({
    input: fs.createReadStream(intermediateFile),
    crlfDelay: Infinity
  });

  let isFirst = true;
  let processed = 0;
  let sclAggregated = 0;
  let lineNumber = 0;

  for await (const line of rl) {
    lineNumber++;
    if (lineNumber === 1 && line.trim() === '[') continue;
    if (line.trim() === ']') continue;

    let cleanLine = line.trim();
    if (cleanLine.endsWith(',')) {
      cleanLine = cleanLine.slice(0, -1);
    }

    if (!cleanLine) continue;

    let record;
    try {
      record = JSON.parse(cleanLine);
    } catch (e) {
      console.error(`Ошибка парсинга строки ${lineNumber}:`, e.message);
      continue;
    }

    const result = processIntermediateData(record);
    if (result) {
      output.write((isFirst ? '' : ',\n') + JSON.stringify(result));
      isFirst = false;
      processed++;
      sclAggregated += result.sclCount || 0;
    }
  }

  output.write('\n]\n');
  output.end();
  await waitWritableFinish(output);

  console.log('Обработка завершена!');
  console.log('Обработано записей с qml.ready:', processed);
  console.log('Агрегировано скроллов:', sclAggregated);
  console.log('Результат сохранен в:', outputFile);
}

async function processIncremental() {
  console.log('=== ИНКРЕМЕНТАЛЬНАЯ ОБРАБОТКА ===');
  console.log(`Входной файл: ${inputFile}`);
  console.log(`Выходной файл: ${outputFile}`);
  console.log(`Checkpoint: ${checkpointFile}`);
  console.log(`Changed sessions: ${changedSessionsFile}`);

  if (!fs.existsSync(inputFile)) {
    console.error(`Ошибка: файл ${inputFile} не найден!`);
    process.exit(1);
  }

  const stats = fs.statSync(inputFile);
  const checkpoint = loadCheckpoint();
  const rebuild = shouldRebuild(checkpoint, stats);
  const startOffset = rebuild || !checkpoint ? 0 : (checkpoint.offset || 0);

  if (rebuild) {
    console.log('Режим: полный пересбор статистики из mobile NDJSON.');
    restoreStateFromCheckpoint(null);
  } else {
    console.log(`Режим: инкрементальный запуск с offset=${startOffset}.`);
    restoreStateFromCheckpoint(checkpoint);
  }

  const { lines, nextOffset, hasPartialTail } = await readNewCompleteLines(startOffset);
  if (hasPartialTail) {
    console.log('Обнаружена незавершённая последняя строка, она будет обработана на следующем запуске.');
  }

  const existingOutput = rebuild ? [] : loadJsonFile(outputFile, []);
  const newResults = [];
  const changedSessions = new Set();
  const touchedSessionKeys = new Set();

  for (const line of lines) {
    if (!line.trim()) {
      continue;
    }

    let obj;
    try {
      obj = JSON.parse(line);
      parsed++;
    } catch (error) {
      console.error('Ошибка парсинга:', error, line.slice(0, 100));
      continue;
    }

    const result = processRecord(obj);
    if (result) {
      newResults.push(result);
      changedSessions.add(String(result.sessA));
      touchedSessionKeys.add(getSessionKey(result.sessA));
      processed++;
      sclAggregated += result.sclCount || 0;
    }

    const sessA = obj?.sess?.a;
    const tm = obj?.prm?.tm;
    if (sessA && tm) {
      touchedSessionKeys.add(getSessionKey(sessA));
    }
  }

  // Flush sessions that did not have qml.ready after their last activity.
  // This covers the case: single page -> actions (qml) -> session ends without another qml.ready.
  let syntheticEmitted = 0;
  for (const sessionKey of touchedSessionKeys) {
    const cutoffTm = lastEventTmBySession[sessionKey];
    if (!cutoffTm) {
      continue;
    }

    // If a real qml.ready exists with tm equal to the last event, no need to emit synthetic.
    // Otherwise, try to build a synthetic summary at session end.
    const sessA = sessionKey;
    const summary = buildSessionSummaryAtTime(sessionKey, sessA, cutoffTm);
    if (!summary) {
      continue;
    }

    // Avoid duplicating an identical synthetic record already produced in this run.
    // (qml.ready results have synthetic=false/undefined and are fine to coexist).
    const alreadyInBatch = newResults.some(
      (item) =>
        String(item?.sessA) === String(sessA) &&
        Number(item?.tm) === Number(cutoffTm) &&
        item?.synthetic === true
    );
    if (alreadyInBatch) {
      continue;
    }

    newResults.push(summary);
    changedSessions.add(String(summary.sessA));
    processed++;
    sclAggregated += summary.sclCount || 0;
    syntheticEmitted++;
  }

  const mergedOutput = rebuild ? newResults : existingOutput.concat(newResults);
  writeJsonAtomic(outputFile, mergedOutput);
  writeChangedSessions(changedSessionsFile, changedSessions, rebuild);
  writeJsonAtomic(checkpointFile, buildCheckpoint(stats, nextOffset));

  console.log('Инкрементальная обработка завершена!');
  console.log('Новых строк NDJSON обработано:', lines.length);
  console.log('Новых/обновлённых qml.ready записей:', newResults.length);
  console.log('Синтетических записей (флаш сессий без нового qml.ready):', syntheticEmitted);
  console.log('Изменённых session_id:', changedSessions.size);
  console.log('Агрегировано скроллов:', sclAggregated);
  console.log('Результат сохранен в:', outputFile);
}

async function processFile() {
  if (mode === 'extract') {
    await extractData();
  } else if (mode === 'process') {
    await processData();
  } else if (mode === 'incremental') {
    await processIncremental();
  } else if (mode === 'full') {
    await extractData();
    await processData();
  } else {
    console.error('Неизвестный режим:', mode);
    console.error('Доступные режимы: extract, process, incremental, full');
    process.exit(1);
  }
}

module.exports = { MouseMoveStatistics };

processFile().catch(console.error);
