require('../../js/load-env').loadEnv();
const fs = require('fs');
const readline = require('readline');

// Название сайта (через SITE_NAME) добавляется в имена входного, промежуточного и выходного файлов.
// Например: SITE_NAME=zwilling INPUT_FILE=desktop.zwilling.ndjson MAX_MMV_RECORDS=10 MODE=full node src/process_mmv_clk_enhanced_new.js
const siteName = (process.env.SITE_NAME || '').trim().replace(/[^\w.-]/g, '_') || null;

// Входной файл: INPUT_FILE или JSON_PATH (для совместимости с пайплайном).
const inputFile = process.env.INPUT_FILE || process.env.JSON_PATH || 'desktop.ndjson';

// Промежуточный файл: зависит от SITE_NAME (если задан).
const intermediateFile = siteName
  ? `intermediate_${siteName}.json`
  : 'intermediate_data_desktop.json';

// Количество последних mmv записей для сбора
const MAX_MMV_RECORDS = parseInt(process.env.MAX_MMV_RECORDS, 10) || 5;

// Выходной файл: OUTPUT_FILE / MMV_STATISTICS_JSON или имя по SITE_NAME.
const outputFile = process.env.OUTPUT_FILE
  || process.env.MMV_STATISTICS_JSON
  || (siteName
    ? `${siteName}_mmv_clk_statistics_max${MAX_MMV_RECORDS}.json`
    : 'desktop_mmv_clk_statistics.json');

const checkpointFile = process.env.MMV_CHECKPOINT_FILE || 'mmv_process_checkpoint.json';
const changedSessionsFile = process.env.CHANGED_SESSIONS_FILE || 'changed_sessions.json';
const forceFullRebuild = process.env.FORCE_FULL_REBUILD === '1';

let parsed = 0;
let processed = 0;
let sclAggregated = 0;
let isFirst = true;

// Режим работы: extract, process, incremental, full
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
  fs.writeFileSync(tmpPath, JSON.stringify(value, null, 2));
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
  for (const key of Object.keys(mmvHistoryBySession)) {
    delete mmvHistoryBySession[key];
  }
  for (const key of Object.keys(sclHistoryBySession)) {
    delete sclHistoryBySession[key];
  }
  for (const key of Object.keys(lastMmvBySession)) {
    delete lastMmvBySession[key];
  }

  if (!checkpoint) {
    return;
  }

  const mmvState = checkpoint.mmvHistoryBySession || {};
  for (const [sessionKey, history] of Object.entries(mmvState)) {
    mmvHistoryBySession[sessionKey] = Array.isArray(history) ? history : [];
    const lastRecord = mmvHistoryBySession[sessionKey][mmvHistoryBySession[sessionKey].length - 1];
    if (lastRecord) {
      lastMmvBySession[sessionKey] = lastRecord;
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
    mmvHistoryBySession,
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
  if (previous.ino !== null && stats.ino !== undefined && previous.ino !== stats.ino) {
    return true;
  }

  if (stats.size < (checkpoint.offset || 0)) {
    return true;
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
        this.mmvGroupSizes = [];
    }

    // Сброс данных
    reset() {
        this.points = [];
        this.mmvGroupSizes = [];
    }

    // Размеры mmv-групп (число точек в каждой записи mmv) для признака пакетирования
    setMmvGroups(groups) {
        this.mmvGroupSizes = (groups || [])
            .filter(group => Array.isArray(group) && group.length > 0)
            .map(group => group.length);
    }

    _percentile(values, p) {
        if (!values.length) return 0;
        const sorted = [...values].sort((a, b) => a - b);
        const rank = (sorted.length - 1) * (p / 100);
        const lower = Math.floor(rank);
        const upper = Math.ceil(rank);
        if (lower === upper) return sorted[lower];
        const weight = rank - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    _getStepVectors() {
        const vectors = [];
        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i - 1].x;
            const dy = this.points[i].y - this.points[i - 1].y;
            vectors.push({
                dx,
                dy,
                dist: Math.sqrt(dx * dx + dy * dy)
            });
        }
        return vectors;
    }

    _angleBetween(v1, v2) {
        const cross = v1.dx * v2.dy - v1.dy * v2.dx;
        const dot = v1.dx * v2.dx + v1.dy * v2.dy;
        if (dot === 0 && cross === 0) return null;
        return Math.atan2(cross, dot);
    }

    _getAbsTurnAnglesDeg() {
        const vectors = this._getStepVectors();
        const angles = [];
        for (let i = 1; i < vectors.length; i++) {
            const angle = this._angleBetween(vectors[i - 1], vectors[i]);
            if (angle !== null) {
                angles.push(Math.abs(angle * 180 / Math.PI));
            }
        }
        return angles;
    }

    _calculateDiscreteEntropy(values, binSize) {
        if (!values.length || binSize <= 0) return 0;
        const counts = new Map();
        values.forEach(value => {
            const bin = Math.floor(value / binSize);
            counts.set(bin, (counts.get(bin) || 0) + 1);
        });
        const total = values.length;
        let entropy = 0;
        counts.forEach(count => {
            const p = count / total;
            entropy -= p * Math.log2(p);
        });
        return entropy;
    }

    // Добавление новой точки траектории
    addPoint(x, y, timestamp) {
        this.points.push({ x, y, timestamp });
        
        // Ограничиваем размер массива
        if (this.points.length > this.maxPoints) {
            this.points = this.points.slice(-this.maxPoints);
        }
    }

    // Обработка события движения мыши
    run(event) {
        const { clientX: x, clientY: y, tm: timestamp } = event;
        
        // Проверяем, есть ли движение
        if (this.points.length > 0) {
            const lastPoint = this.points[this.points.length - 1];
            const distance = Math.sqrt((x - lastPoint.x) ** 2 + (y - lastPoint.y) ** 2);
            
            // Игнорируем очень маленькие движения (шум)
            if (distance < 1) return;
            
            // Проверяем паузу в движении (более 15 секунд - новое движение)
            if (timestamp - lastPoint.timestamp > 15000) {
                this.reset();
            }
        }
        
        this.addPoint(x, y, timestamp);
    }

    // Вычисление энтропии направлений движения
    calculateDirectionEntropy() {
        if (this.points.length < 2) return 0;
        
        const directions = [];
        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i-1].x;
            const dy = this.points[i].y - this.points[i-1].y;
            const angle = Math.atan2(dy, dx);
            directions.push(angle);
        }
        
        // Группируем направления в интервалы
        const bins = 16; // 16 направлений
        const binSize = (2 * Math.PI) / bins;
        const counts = new Array(bins).fill(0);
        
        directions.forEach(angle => {
            let normalizedAngle = angle + Math.PI; // Преобразуем в [0, 2π]
            const binIndex = Math.floor(normalizedAngle / binSize);
            counts[binIndex % bins]++;
        });
        
        // Вычисляем энтропию
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

    // Вычисление джерка (третья производная по времени)
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
                // Скорости
                const v1x = (p2.x - p1.x) / dt1;
                const v1y = (p2.y - p1.y) / dt1;
                const v2x = (p3.x - p2.x) / dt2;
                const v2y = (p3.y - p2.y) / dt2;
                const v3x = (p4.x - p3.x) / dt3;
                const v3y = (p4.y - p3.y) / dt3;
                
                // Ускорения
                const a1x = (v2x - v1x) / dt2;
                const a1y = (v2y - v1y) / dt2;
                const a2x = (v3x - v2x) / dt3;
                const a2y = (v3y - v2y) / dt3;
                
                // Джерк
                const jerkx = (a2x - a1x) / dt3;
                const jerky = (a2y - a1y) / dt3;
                const jerk = Math.sqrt(jerkx * jerkx + jerky * jerky);
                
                jerks.push(jerk);
            }
        }
        
        return jerks.length > 0 ? jerks.reduce((a, b) => a + b, 0) / jerks.length : 0;
    }

    // Вычисление линейности траектории
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

    // Вычисление микродвижений (дрожания)
    calculateJitter(threshold = 3) {
        if (this.points.length < 2) return 0;
        
        let jitterCount = 0;
        let totalMovements = 0;
        
        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i-1].x;
            const dy = this.points[i].y - this.points[i-1].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            totalMovements++;
            if (distance < threshold) {
                jitterCount++;
            }
        }
        
        return totalMovements > 0 ? jitterCount / totalMovements : 0;
    }

    // Микродрожание: доля шагов короче 2 px
    calculateMicroJitter2px() {
        return this.calculateJitter(2);
    }

    // Среднее число точек mmv в одной записи (пакетирование)
    calculateMmvPointsPerGroup() {
        if (this.mmvGroupSizes.length === 0) return 0;
        const total = this.mmvGroupSizes.reduce((sum, size) => sum + size, 0);
        return total / this.mmvGroupSizes.length;
    }

    // Доля шагов, входящих в длинные почти прямые серии (угол < 5°)
    calculateLongStraightRunShare(straightAngleDeg = 5, minRunLength = 5) {
        const absAngles = this._getAbsTurnAnglesDeg();
        if (!absAngles.length) return 0;

        let currentRun = 0;
        let straightSteps = 0;

        absAngles.forEach(angle => {
            if (angle < straightAngleDeg) {
                currentRun++;
                return;
            }
            if (currentRun >= minRunLength) {
                straightSteps += currentRun;
            }
            currentRun = 0;
        });

        if (currentRun >= minRunLength) {
            straightSteps += currentRun;
        }

        return straightSteps / absAngles.length;
    }

    // Доля поворотов с углом > 45° на 100 пар соседних векторов
    calculateTurnRate45Per100() {
        const absAngles = this._getAbsTurnAnglesDeg();
        if (!absAngles.length) return 0;
        const sharpTurns = absAngles.filter(angle => angle > 45).length;
        return 100 * sharpTurns / absAngles.length;
    }

    calculateMeanAbsAngleDeg() {
        const absAngles = this._getAbsTurnAnglesDeg();
        if (!absAngles.length) return 0;
        return absAngles.reduce((sum, angle) => sum + angle, 0) / absAngles.length;
    }

    calculateP90AbsAngleDeg() {
        return this._percentile(this._getAbsTurnAnglesDeg(), 90);
    }

    // Энтропия распределения углов поворота (бины по 10°)
    calculateAngleEntropy(binSizeDeg = 10) {
        return this._calculateDiscreteEntropy(this._getAbsTurnAnglesDeg(), binSizeDeg);
    }

    // Сумма углов поворота (рад) на 100 px пройденного пути
    calculateCurvaturePer100px() {
        const vectors = this._getStepVectors();
        const absAngles = this._getAbsTurnAnglesDeg();
        const pathLength = vectors.reduce((sum, vector) => sum + vector.dist, 0);
        if (!pathLength || !absAngles.length) return 0;

        const radiansSum = absAngles.reduce((sum, angle) => sum + (angle * Math.PI / 180), 0);
        return 100 * radiansSum / pathLength;
    }

    // Вычисление вариативности скорости
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
        
        return Math.sqrt(variance) / meanSpeed; // Коэффициент вариации
    }

    // Вычисление кривизны траектории
    calculateCurvature() {
        if (this.points.length < 3) return 0;
        
        const curvatures = [];
        for (let i = 1; i < this.points.length - 1; i++) {
            const p1 = this.points[i-1];
            const p2 = this.points[i];
            const p3 = this.points[i+1];
            
            // Вычисляем кривизну по формуле для дискретных точек
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

    // Вычисление паттернов повторяемости
    calculateRepetitionPatterns() {
        if (this.points.length < 6) return 0;
        
        const segmentLength = 3; // Длина сегмента для сравнения
        let repetitions = 0;
        let totalComparisons = 0;
        
        for (let i = 0; i <= this.points.length - segmentLength * 2; i++) {
            for (let j = i + segmentLength; j <= this.points.length - segmentLength; j++) {
                const segment1 = this.points.slice(i, i + segmentLength);
                const segment2 = this.points.slice(j, j + segmentLength);
                
                // Нормализуем сегменты (приводим к началу координат)
                const normalized1 = this.normalizeSegment(segment1);
                const normalized2 = this.normalizeSegment(segment2);
                
                // Вычисляем схожесть сегментов
                const similarity = this.calculateSegmentSimilarity(normalized1, normalized2);
                totalComparisons++;
                
                if (similarity > 0.8) { // Порог схожести
                    repetitions++;
                }
            }
        }
        
        return totalComparisons > 0 ? repetitions / totalComparisons : 0;
    }

    // Нормализация сегмента
    normalizeSegment(segment) {
        const start = segment[0];
        return segment.map(point => ({
            x: point.x - start.x,
            y: point.y - start.y
        }));
    }

    // Вычисление схожести сегментов
    calculateSegmentSimilarity(seg1, seg2) {
        if (seg1.length !== seg2.length) return 0;
        
        let totalDistance = 0;
        for (let i = 0; i < seg1.length; i++) {
            const dx = seg1[i].x - seg2[i].x;
            const dy = seg1[i].y - seg2[i].y;
            totalDistance += Math.sqrt(dx * dx + dy * dy);
        }
        
        const avgDistance = totalDistance / seg1.length;
        return Math.max(0, 1 - avgDistance / 50); // Нормализация по максимальному расстоянию
    }

    // Вычисление временных интервалов
    calculateTimeIntervals() {
        if (this.points.length < 2) return { mean: 0, std: 0, irregularity: 0 };
        
        const intervals = [];
        for (let i = 1; i < this.points.length; i++) {
            intervals.push(this.points[i].timestamp - this.points[i-1].timestamp);
        }
        
        const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const variance = intervals.reduce((sum, interval) => sum + (interval - mean) ** 2, 0) / intervals.length;
        const std = Math.sqrt(variance);
        const irregularity = std / mean; // Коэффициент вариации
        
        return { mean, std, irregularity };
    }

    // Вычисление общего расстояния
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

    // Вычисление средней скорости
    calculateAverageSpeed() {
        const totalDistance = this.calculateTotalDistance();
        const totalDuration = this.points[this.points.length - 1].timestamp - this.points[0].timestamp;
        
        return totalDuration > 0 ? totalDistance / totalDuration : 0;
    }

    // Вычисление вариативности ускорения
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

    // Вычисление общего индекса бота
    calculateBotScore(features) {
        let score = 0;
        
        // Низкая энтропия направлений указывает на бота
        if (features.directionEntropy < 2.0) score += 0.3;
        else if (features.directionEntropy < 2.5) score += 0.1;
        
        // Высокая линейность указывает на бота
        if (features.linearity > 0.9) score += 0.2;
        else if (features.linearity > 0.8) score += 0.1;
        
        // Низкий джерк указывает на бота
        if (features.jerk < 0.001) score += 0.2;
        else if (features.jerk < 0.01) score += 0.1;
        
        // Низкое дрожание указывает на бота
        if (features.jitter < 0.1) score += 0.15;
        else if (features.jitter < 0.2) score += 0.05;

        if (features.microJitter2px < 0.12) score += 0.1;
        else if (features.microJitter2px < 0.15) score += 0.05;

        if (features.mmvPointsPerGroup >= 30) score += 0.15;
        else if (features.mmvPointsPerGroup >= 25) score += 0.05;

        if (features.longStraightRunShare >= 0.45) score += 0.2;
        else if (features.longStraightRunShare >= 0.35) score += 0.1;

        if (features.turnRate45Per100 < 3) score += 0.15;
        else if (features.turnRate45Per100 < 4) score += 0.05;

        if (features.meanAbsAngleDeg < 9) score += 0.1;
        if (features.angleEntropy < 2.2) score += 0.1;
        if (features.curvaturePer100px < 3) score += 0.05;
        
        // Низкая вариативность скорости указывает на бота
        if (features.speedVariability < 0.3) score += 0.15;
        else if (features.speedVariability < 0.5) score += 0.05;
        
        // Высокие паттерны повторяемости указывают на бота
        if (features.repetitionPatterns > 0.3) score += 0.2;
        else if (features.repetitionPatterns > 0.1) score += 0.1;
        
        // Низкая нерегулярность временных интервалов указывает на бота
        if (features.timeIntervals.irregularity < 0.3) score += 0.1;
        
        return Math.min(1.0, score); // Ограничиваем максимальный балл
    }

    // Основной метод для получения всех признаков
    getBotDetectionFeatures() {
        if (this.points.length < this.minPointsForAnalysis) {
            return null;
        }
        
        const features = {
            // Основные признаки
            pointCount: this.points.length,
            totalDuration: this.points[this.points.length - 1].timestamp - this.points[0].timestamp,
            
            // Признаки для детекции ботов
            directionEntropy: this.calculateDirectionEntropy(),
            jerk: this.calculateJerk(),
            linearity: this.calculateLinearity(),
            jitter: this.calculateJitter(),
            microJitter2px: this.calculateMicroJitter2px(),
            mmvPointsPerGroup: this.calculateMmvPointsPerGroup(),
            longStraightRunShare: this.calculateLongStraightRunShare(),
            turnRate45Per100: this.calculateTurnRate45Per100(),
            meanAbsAngleDeg: this.calculateMeanAbsAngleDeg(),
            p90AbsAngleDeg: this.calculateP90AbsAngleDeg(),
            angleEntropy: this.calculateAngleEntropy(),
            curvaturePer100px: this.calculateCurvaturePer100px(),
            speedVariability: this.calculateSpeedVariability(),
            curvature: this.calculateCurvature(),
            repetitionPatterns: this.calculateRepetitionPatterns(),
            
            // Временные характеристики
            timeIntervals: this.calculateTimeIntervals(),
            
            // Дополнительные признаки
            totalDistance: this.calculateTotalDistance(),
            averageSpeed: this.calculateAverageSpeed(),
            accelerationVariability: this.calculateAccelerationVariability()
        };
        
        // Вычисляем общий "индекс бота" (чем выше, тем больше вероятность бота)
        features.botScore = this.calculateBotScore(features);
        
        return features;
    }

    // Метод для совместимости с существующим кодом
    send() {
        const features = this.getBotDetectionFeatures();
        if (!features) return null;
        
        // Возвращаем данные в формате, совместимом с существующим кодом
        return {
            '_b': {
                'n': features.pointCount,
                'tm': this.points[this.points.length - 1].timestamp,
                'ms': features.totalDuration,
                'cxy2': [[this.points[0].x, this.points[0].y], [this.points[this.points.length - 1].x, this.points[this.points.length - 1].y]],
                'l': features.totalDistance
            },
            // Новые признаки для детекции ботов
            'bot_detection': {
                'direction_entropy': features.directionEntropy,
                'jerk': features.jerk,
                'linearity': features.linearity,
                'jitter': features.jitter,
                'micro_jitter_2px': features.microJitter2px,
                'mmv_points_per_group': features.mmvPointsPerGroup,
                'long_straight_run_share': features.longStraightRunShare,
                'turn_rate_45_per_100': features.turnRate45Per100,
                'mean_abs_angle_deg': features.meanAbsAngleDeg,
                'p90_abs_angle_deg': features.p90AbsAngleDeg,
                'angle_entropy': features.angleEntropy,
                'curvature_per_100px': features.curvaturePer100px,
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

// Для хранения последних mmv записей по сессиям
const lastMmvBySession = {};
// Для хранения всех mmv записей по сессиям (для поиска ближайшей предыдущей)
const mmvHistoryBySession = {};
// Для хранения всех scl записей по сессиям (для агрегации с mmv+clk)
const sclHistoryBySession = {};
const lastEventTmBySession = {};
const firstQmlTmBySession = {};
const hasQmlReadyBySession = {};
const lastQmlReadyTmBySession = {};

function getSessionKey(sessA) {
  return `${sessA}`;
}

function parseMmvCoordinates(lstr, tm) {
  let absTm = tm;
  return lstr.split(';').map(el => {
    const [t, x, y] = el.split(',').map(val => parseInt(val, 36));
    absTm += t;
    return { clientX: x, clientY: y, tm: absTm };
  });
}

function hasMmv(obj) {
  return obj?.prm?.data?.e?.mmv?.l;
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

function isQml(obj) {
  return obj?.act === 'qml';
}

function parseSclCoordinates(lstr, tm) {
  let absTm = tm;
  return lstr.split(';').map(el => {
    const [t, x, y] = el.split(',').map(val => parseInt(val, 36));
    absTm += t;
    return { clientX: x, clientY: y, tm: absTm };
  });
}

function getAllMmvCoordinatesUpToPreviousClk(sessionKey, currentTm) {
  const history = mmvHistoryBySession[sessionKey] || [];
  
  // Собираем координаты из последних MAX_MMV_RECORDS записей mmv
  let allCoordinates = [];
  
  // Проходим по истории в обратном порядке, чтобы найти ближайшую предыдущую mmv+clk
  for (let i = history.length - 1; i >= 0; i--) {
    const mmvRecord = history[i];
    
    // Если это запись mmv+clk (а не просто mmv), то останавливаемся
    // и собираем все координаты от начала до этой записи
    if (mmvRecord.type === 'mmv_clk' && mmvRecord.tm < currentTm) {
      // Собираем все координаты от начала до этой записи
      for (let j = 0; j <= i; j++) {
        allCoordinates = [...allCoordinates, ...history[j].coordinates];
      }
      break;
    }
    
    // Если это просто mmv запись, продолжаем искать
    if (mmvRecord.type === 'mmv' && mmvRecord.tm < currentTm) {
      // Если это первая запись или предыдущая была mmv+clk, то собираем координаты
      if (i === 0 || (i > 0 && history[i-1].type === 'mmv_clk')) {
        allCoordinates = [...allCoordinates, ...mmvRecord.coordinates];
      }
    }
  }
  
  return allCoordinates;
}

function getMmvGroupsUpToPreviousClk(sessionKey, currentTm) {
  const history = mmvHistoryBySession[sessionKey] || [];
  let groups = [];

  for (let i = history.length - 1; i >= 0; i--) {
    const mmvRecord = history[i];

    if (mmvRecord.type === 'mmv_clk' && mmvRecord.tm < currentTm) {
      for (let j = 0; j <= i; j++) {
        if (history[j].coordinates?.length) {
          groups.push(history[j].coordinates);
        }
      }
      break;
    }

    if (mmvRecord.type === 'mmv' && mmvRecord.tm < currentTm) {
      if (i === 0 || (i > 0 && history[i - 1].type === 'mmv_clk')) {
        groups = [...groups, ...[mmvRecord.coordinates].filter(coords => coords?.length)];
      }
    }
  }

  return groups;
}

function getMmvGroupsUpToQmlReady(sessionKey, currentTm) {
  const history = mmvHistoryBySession[sessionKey] || [];
  const groups = [];

  for (const mmvRecord of history) {
    if (mmvRecord.tm < currentTm && mmvRecord.coordinates?.length) {
      groups.push(mmvRecord.coordinates);
    }
  }

  return groups;
}

function buildMouseMoveStatistics(resultCoordinates, mmvGroups) {
  const sm = new MouseMoveStatistics();
  sm.setMmvGroups(mmvGroups);

  for (const point of resultCoordinates) {
    sm.run(point);
  }

  return sm.send();
}

function getAllSclCoordinatesBetweenClks(sessionKey, currentTm) {
  const sclHistory = sclHistoryBySession[sessionKey] || [];
  const mmvHistory = mmvHistoryBySession[sessionKey] || [];
  
  // Находим время предыдущей mmv+clk записи
  let previousClkTime = null;
  for (let i = mmvHistory.length - 1; i >= 0; i--) {
    const mmvRecord = mmvHistory[i];
    if (mmvRecord.type === 'mmv_clk' && mmvRecord.tm < currentTm) {
      previousClkTime = mmvRecord.tm;
      break;
    }
  }
  
  // Собираем все скроллы между предыдущим clk и текущим
  let allSclCoordinates = [];
  
  for (const sclRecord of sclHistory) {
    if (sclRecord.tm > (previousClkTime || 0) && sclRecord.tm < currentTm) {
      allSclCoordinates = [...allSclCoordinates, ...sclRecord.coordinates];
    }
  }
  
  return allSclCoordinates;
}

function getAllMmvCoordinatesUpToQmlReady(sessionKey, currentTm) {
  const history = mmvHistoryBySession[sessionKey] || [];
  let allCoordinates = [];

  for (const mmvRecord of history) {
    if (mmvRecord.tm < currentTm) {
      allCoordinates = [...allCoordinates, ...mmvRecord.coordinates];
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

function updateSessionMarkers(obj) {
  const sessA = obj?.sess?.a;
  const tm = obj?.prm?.tm;
  if (!sessA || !tm) {
    return null;
  }

  const sessionKey = getSessionKey(sessA);
  lastEventTmBySession[sessionKey] = tm;

  if (isQmlReady(obj)) {
    hasQmlReadyBySession[sessionKey] = true;
    lastQmlReadyTmBySession[sessionKey] = tm;
  }

  if (isQml(obj) && firstQmlTmBySession[sessionKey] === undefined) {
    firstQmlTmBySession[sessionKey] = tm;
  }

  return sessionKey;
}

function buildIntermediateSessionSummaryAtTime(sessionKey, sessA, cutoffTm) {
  const previousMmvCoordinates = getAllMmvCoordinatesUpToQmlReady(sessionKey, cutoffTm);
  const sclCoordinates = getAllSclCoordinatesUpToQmlReady(sessionKey, cutoffTm);

  return {
    sessA,
    tm: cutoffTm,
    ts: null,
    ip: null,
    ua: '',
    type: 'qml_ready',
    synthetic: true,
    syntheticReason: 'session_end_or_no_next_ready',
    currentCoordinates: [],
    previousMmvCoordinates,
    mmvGroups: getMmvGroupsUpToQmlReady(sessionKey, cutoffTm),
    sclCoordinates,
    sclCount: sclCoordinates.length
  };
}

// Функция для извлечения и сохранения промежуточных данных
function extractIntermediateData(obj) {
  const sessA = obj?.sess?.a;
  const tm = obj?.prm?.tm;
  const ts = obj?.ts;
  const ip = obj?.ip;
 
  if (!sessA || !tm) {
    return null;
  }
  
  const sessionKey = getSessionKey(sessA);
  const hasMmvRecord = hasMmv(obj);
  const hasClkRecord = hasClk(obj);
  const hasSclRecord = hasScl(obj);
  
  // Сохраняем scl запись в историю
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
    
    if (sclHistoryBySession[sessionKey].length > MAX_MMV_RECORDS) {
      sclHistoryBySession[sessionKey] = sclHistoryBySession[sessionKey].slice(-MAX_MMV_RECORDS);
    }
  }
  
  // Сохраняем mmv запись в историю
  if (hasMmvRecord) {
    const mmvL = obj.prm.data.e.mmv.l;
    const mmvRecord = {
      sessA,
      tm,
      coordinates: parseMmvCoordinates(mmvL, tm),
      type: hasClkRecord ? 'mmv_clk' : 'mmv'
    };
    
    if (!mmvHistoryBySession[sessionKey]) {
      mmvHistoryBySession[sessionKey] = [];
    }
    mmvHistoryBySession[sessionKey].push(mmvRecord);
    
    if (mmvHistoryBySession[sessionKey].length > MAX_MMV_RECORDS) {
      mmvHistoryBySession[sessionKey] = mmvHistoryBySession[sessionKey].slice(-MAX_MMV_RECORDS);
    }
    
    lastMmvBySession[sessionKey] = mmvRecord;
  }
  
  // Возвращаем данные для записи в промежуточный файл
  if (hasClkRecord && hasMmvRecord) {
    const currentCoordinates = parseMmvCoordinates(obj.prm.data.e.mmv.l, tm);
    const previousMmvCoordinates = getAllMmvCoordinatesUpToPreviousClk(sessionKey, tm);
    const mmvGroups = [...getMmvGroupsUpToPreviousClk(sessionKey, tm), currentCoordinates];
    const sclCoordinates = getAllSclCoordinatesBetweenClks(sessionKey, tm);
    
    return {
      sessA,
      tm,
      ts,
      ip,
      type: 'mmv_clk',
      currentCoordinates,
      previousMmvCoordinates,
      mmvGroups,
      sclCoordinates,
      sclCount: sclCoordinates.length
    };
  }

  if (isQmlReady(obj)) {
    const previousMmvCoordinates = getAllMmvCoordinatesUpToQmlReady(sessionKey, tm);
    const mmvGroups = getMmvGroupsUpToQmlReady(sessionKey, tm);
    const sclCoordinates = getAllSclCoordinatesUpToQmlReady(sessionKey, tm);
    const ua = (obj?.ua && typeof obj.ua === 'object' && obj.ua.v != null) ? String(obj.ua.v) : '';

    return {
      sessA,
      tm,
      ts,
      ip,
      ua,
      type: 'qml_ready',
      currentCoordinates: [],
      previousMmvCoordinates,
      mmvGroups,
      sclCoordinates,
      sclCount: sclCoordinates.length
    };
  }
  
  return null;
}

// Функция для обработки промежуточных данных и вычисления статистик
function processIntermediateData(intermediateRecord) {
  const {
    currentCoordinates,
    previousMmvCoordinates,
    mmvGroups,
    sclCoordinates,
    ...recordData
  } = intermediateRecord;
  
  // Объединяем все координаты
  const resultCoordinates = [...previousMmvCoordinates, ...sclCoordinates, ...currentCoordinates];
  const resolvedMmvGroups = Array.isArray(mmvGroups) && mmvGroups.length > 0
    ? mmvGroups
    : (currentCoordinates?.length ? [currentCoordinates] : []);
  
  // Вычисляем статистики
  const statistics = buildMouseMoveStatistics(resultCoordinates, resolvedMmvGroups);
  
  if (statistics) {
    // Вычисляем данные скроллов
    let scrollData = [];
    if (sclCoordinates.length > 0) {
      for (const sclRecord of sclHistoryBySession[getSessionKey(recordData.sessA)] || []) {
        if (sclRecord.tm > (recordData.tm - 10000) && sclRecord.tm < recordData.tm) {
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
  // const sessB = obj?.sess?.b; // больше не используем
  const tm = obj?.prm?.tm;
  const ts = obj?.ts;
  const ip = obj?.ip;
 
  if (!sessA || !tm) {
    return null;
  }
  
  const sessionKey = getSessionKey(sessA);
  updateSessionMarkers(obj);
  const hasMmvRecord = hasMmv(obj);
  const hasClkRecord = hasClk(obj);
  const hasSclRecord = hasScl(obj);
  
  // Сохраняем scl запись в историю
  if (hasSclRecord) {
    const sclL = obj.prm.data.e.scl.l;
    const sclRecord = {
      sessA,
      tm,
      coordinates: parseSclCoordinates(sclL, tm),
      type: 'scl'
    };
    
    // Сохраняем в историю скроллов
    if (!sclHistoryBySession[sessionKey]) {
      sclHistoryBySession[sessionKey] = [];
    }
    sclHistoryBySession[sessionKey].push(sclRecord);
    
    // Ограничиваем размер истории до MAX_MMV_RECORDS
    if (sclHistoryBySession[sessionKey].length > MAX_MMV_RECORDS) {
      sclHistoryBySession[sessionKey] = sclHistoryBySession[sessionKey].slice(-MAX_MMV_RECORDS);
    }
  }
  
  // Сохраняем mmv запись в историю
  if (hasMmvRecord) {
    const mmvL = obj.prm.data.e.mmv.l;
    const mmvRecord = {
      sessA,
      tm,
      coordinates: parseMmvCoordinates(mmvL, tm),
      type: hasClkRecord ? 'mmv_clk' : 'mmv' // Добавляем тип записи
    };
    
    // Сохраняем в историю
    if (!mmvHistoryBySession[sessionKey]) {
      mmvHistoryBySession[sessionKey] = [];
    }
    mmvHistoryBySession[sessionKey].push(mmvRecord);
    
    // Ограничиваем размер истории до MAX_MMV_RECORDS
    if (mmvHistoryBySession[sessionKey].length > MAX_MMV_RECORDS) {
      mmvHistoryBySession[sessionKey] = mmvHistoryBySession[sessionKey].slice(-MAX_MMV_RECORDS);
    }
    
    // Обновляем последнюю запись
    lastMmvBySession[sessionKey] = mmvRecord;
  }
  
  // Обрабатываем записи с clk
  if (hasClkRecord && hasMmvRecord) {
    const currentCoordinates = parseMmvCoordinates(obj.prm.data.e.mmv.l, tm);
    
    // Получаем все координаты mmv от начала сессии до ближайшей предыдущей записи mmv+clk
    const previousMmvCoordinates = getAllMmvCoordinatesUpToPreviousClk(sessionKey, tm);
    
    // Получаем все координаты scl между предыдущим clk и текущим
    const sclCoordinates = getAllSclCoordinatesBetweenClks(sessionKey, tm);
    
    // Объединяем все координаты: сначала предыдущие mmv, затем scl, затем текущие mmv
    const resultCoordinates = [...previousMmvCoordinates, ...sclCoordinates, ...currentCoordinates];
    const mmvGroups = [...getMmvGroupsUpToPreviousClk(sessionKey, tm), currentCoordinates];
    
    // Вычисляем статистики для всех объединенных координат
    const statistics = buildMouseMoveStatistics(resultCoordinates, mmvGroups);
    
    if (statistics) {
      sclAggregated += sclCoordinates.length;
      
      // Вычисляем данные скроллов для preprocessor.py
      let scrollData = [];
      if (sclCoordinates.length > 0) {
        // Группируем координаты скроллов по записям (каждая запись scl - это отдельный скролл)
        const sclHistory = sclHistoryBySession[sessionKey] || [];
        const mmvHistory = mmvHistoryBySession[sessionKey] || [];
        
        // Находим время предыдущей mmv+clk записи
        let previousClkTime = null;
        for (let i = mmvHistory.length - 1; i >= 0; i--) {
          const mmvRecord = mmvHistory[i];
          if (mmvRecord.type === 'mmv_clk' && mmvRecord.tm < tm) {
            previousClkTime = mmvRecord.tm;
            break;
          }
        }
        
        // Собираем данные скроллов между предыдущим clk и текущим
        for (const sclRecord of sclHistory) {
          if (sclRecord.tm > (previousClkTime || 0) && sclRecord.tm < tm) {
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
        tm,
        ts,
        ip,
        type: 'mmv_clk',
        statistics: statistics,
        sclCount: sclCoordinates.length,
        scrollData: scrollData // Добавляем реальные данные скроллов
      };
    }
  }

  if (isQmlReady(obj)) {
    const previousMmvCoordinates = getAllMmvCoordinatesUpToQmlReady(sessionKey, tm);
    const mmvGroups = getMmvGroupsUpToQmlReady(sessionKey, tm);
    const sclCoordinates = getAllSclCoordinatesUpToQmlReady(sessionKey, tm);
    const resultCoordinates = [...previousMmvCoordinates, ...sclCoordinates];
    const statistics = buildMouseMoveStatistics(resultCoordinates, mmvGroups);
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

      return {
        sessA,
        tm,
        ts,
        ip,
        ua: (obj?.ua && typeof obj.ua === 'object' && obj.ua.v != null) ? String(obj.ua.v) : '',
        type: 'qml_ready',
        statistics,
        sclCount: sclCoordinates.length,
        scrollData
      };
    }
  }
  
  return null;
}

function buildSessionSummaryAtTime(sessionKey, sessA, cutoffTm) {
  const previousMmvCoordinates = getAllMmvCoordinatesUpToQmlReady(sessionKey, cutoffTm);
  const sclCoordinates = getAllSclCoordinatesUpToQmlReady(sessionKey, cutoffTm);
  const resultCoordinates = [...previousMmvCoordinates, ...sclCoordinates];
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
    console.log('Режим: полный пересбор статистики из source NDJSON.');
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

  let syntheticEmitted = 0;
  for (const sessionKey of touchedSessionKeys) {
    const lastEventTm = lastEventTmBySession[sessionKey];
    if (!lastEventTm) {
      continue;
    }

    const sessA = sessionKey;
    const hasQmlReady = hasQmlReadyBySession[sessionKey] === true;
    const startsWithQml = firstQmlTmBySession[sessionKey] !== undefined;
    const cutoffTm = !hasQmlReady && startsWithQml
      ? Number(lastEventTm) + 1
      : lastEventTm;
    const summary = buildSessionSummaryAtTime(sessionKey, sessA, cutoffTm);
    if (!summary) {
      continue;
    }

    if (!hasQmlReady && startsWithQml) {
      summary.syntheticReason = 'first_qml_session_end';
      summary.firstQmlTm = firstQmlTmBySession[sessionKey];
    }

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
  console.log('Новых/обновлённых mmv+clk записей:', newResults.length);
  console.log('Синтетических записей (флаш сессий без нового qml.ready):', syntheticEmitted);
  console.log('Изменённых session_id:', changedSessions.size);
  console.log('Агрегировано скроллов:', sclAggregated);
  console.log('Результат сохранен в:', outputFile);
}

// Этап 1: Извлечение и сохранение промежуточных данных
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
  let syntheticExtracted = 0;
  const touchedSessionKeys = new Set();

  function writeIntermediateRecord(record) {
    intermediateOutput.write((isFirst ? '' : ',\n') + JSON.stringify(record));
    isFirst = false;
    extracted++;
  }

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

    const sessionKey = updateSessionMarkers(obj);
    if (sessionKey) {
      touchedSessionKeys.add(sessionKey);
    }

    const result = extractIntermediateData(obj);
    if (result) {
      writeIntermediateRecord(result);
    }
  }

  // Flush touched-сессии, у которых после последней активности не было qml.ready.
  for (const sessionKey of touchedSessionKeys) {
    const lastEventTm = lastEventTmBySession[sessionKey];
    if (!lastEventTm) {
      continue;
    }

    const lastQmlReadyTm = lastQmlReadyTmBySession[sessionKey];
    if (lastQmlReadyTm !== undefined && Number(lastQmlReadyTm) >= Number(lastEventTm)) {
      continue;
    }

    const startsWithQml = firstQmlTmBySession[sessionKey] !== undefined;
    const cutoffTm = startsWithQml ? Number(lastEventTm) + 1 : lastEventTm;
    const summary = buildIntermediateSessionSummaryAtTime(sessionKey, sessionKey, cutoffTm);
    if (startsWithQml) {
      summary.syntheticReason = 'first_qml_session_end';
      summary.firstQmlTm = firstQmlTmBySession[sessionKey];
    }

    writeIntermediateRecord(summary);
    syntheticExtracted++;
  }

  intermediateOutput.write('\n]\n');
  intermediateOutput.end();
  
  console.log('Извлечение завершено!');
  console.log('Успешно распарсено объектов:', parsed);
  console.log('Извлечено записей с mmv+clk/qml_ready:', extracted);
  console.log('Синтетических qml_ready записей:', syntheticExtracted);
  console.log('Промежуточные данные сохранены в:', intermediateFile);
  
  return extracted;
}

// Этап 2: Обработка промежуточных данных и вычисление статистик
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
    
    // Пропускаем первую и последнюю строки (открывающая и закрывающая скобки JSON массива)
    if (lineNumber === 1 && line.trim() === '[') continue;
    if (line.trim() === ']') continue;
    
    // Убираем запятую в конце строки, если она есть
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
      
      // Выводим прогресс каждые 1000 записей
      // if (processed % 1000 === 0) {
      //   console.log(`Обработано записей: ${processed}`);
      // }
    }
  }

  output.write('\n]\n');
  output.end();
  
  console.log('Обработка завершена!');
  console.log('Обработано записей с mmv+clk:', processed);
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

// Экспорт класса для тестирования
module.exports = { MouseMoveStatistics };

processFile().catch(console.error);
