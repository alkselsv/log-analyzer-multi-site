/**
 * Скрипт для разделения записей по типу устройства (мобильные/не мобильные)
 * Логика: находит записи с act="qml.ready", определяет тип устройства по mb,
 * затем все записи с тем же sess.a сохраняются в соответствующий файл
 * mb=0 - не мобильные, mb!=0 - мобильные
 * если mb по сессии не найден, наличие touch-событий (tmv/tst/ted/tbb/tcl/tbf)
 * относит сессию к мобильным
 *
 * Входной файл: JSON_PATH (по умолчанию /var/www/mlog/1_466.json)
 *
 * Использование:
 *   node split_records.js
 *
 * Переменные окружения:
 *   JSON_PATH - путь к входному файлу (по умолчанию: /var/www/mlog/1_466.json)
 *   OUTPUT_DESKTOP - путь к файлу для не мобильных устройств (по умолчанию: merged.cloud.desktop.ndjson)
 *   OUTPUT_MOBILE - путь к файлу для мобильных устройств (по умолчанию: merged.cloud.mobile.ndjson)
 */
require('./js/load-env').loadEnv();
const fs = require('fs');

const inputFile = process.env.JSON_PATH || '/var/www/mlog/1_466.json';
const outputDesktop = process.env.OUTPUT_DESKTOP || 'merged.cloud.desktop.ndjson';
const outputMobile = process.env.OUTPUT_MOBILE || 'merged.cloud.mobile.ndjson';
const checkpointFile = process.env.SPLIT_CHECKPOINT_FILE || 'split_records_checkpoint.json';
const forceFullRebuild = process.env.FORCE_FULL_REBUILD === '1';

let totalRecords = 0;
let processedRecords = 0;
let desktopRecords = 0;
let mobileRecords = 0;
let errorCount = 0;
let qmlReadyRecords = 0;
let unmappedRecords = 0;

const sessionMapping = new Map();
const sessionHasMb = new Map();

const TOUCH_EVENT_KEYS = new Set(['tmv', 'tst', 'ted', 'tbb', 'tcl', 'tbf']);

function getFileIdentity(stats) {
  return {
    ino: stats.ino || null,
    size: stats.size,
    mtimeMs: stats.mtimeMs,
  };
}

function loadCheckpoint() {
  if (!fs.existsSync(checkpointFile)) {
    return null;
  }

  try {
    return JSON.parse(fs.readFileSync(checkpointFile, 'utf8'));
  } catch (error) {
    console.warn(`Предупреждение: не удалось прочитать checkpoint '${checkpointFile}', будет полный пересбор.`);
    return null;
  }
}

function saveCheckpoint(checkpoint) {
  const tmpFile = `${checkpointFile}.tmp`;
  fs.writeFileSync(tmpFile, JSON.stringify(checkpoint, null, 2));
  fs.renameSync(tmpFile, checkpointFile);
}

function shouldRebuild(checkpoint, stats) {
  const decision = getRebuildDecision(checkpoint, stats);
  return decision.rebuild;
}

function getRebuildDecision(checkpoint, stats) {
  if (forceFullRebuild) {
    return { rebuild: true, reason: 'FORCE_FULL_REBUILD=1' };
  }

  if (!checkpoint || !checkpoint.source) {
    return { rebuild: true, reason: 'checkpoint отсутствует или повреждён' };
  }

  const previous = checkpoint.source;
  const offset = Number(checkpoint.offset || 0);

  // Критично: входной файл стал меньше уже обработанного offset (обрезка/ротация).
  if (stats.size < offset) {
    return {
      rebuild: true,
      reason: `размер входа (${stats.size}) меньше checkpoint.offset (${offset})`,
    };
  }

  // Критично: mtime откатился назад относительно предыдущего запуска
  // (часто признак подмены/отката source-файла).
  if (
    typeof previous.mtimeMs === 'number' &&
    typeof stats.mtimeMs === 'number' &&
    stats.mtimeMs < previous.mtimeMs
  ) {
    return {
      rebuild: true,
      reason: `mtime входа откатился назад (${stats.mtimeMs} < ${previous.mtimeMs})`,
    };
  }

  // Некритично: inode может меняться при ротации/переоткрытии файла.
  // Если размер и mtime не противоречат checkpoint, продолжаем инкрементально.
  if (previous.ino !== null && stats.ino !== undefined && previous.ino !== stats.ino) {
    return {
      rebuild: false,
      reason: `inode изменился (${previous.ino} -> ${stats.ino}), продолжаем инкрементально`,
    };
  }

  return { rebuild: false, reason: `checkpoint валиден, offset=${offset}` };
}

function restoreSessionMapping(checkpoint) {
  sessionMapping.clear();
  sessionHasMb.clear();
  const mapping = checkpoint && checkpoint.sessionMapping ? checkpoint.sessionMapping : {};
  for (const [sessionId, isMobile] of Object.entries(mapping)) {
    sessionMapping.set(sessionId, isMobile === true);
  }

  const hasMbMapping = checkpoint && checkpoint.sessionHasMb ? checkpoint.sessionHasMb : {};
  for (const [sessionId, hasMb] of Object.entries(hasMbMapping)) {
    sessionHasMb.set(sessionId, hasMb === true);
  }
}

function dumpSessionMapping() {
  return Object.fromEntries(sessionMapping.entries());
}

function dumpSessionHasMb() {
  return Object.fromEntries(sessionHasMb.entries());
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

function isMobileDevice(obj) {
  if (obj === null || typeof obj !== 'object') {
    return null;
  }

  if (Array.isArray(obj)) {
    for (const item of obj) {
      const result = isMobileDevice(item);
      if (result !== null) {
        return result;
      }
    }
    return null;
  }

  if ('mb' in obj) {
    return obj.mb !== 0;
  }

  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key) && typeof obj[key] === 'object') {
      const result = isMobileDevice(obj[key]);
      if (result !== null) {
        return result;
      }
    }
  }

  return null;
}

function hasMb(obj) {
  if (obj === null || typeof obj !== 'object') {
    return false;
  }

  if (Array.isArray(obj)) {
    return obj.some((item) => hasMb(item));
  }

  if ('mb' in obj) {
    return true;
  }

  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key) && typeof obj[key] === 'object') {
      if (hasMb(obj[key])) {
        return true;
      }
    }
  }

  return false;
}

function hasTouchEvent(obj) {
  const events = obj?.prm?.data?.e;
  if (!events || typeof events !== 'object') {
    return false;
  }

  return Object.keys(events).some((key) => TOUCH_EVENT_KEYS.has(key));
}

function getSessionId(obj) {
  if (obj && obj.sess && typeof obj.sess === 'object' && obj.sess.a) {
    return obj.sess.a;
  }
  return null;
}

function updateMappingFromLines(lines) {
  console.log('=== Построение/обновление маппинга сессий для нового хвоста ===');

  const touchSessions = new Set();

  for (const entry of lines) {
    const { obj } = entry;
    if (!obj) {
      continue;
    }

    const sessionId = getSessionId(obj);
    if (!sessionId) {
      continue;
    }

    if (hasMb(obj)) {
      sessionHasMb.set(sessionId, true);
    }

    if (hasTouchEvent(obj)) {
      touchSessions.add(sessionId);
    }
  }

  for (const entry of lines) {
    const { obj } = entry;
    if (!obj) {
      continue;
    }

    const sessionId = getSessionId(obj);
    if (!sessionId) {
      continue;
    }

    if (obj.act === 'qml.ready') {
      const isMobile = isMobileDevice(obj);
      if (isMobile !== null) {
        sessionMapping.set(sessionId, isMobile === true);
      }
      qmlReadyRecords++;
    }
  }

  for (const sessionId of touchSessions) {
    if (!sessionHasMb.get(sessionId)) {
      sessionMapping.set(sessionId, true);
    }
  }

  console.log(`Найдено новых записей qml.ready: ${qmlReadyRecords}`);
  console.log(`Уникальных сессий в маппинге: ${sessionMapping.size}`);
}

async function distributeRecords(lines, rewriteOutputs) {
  console.log('\n=== Распределение записей ===');

  const desktopStream = fs.createWriteStream(outputDesktop, { flags: rewriteOutputs ? 'w' : 'a' });
  const mobileStream = fs.createWriteStream(outputMobile, { flags: rewriteOutputs ? 'w' : 'a' });

  for (const entry of lines) {
    const { line, obj } = entry;
    totalRecords++;

    if (!obj) {
      console.error(`Ошибка парсинга строки ${totalRecords}:`, 'invalid JSON');
      errorCount++;
      continue;
    }
    processedRecords++;

    const sessionId = getSessionId(obj);
    let isMobile = null;

    if (sessionId && sessionMapping.has(sessionId)) {
      isMobile = sessionMapping.get(sessionId);
    } else {
      const mobileFromRecord = isMobileDevice(obj);
      if (mobileFromRecord !== null) {
        isMobile = mobileFromRecord;
      } else if (sessionId && !sessionHasMb.get(sessionId) && hasTouchEvent(obj)) {
        isMobile = true;
        sessionMapping.set(sessionId, true);
      } else {
        isMobile = false;
      }
      if (sessionId) {
        unmappedRecords++;
      }
    }

    if (isMobile) {
      mobileStream.write(line + '\n');
      mobileRecords++;
    } else {
      desktopStream.write(line + '\n');
      desktopRecords++;
    }

    if (totalRecords % 10000 === 0) {
      console.log(`Обработано записей: ${totalRecords}, не мобильные: ${desktopRecords}, мобильные: ${mobileRecords}`);
    }
  }

  desktopStream.end();
  mobileStream.end();

  await Promise.all([
    new Promise((resolve) => desktopStream.on('finish', resolve)),
    new Promise((resolve) => mobileStream.on('finish', resolve)),
  ]);
}

function parseLines(lines) {
  const parsedEntries = [];

  for (const line of lines) {
    if (!line.trim()) {
      continue;
    }

    try {
      const obj = JSON.parse(line);
      parsedEntries.push({ line, obj });
    } catch (error) {
      parsedEntries.push({ line, obj: null });
    }
  }

  return parsedEntries;
}

async function processFile() {
  console.log('=== Обработка файла ===');
  console.log(`Входной файл: ${inputFile}`);
  console.log(`Файл для не мобильных устройств: ${outputDesktop}`);
  console.log(`Файл для мобильных устройств: ${outputMobile}`);
  console.log(`Checkpoint: ${checkpointFile}`);

  if (!fs.existsSync(inputFile)) {
    console.error(`Ошибка: файл ${inputFile} не найден!`);
    process.exit(1);
  }

  const stats = fs.statSync(inputFile);
  const checkpoint = loadCheckpoint();
  const decision = getRebuildDecision(checkpoint, stats);
  const rebuild = decision.rebuild;
  const startOffset = rebuild || !checkpoint ? 0 : (checkpoint.offset || 0);

  if (rebuild) {
    console.log(`Режим: полный пересбор выходных файлов и checkpoint (${decision.reason}).`);
  } else {
    console.log(`Режим: инкрементальный запуск с offset=${startOffset} (${decision.reason}).`);
  }

  restoreSessionMapping(rebuild ? null : checkpoint);

  const { lines, nextOffset, hasPartialTail } = await readNewCompleteLines(startOffset);
  if (hasPartialTail) {
    console.log('Обнаружена незавершённая последняя строка, она будет обработана на следующем запуске.');
  }

  if (lines.length === 0) {
    console.log('Новых завершённых строк не найдено.');
    saveCheckpoint({
      source: getFileIdentity(stats),
      offset: nextOffset,
      sessionMapping: dumpSessionMapping(),
      sessionHasMb: dumpSessionHasMb(),
    });
    return;
  }

  const parsedEntries = parseLines(lines);
  updateMappingFromLines(parsedEntries);
  await distributeRecords(parsedEntries, rebuild);

  saveCheckpoint({
    source: getFileIdentity(stats),
    offset: nextOffset,
    sessionMapping: dumpSessionMapping(),
    sessionHasMb: dumpSessionHasMb(),
  });

  console.log('\n=== Обработка завершена ===');
  console.log(`Всего записей: ${totalRecords}`);
  console.log(`Успешно обработано: ${processedRecords}`);
  console.log(`Найдено записей qml.ready: ${qmlReadyRecords}`);
  console.log(`Уникальных сессий в маппинге: ${sessionMapping.size}`);
  console.log(`Записей без маппинга (использована старая логика): ${unmappedRecords}`);
  console.log(`Не мобильные устройства (mb=0 или отсутствует): ${desktopRecords}`);
  console.log(`Мобильные устройства (mb!=0): ${mobileRecords}`);
  console.log(`Ошибок парсинга: ${errorCount}`);
  console.log(`Не мобильные устройства сохранены в: ${outputDesktop}`);
  console.log(`Мобильные устройства сохранены в: ${outputMobile}`);
}

processFile().catch((error) => {
  console.error('Критическая ошибка:', error);
  process.exit(1);
});
