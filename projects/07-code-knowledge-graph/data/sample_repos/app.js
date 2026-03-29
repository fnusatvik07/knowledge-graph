/**
 * Sample JavaScript application demonstrating functions, classes, and imports.
 * Used as sample data for multi-language code knowledge graph construction.
 */

import { readFileSync, writeFileSync } from 'fs';
import path from 'path';

// ── Configuration ───────────────────────────────────────────────────────

const DEFAULT_CONFIG = {
    outputDir: './output',
    maxRetries: 3,
    timeout: 5000,
};

// ── Logger ──────────────────────────────────────────────────────────────

class Logger {
    constructor(prefix = 'App') {
        this.prefix = prefix;
        this.entries = [];
    }

    info(message) {
        const entry = this._format('INFO', message);
        this.entries.push(entry);
        console.log(entry);
    }

    error(message) {
        const entry = this._format('ERROR', message);
        this.entries.push(entry);
        console.error(entry);
    }

    _format(level, message) {
        return `[${this.prefix}] ${level}: ${message}`;
    }

    getEntries() {
        return [...this.entries];
    }
}

// ── Data Processing ─────────────────────────────────────────────────────

class DataProcessor {
    constructor(config = DEFAULT_CONFIG) {
        this.config = config;
        this.logger = new Logger('DataProcessor');
    }

    loadData(filePath) {
        try {
            const fullPath = path.resolve(filePath);
            const raw = readFileSync(fullPath, 'utf-8');
            this.logger.info(`Loaded data from ${fullPath}`);
            return JSON.parse(raw);
        } catch (err) {
            this.logger.error(`Failed to load data: ${err.message}`);
            return null;
        }
    }

    transformData(records) {
        this.logger.info(`Transforming ${records.length} records`);
        return records.map(record => this.processRecord(record));
    }

    processRecord(record) {
        const validated = validateRecord(record);
        return { ...validated, processedAt: new Date().toISOString() };
    }

    saveResults(data, filename) {
        const outputPath = path.join(this.config.outputDir, filename);
        const json = JSON.stringify(data, null, 2);
        writeFileSync(outputPath, json);
        this.logger.info(`Saved results to ${outputPath}`);
    }
}

// ── Utility Functions ───────────────────────────────────────────────────

function validateRecord(record) {
    if (!record || typeof record !== 'object') {
        throw new Error('Invalid record: must be an object');
    }
    if (!record.id) {
        throw new Error('Invalid record: missing id field');
    }
    return record;
}

function computeStatistics(records) {
    if (!records || records.length === 0) {
        return { count: 0, mean: 0, max: 0, min: 0 };
    }
    const values = records.map(r => r.value || 0);
    const sum = values.reduce((a, b) => a + b, 0);
    return {
        count: values.length,
        mean: sum / values.length,
        max: Math.max(...values),
        min: Math.min(...values),
    };
}

function formatOutput(data, format = 'json') {
    if (format === 'json') {
        return JSON.stringify(data, null, 2);
    }
    if (format === 'csv') {
        const keys = Object.keys(data[0] || {});
        const header = keys.join(',');
        const rows = data.map(row => keys.map(k => row[k]).join(','));
        return [header, ...rows].join('\n');
    }
    throw new Error(`Unsupported format: ${format}`);
}

// ── Main Pipeline ───────────────────────────────────────────────────────

function runPipeline(inputFile, outputFile) {
    const processor = new DataProcessor();
    const rawData = processor.loadData(inputFile);
    if (!rawData) return;

    const transformed = processor.transformData(rawData);
    const stats = computeStatistics(transformed);
    processor.logger.info(`Statistics: ${formatOutput(stats)}`);
    processor.saveResults(transformed, outputFile);
}

export { Logger, DataProcessor, validateRecord, computeStatistics, formatOutput, runPipeline };
