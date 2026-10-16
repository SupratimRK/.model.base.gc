import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import ort from 'onnxruntime-node';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');

const MODEL_PATH = path.join(ROOT_DIR, 'models', 'model_int8.onnx');
const VOCAB_PATH = path.join(ROOT_DIR, 'models', 'tokenizer', 'vocab.txt');
const CONFIG_PATH = path.join(ROOT_DIR, 'config', 'model_config.json');

const TITLE_PREFIXES = new Set([
  'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sri', 'shri', 'smt', 'master',
  'md', 'mohd', 'sk', 'sekh', 'sheikh', 'syed', 'kazi', 'adv', 'er', 'ca'
]);

export class CompactGenderClassifier {
  constructor() {
    this.session = null;
    this.vocab = new Map();
    this.idToVocab = new Map();
    this.maxLength = 20;
    this.isReady = false;

    // Special token IDs
    this.padId = 0;
    this.unkId = 1;
    this.clsId = 2;
    this.sepId = 3;
    this.maskId = 4;
  }

  async init(modelPath = MODEL_PATH) {
    if (this.isReady) return;

    // 1. Load Vocab
    if (!fs.existsSync(VOCAB_PATH)) {
      throw new Error(`Vocabulary file not found at: ${VOCAB_PATH}`);
    }
    const lines = fs.readFileSync(VOCAB_PATH, 'utf-8').split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const token = lines[i];
      if (token.length > 0) {
        this.vocab.set(token, i);
        this.idToVocab.set(i, token);
      }
    }

    // 2. Load Config if available
    if (fs.existsSync(CONFIG_PATH)) {
      try {
        const cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
        this.maxLength = cfg.max_length || 20;
      } catch (e) {}
    }

    // 3. Initialize ONNX Session with single-thread optimization
    const sessionOptions = {
      intraOpNumThreads: 1,
      interOpNumThreads: 1,
      executionMode: 'sequential',
      graphOptimizationLevel: 'all'
    };

    this.session = await ort.InferenceSession.create(modelPath, sessionOptions);
    this.isReady = true;
  }

  /**
   * Parses input string into given name and gender-neutral surnames.
   */
  parseNameTokens(text) {
    if (!text || typeof text !== 'string') {
      return { givenName: '', surnames: [] };
    }

    const words = text.trim().split(/\s+/).filter(w => w.length > 0);
    if (words.length === 0) {
      return { givenName: '', surnames: [] };
    }

    // Filter out common honorific titles (e.g., Dr., Mr., Smt.)
    let idx = 0;
    while (idx < words.length) {
      const cleaned = words[idx].toLowerCase().replace(/[^a-z]/g, '');
      if (TITLE_PREFIXES.has(cleaned)) {
        idx++;
      } else {
        break;
      }
    }

    const remaining = words.slice(idx);
    if (remaining.length === 0) {
      return { givenName: words[0], surnames: [] };
    }

    return {
      givenName: remaining[0],
      surnames: remaining.slice(1)
    };
  }

  tokenize(text) {
    if (!text || typeof text !== 'string') return [this.clsId, this.sepId];

    // Pre-tokenize: Whitespace, Punctuation, and Individual Digits
    const rawTokens = text.match(/[\p{L}]+|\d|[^\s\p{L}\d]/gu) || [];
    const tokenIds = [this.clsId];

    for (const chunk of rawTokens) {
      if (tokenIds.length >= this.maxLength - 1) break;

      // If chunk is in vocab directly
      if (this.vocab.has(chunk)) {
        tokenIds.push(this.vocab.get(chunk));
        continue;
      }

      // WordPiece greedy subword matching
      let start = 0;
      let isBad = false;
      const subTokens = [];

      while (start < chunk.length) {
        let end = chunk.length;
        let curSubId = null;

        while (start < end) {
          const sub = (start === 0) ? chunk.slice(start, end) : '##' + chunk.slice(start, end);
          if (this.vocab.has(sub)) {
            curSubId = this.vocab.get(sub);
            break;
          }
          end--;
        }

        if (curSubId === null) {
          isBad = true;
          break;
        }

        subTokens.push(curSubId);
        start = end;
      }

      if (isBad) {
        tokenIds.push(this.unkId);
      } else {
        for (const tid of subTokens) {
          if (tokenIds.length >= this.maxLength - 1) break;
          tokenIds.push(tid);
        }
      }
    }

    tokenIds.push(this.sepId);
    return tokenIds;
  }

  softmax(logits) {
    const maxVal = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - maxVal));
    const sum = exps.reduce((acc, v) => acc + v, 0);
    return exps.map(v => v / (sum || 1e-9));
  }

  async predict(rawText) {
    if (!this.isReady) {
      await this.init();
    }

    const { givenName, surnames } = this.parseNameTokens(rawText);
    const queryName = givenName || rawText;

    const tokenIds = this.tokenize(queryName);
    const paddedIds = new Int32Array(this.maxLength).fill(this.padId);
    const attentionMask = new Int32Array(this.maxLength).fill(0);

    for (let i = 0; i < tokenIds.length && i < this.maxLength; i++) {
      paddedIds[i] = tokenIds[i];
      attentionMask[i] = 1;
    }

    const inputTensor = new ort.Tensor('int32', paddedIds, [1, this.maxLength]);
    const maskTensor = new ort.Tensor('int32', attentionMask, [1, this.maxLength]);

    const feeds = {
      input_ids: inputTensor,
      attention_mask: maskTensor
    };

    const results = await this.session.run(feeds);
    const logits = Array.from(results.logits.data);
    const probs = this.softmax(logits);

    // 0: MALE, 1: FEMALE
    const isFemale = probs[1] >= probs[0];
    const classification = isFemale ? 'FEMALE' : 'MALE';
    const confidence = isFemale ? probs[1] : probs[0];

    return {
      text: rawText,
      gender: classification,
      confidence: confidence,
      given_name: givenName,
      surnames: surnames,
      raw_probs: {
        male: probs[0],
        female: probs[1]
      }
    };
  }

  async predictBatch(texts) {
    if (!this.isReady) {
      await this.init();
    }

    const results = [];
    for (const t of texts) {
      results.push(await this.predict(t));
    }
    return results;
  }
}
