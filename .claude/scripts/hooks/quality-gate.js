#!/usr/bin/env node
/**
 * Repo-local quality gate hook.
 *
 * This repo is Python-first, so keep the hook self-contained and lightweight:
 * - For edited Python files, run Ruff if available.
 * - Never block the Claude session on missing tooling.
 * - Always pass the original hook payload through stdout.
 */

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const MAX_STDIN = 1024 * 1024;

function readStdin(callback) {
  let raw = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => {
    if (raw.length >= MAX_STDIN) {
      return;
    }
    raw += chunk.slice(0, MAX_STDIN - raw.length);
  });
  process.stdin.on('end', () => callback(raw));
}

function log(message) {
  process.stderr.write(`[QualityGate] ${message}\n`);
}

function findRuff() {
  const candidates = [
    path.join(os.homedir(), 'miniconda3', 'envs', 'mobile-sam', 'bin', 'ruff'),
    'ruff',
  ];
  for (const candidate of candidates) {
    const probe = spawnSync(candidate, ['--version'], {
      encoding: 'utf8',
      timeout: 5000,
    });
    if (!probe.error && probe.status === 0) {
      return candidate;
    }
  }
  return null;
}

function run(command, args) {
  return spawnSync(command, args, {
    cwd: process.cwd(),
    encoding: 'utf8',
    env: process.env,
    timeout: 15000,
  });
}

function checkPythonFile(filePath) {
  if (!filePath || !fs.existsSync(filePath) || path.extname(filePath) !== '.py') {
    return;
  }

  const ruff = findRuff();
  if (!ruff) {
    return;
  }

  const fix = String(process.env.ECC_QUALITY_GATE_FIX || '').toLowerCase() === 'true';
  const commands = fix
    ? [
        ['check', '--fix', filePath],
        ['format', filePath],
      ]
    : [
        ['check', filePath],
        ['format', '--check', filePath],
      ];

  for (const args of commands) {
    const result = run(ruff, args);
    if (result.error) {
      log(`${args.join(' ')} failed: ${result.error.message}`);
      return;
    }
    if (result.status !== 0) {
      const output = `${result.stdout || ''}${result.stderr || ''}`.trim();
      log(`${args.join(' ')} reported issues for ${filePath}`);
      if (output) {
        process.stderr.write(`${output}\n`);
      }
      return;
    }
  }
}

function main(raw) {
  try {
    const input = JSON.parse(raw);
    const filePath = path.resolve(String(input.tool_input?.file_path || ''));
    checkPythonFile(filePath);
  } catch {
    // Hook input is best-effort only.
  }
  process.stdout.write(raw);
}

readStdin(main);
