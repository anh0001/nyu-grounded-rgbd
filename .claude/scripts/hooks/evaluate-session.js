#!/usr/bin/env node
/**
 * Repo-local stop hook.
 *
 * The previous version depended on missing shared ECC libraries. Keep this
 * version self-contained: it lightly inspects the transcript when available and
 * emits a note for longer sessions, but never blocks shutdown.
 */

'use strict';

const fs = require('fs');

const MAX_STDIN = 1024 * 1024;
const MIN_USER_MESSAGES = 10;

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

function countUserMessages(transcriptPath) {
  if (!transcriptPath || !fs.existsSync(transcriptPath)) {
    return 0;
  }
  const content = fs.readFileSync(transcriptPath, 'utf8');
  const matches = content.match(/"type"\s*:\s*"user"/g);
  return matches ? matches.length : 0;
}

function main(raw) {
  try {
    const input = JSON.parse(raw || '{}');
    const transcriptPath = input.transcript_path || process.env.CLAUDE_TRANSCRIPT_PATH;
    const messageCount = countUserMessages(transcriptPath);
    if (messageCount >= MIN_USER_MESSAGES) {
      process.stderr.write(
        `[SessionReview] Long session detected (${messageCount} user turns). Consider extracting reusable workflow notes.\n`
      );
    }
  } catch {
    // Best-effort only.
  }
  process.exit(0);
}

readStdin(main);
