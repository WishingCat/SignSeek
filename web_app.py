"""SignSeek local web UI.

Run:
  python3 web_app.py --open

The server intentionally uses only the Python standard library. It receives
compressed image data from the browser, writes temporary frame files, and
reuses query.run_query() for the actual recognition pipeline.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import tempfile
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

import config
import query


HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 5


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>见手知意 SignSeek</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #2a1a10;
      --muted: #806c5a;
      --hairline: rgba(126, 78, 27, .18);
      --glass: rgba(255, 250, 238, .66);
      --glass-strong: rgba(255, 246, 224, .82);
      --blue: #f59e0b;
      --teal: #2f7d64;
      --rose: #be123c;
      --amber: #f59e0b;
      --gold: #fbbf24;
      --shadow: 0 24px 70px rgba(120, 72, 20, .16);
      --radius: 24px;
      --ease: cubic-bezier(.2, .8, .2, 1);
      --ease-pop: cubic-bezier(.16, 1, .3, 1);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
      background:
        linear-gradient(135deg, rgba(255, 252, 244, .98), rgba(255, 238, 199, .84) 46%, rgba(255, 250, 238, .96)),
        linear-gradient(45deg, rgba(245, 158, 11, .18), rgba(251, 191, 36, .14), rgba(47, 125, 100, .07));
      background-attachment: fixed;
      animation: pageFade .62s var(--ease-pop) both;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,.42) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.32) 1px, transparent 1px);
      background-size: 72px 72px;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,.72), transparent);
      animation: gridDrift 24s linear infinite;
    }

    button, input, select {
      font: inherit;
      letter-spacing: 0;
    }

    button {
      border: 0;
      cursor: pointer;
      -webkit-tap-highlight-color: transparent;
    }

    button:disabled {
      cursor: not-allowed;
      opacity: .45;
    }

    .shell {
      width: min(1180px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 26px 0 44px;
      animation: shellRise .7s var(--ease-pop) both;
    }

    .topbar {
      display: grid;
      grid-template-columns: 1fr;
      align-items: center;
      gap: 20px;
      margin-bottom: 22px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .mark {
      width: 44px;
      height: 44px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      color: #fff;
      font-weight: 760;
      background: linear-gradient(135deg, #6b3a12, #d97706 48%, #fbbf24);
      box-shadow: 0 12px 28px rgba(146, 86, 20, .28);
      transition: transform .28s var(--ease), box-shadow .28s var(--ease), filter .28s var(--ease);
    }

    .brand:hover .mark {
      transform: translateY(-2px) rotate(-3deg) scale(1.04);
      box-shadow: 0 18px 36px rgba(146, 86, 20, .32);
      filter: saturate(1.12);
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4.2vw, 54px);
      line-height: 1;
      font-weight: 760;
      letter-spacing: 0;
    }

    .brand-title {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 10px;
      margin: 0;
      font-size: clamp(30px, 4.8vw, 58px);
      line-height: 1.16;
      font-weight: 840;
      letter-spacing: 0;
      padding-bottom: .08em;
    }

    .brand-title span {
      display: inline-block;
      background: linear-gradient(135deg, var(--green-dark), var(--green) 46%, #f4b545);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      padding-bottom: .04em;
    }

    .subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }

    .status-pill {
      min-width: 190px;
      border: 1px solid var(--hairline);
      background: rgba(255, 255, 255, .52);
      backdrop-filter: blur(22px) saturate(1.35);
      border-radius: 999px;
      padding: 10px 14px;
      color: #5c3b1a;
      box-shadow: 0 12px 32px rgba(120, 72, 20, .10);
      text-align: center;
      font-size: 13px;
      transition: transform .22s var(--ease), background .22s var(--ease), box-shadow .22s var(--ease);
    }

    .status-pill:hover {
      transform: translateY(-1px);
      background: rgba(255, 248, 232, .76);
      box-shadow: 0 16px 36px rgba(120, 72, 20, .13);
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 20px;
      align-items: stretch;
    }

    .panel {
      position: relative;
      border: 1px solid var(--hairline);
      border-radius: var(--radius);
      background: var(--glass);
      backdrop-filter: blur(30px) saturate(1.45);
      box-shadow: var(--shadow);
      overflow: hidden;
      transition: transform .28s var(--ease), box-shadow .28s var(--ease), border-color .28s var(--ease), background .28s var(--ease);
    }

    .panel::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      background: linear-gradient(135deg, rgba(255,255,255,.68), transparent 35%, rgba(255,255,255,.22) 72%, transparent);
      opacity: .55;
    }

    .panel:hover {
      transform: translateY(-2px);
      box-shadow: 0 30px 86px rgba(120, 72, 20, .20);
      border-color: rgba(245, 158, 11, .26);
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 64px;
      padding: 18px 20px;
      border-bottom: 1px solid rgba(17, 24, 39, .08);
      background: rgba(255, 255, 255, .28);
    }

    .panel-title {
      margin: 0;
      font-size: 17px;
      font-weight: 680;
    }

    .panel-body {
      padding: 18px;
    }

    .frame-control {
      display: grid;
      gap: 12px;
      margin-bottom: 16px;
    }

    .label-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: #5c3b1a;
      font-size: 13px;
      font-weight: 620;
    }

    .segmented {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 6px;
      padding: 6px;
      border-radius: 16px;
      background: rgba(255, 255, 255, .58);
      border: 1px solid rgba(17, 24, 39, .08);
    }

    .segmented button {
      height: 38px;
      border-radius: 12px;
      color: #7a5a38;
      background: transparent;
      font-weight: 650;
      transition: transform .18s var(--ease), background .18s var(--ease), color .18s var(--ease), box-shadow .18s var(--ease);
    }

    .segmented button:hover {
      background: rgba(255, 255, 255, .72);
      transform: translateY(-1px);
    }

    .segmented button:active {
      transform: scale(.94);
    }

    .segmented button.active {
      color: #fff;
      background: linear-gradient(135deg, #f59e0b, #fbbf24 56%, #2f7d64);
      box-shadow: 0 9px 24px rgba(245, 158, 11, .30);
      animation: softPop .28s var(--ease-pop);
    }

    .slots {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
    }

    .slot {
      position: relative;
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr);
      align-items: center;
      gap: 12px;
      min-height: 124px;
      padding: 10px;
      border-radius: 18px;
      border: 1px solid rgba(17, 24, 39, .09);
      background: rgba(255, 255, 255, .50);
      overflow: hidden;
      transform: translateZ(0);
      transition: transform .24s var(--ease), border-color .24s var(--ease), background .24s var(--ease), box-shadow .24s var(--ease);
      animation: listIn .42s var(--ease-pop) both;
    }

    .slot::after {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(110deg, transparent 20%, rgba(255,255,255,.46), transparent 80%);
      transform: translateX(-120%);
      transition: transform .65s var(--ease);
    }

    .slot:hover,
    .slot.dragging {
      transform: translateY(-2px);
      border-color: rgba(245, 158, 11, .36);
      background: rgba(255, 250, 238, .78);
      box-shadow: 0 16px 36px rgba(120, 72, 20, .12);
    }

    .slot.dragging {
      outline: 2px solid rgba(245, 158, 11, .22);
      outline-offset: -3px;
    }

    .slot:hover::after,
    .slot.dragging::after {
      transform: translateX(120%);
    }

    .thumb {
      width: 72px;
      aspect-ratio: 1;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(17, 24, 39, .08);
      background: rgba(255, 255, 255, .65);
      display: grid;
      place-items: center;
      color: #c08a3b;
      font-size: 22px;
      font-weight: 700;
      transition: transform .24s var(--ease), box-shadow .24s var(--ease);
    }

    .thumb img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      animation: imageReveal .34s var(--ease-pop) both;
    }

    .slot:hover .thumb {
      transform: scale(1.03);
      box-shadow: 0 8px 20px rgba(120, 72, 20, .12);
    }

    .slot-meta {
      min-width: 0;
    }

    .slot-title {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 4px;
    }

    .slot-name {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .slot-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      grid-column: 1 / -1;
    }

    .slot-actions .file-button {
      flex: 1;
    }

    .file-button,
    .icon-button {
      height: 38px;
      border-radius: 13px;
      display: inline-grid;
      place-items: center;
      border: 1px solid rgba(17, 24, 39, .10);
      background: rgba(255, 255, 255, .72);
      color: #3b2716;
      font-weight: 660;
      padding: 0 14px;
      user-select: none;
      transition: transform .18s var(--ease), box-shadow .18s var(--ease), background .18s var(--ease), border-color .18s var(--ease);
    }

    .file-button:hover,
    .icon-button:hover:not(:disabled) {
      transform: translateY(-1px);
      background: rgba(255, 255, 255, .92);
      border-color: rgba(245, 158, 11, .26);
      box-shadow: 0 10px 22px rgba(120, 72, 20, .12);
    }

    .file-button:active,
    .icon-button:active:not(:disabled) {
      transform: scale(.95);
    }

    .icon-button {
      width: 38px;
      padding: 0;
      font-size: 18px;
    }

    .file-input {
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }

    .actions {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 10px;
      margin-top: 16px;
      align-items: center;
    }

    .primary {
      position: relative;
      overflow: hidden;
      min-height: 48px;
      border-radius: 16px;
      color: #fff;
      font-weight: 720;
      background: linear-gradient(135deg, #7c3f10, #f59e0b 52%, #fbbf24);
      box-shadow: 0 16px 34px rgba(245, 158, 11, .32);
      transition: transform .2s var(--ease), box-shadow .2s var(--ease), filter .2s var(--ease);
    }

    .primary::before {
      content: "";
      position: absolute;
      top: -50%;
      bottom: -50%;
      width: 50%;
      left: -70%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,.34), transparent);
      transform: skewX(-18deg);
      transition: left .62s var(--ease);
    }

    .primary:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 22px 44px rgba(245, 158, 11, .38);
      filter: saturate(1.08);
    }

    .primary:hover:not(:disabled)::before {
      left: 120%;
    }

    .primary:active:not(:disabled) {
      transform: scale(.98);
    }

    .secondary {
      min-height: 48px;
      border-radius: 16px;
      padding: 0 16px;
      color: #5c3b1a;
      background: rgba(255,250,238,.70);
      border: 1px solid rgba(17, 24, 39, .10);
      font-weight: 660;
      transition: transform .18s var(--ease), background .18s var(--ease), box-shadow .18s var(--ease);
    }

    .secondary:hover:not(:disabled) {
      transform: translateY(-1px);
      background: rgba(255,247,224,.88);
      box-shadow: 0 12px 26px rgba(120, 72, 20, .10);
    }

    .secondary:active:not(:disabled) {
      transform: scale(.97);
    }

    .message {
      min-height: 18px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      transition: color .2s var(--ease);
    }

    .message:not(:empty) {
      animation: messageIn .24s var(--ease-pop);
    }

    .message.error {
      color: var(--rose);
    }

    .results-empty {
      min-height: 340px;
      display: grid;
      place-items: center;
      padding: 48px;
      text-align: center;
    }

    .empty-inner {
      width: min(480px, 100%);
      animation: floatIn .72s var(--ease-pop) both;
    }

    .empty-title {
      font-size: 28px;
      line-height: 1.12;
      font-weight: 760;
      margin-bottom: 12px;
    }

    .empty-copy {
      color: var(--muted);
      line-height: 1.7;
      font-size: 14px;
    }

    .loading {
      display: none;
      align-items: center;
      gap: 12px;
      color: #5c3b1a;
      font-weight: 650;
    }

    .spinner {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      border: 3px solid rgba(245, 158, 11, .22);
      border-top-color: var(--blue);
      animation: spin 1s linear infinite;
    }

    .loading span {
      position: relative;
    }

    .loading span::after {
      content: "";
      position: absolute;
      width: 4px;
      height: 4px;
      border-radius: 50%;
      right: -10px;
      bottom: 4px;
      background: currentColor;
      box-shadow: 7px 0 0 currentColor, 14px 0 0 currentColor;
      animation: dots 1.2s steps(4, end) infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes dots {
      0%, 20% { opacity: .2; transform: translateY(0); }
      50% { opacity: 1; transform: translateY(-2px); }
      100% { opacity: .2; transform: translateY(0); }
    }

    @keyframes pageFade {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    @keyframes shellRise {
      from { opacity: 0; transform: translateY(18px) scale(.992); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes gridDrift {
      from { background-position: 0 0, 0 0; }
      to { background-position: 72px 72px, 72px 72px; }
    }

    @keyframes softPop {
      0% { transform: scale(.92); }
      70% { transform: scale(1.04); }
      100% { transform: scale(1); }
    }

    @keyframes listIn {
      from { opacity: 0; transform: translateY(10px) scale(.985); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes imageReveal {
      from { opacity: 0; transform: scale(1.08); filter: blur(8px); }
      to { opacity: 1; transform: scale(1); filter: blur(0); }
    }

    @keyframes messageIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes floatIn {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes cardIn {
      from { opacity: 0; transform: translateY(18px) scale(.985); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .result-wrap {
      display: none;
    }

    .summary {
      display: grid;
      gap: 12px;
      padding: 18px;
      border-bottom: 1px solid rgba(17, 24, 39, .08);
      background: rgba(255, 248, 232, .32);
    }

    .query-text {
      padding: 14px 15px;
      border-radius: 16px;
      background: rgba(255, 250, 238, .72);
      color: #3b2716;
      border: 1px solid rgba(17,24,39,.08);
      line-height: 1.55;
      font-size: 14px;
      animation: floatIn .36s var(--ease-pop) both;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.62);
      border: 1px solid rgba(17, 24, 39, .08);
      color: #5c3b1a;
      font-size: 12px;
      max-width: 100%;
      animation: softPop .3s var(--ease-pop) both;
      transition: transform .16s var(--ease), background .16s var(--ease);
    }

    .chip:hover {
      transform: translateY(-1px);
      background: rgba(255,255,255,.78);
    }

    .chip b {
      color: #2a1a10;
      font-weight: 720;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
      gap: 14px;
      padding: 18px;
    }

    .candidate {
      position: relative;
      display: grid;
      grid-template-columns: 152px minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
      border-radius: 20px;
      border: 1px solid rgba(17, 24, 39, .10);
      background: rgba(255,255,255,.58);
      min-height: 220px;
      overflow: hidden;
      animation: cardIn .5s var(--ease-pop) both;
      transition: transform .24s var(--ease), box-shadow .24s var(--ease), border-color .24s var(--ease), background .24s var(--ease);
    }

    .candidate::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: radial-gradient(circle at var(--mx, 50%) var(--my, 0%), rgba(245,158,11,.18), transparent 34%);
      opacity: 0;
      transition: opacity .24s var(--ease);
    }

    .candidate:hover {
      transform: translateY(-3px);
      border-color: rgba(245, 158, 11, .30);
      background: rgba(255,250,238,.82);
      box-shadow: 0 20px 44px rgba(120, 72, 20, .16);
    }

    .candidate:hover::before {
      opacity: 1;
    }

    .candidate.top-match {
      grid-column: 1 / -1;
      grid-template-columns: 190px minmax(0, 1fr);
      min-height: 260px;
      padding: 18px;
      border: 1px solid rgba(245, 158, 11, .44);
      background:
        linear-gradient(135deg, rgba(255, 248, 232, .94), rgba(255, 236, 190, .76)),
        rgba(255, 250, 238, .90);
      box-shadow: 0 28px 58px rgba(120, 72, 20, .20);
    }

    .candidate.top-match::after {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      background: linear-gradient(120deg, rgba(255,255,255,.58), transparent 36%, rgba(251,191,36,.16));
    }

    .candidate.top-match .candidate-image {
      position: relative;
      z-index: 1;
      border-color: rgba(245, 158, 11, .24);
      box-shadow: 0 16px 34px rgba(120, 72, 20, .14);
    }

    .candidate.top-match .candidate-main {
      position: relative;
      z-index: 1;
    }

    .candidate.top-match .words {
      font-size: clamp(24px, 3vw, 34px);
      line-height: 1.18;
    }

    .candidate.top-match .rank {
      font-size: 16px;
      color: #92400e;
    }

    .best-badge {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 5px 10px;
      border-radius: 999px;
      background: linear-gradient(135deg, #f59e0b, #fbbf24);
      color: #fff;
      font-size: 12px;
      font-weight: 760;
      box-shadow: 0 10px 22px rgba(245, 158, 11, .26);
    }

    .candidate-image {
      width: 100%;
      aspect-ratio: 1;
      border-radius: 16px;
      overflow: hidden;
      background: #fff;
      border: 1px solid rgba(17, 24, 39, .08);
      display: grid;
      place-items: center;
      transition: transform .24s var(--ease), box-shadow .24s var(--ease);
    }

    .candidate-image img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      animation: imageReveal .38s var(--ease-pop) both;
    }

    .candidate:hover .candidate-image {
      transform: scale(1.025);
      box-shadow: 0 12px 28px rgba(120, 72, 20, .11);
    }

    .candidate-main {
      min-width: 0;
      display: grid;
      align-content: start;
      gap: 8px;
    }

    .rank-line {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }

    .rank {
      color: var(--blue);
      font-weight: 760;
      font-size: 14px;
    }

    .confidence {
      color: #2f7d64;
      font-size: 13px;
      font-weight: 720;
    }

    .words {
      font-size: 20px;
      line-height: 1.25;
      font-weight: 760;
      word-break: keep-all;
      overflow-wrap: anywhere;
    }

    .meta {
      color: #8a6b4b;
      font-size: 12px;
      line-height: 1.45;
    }

    .reason {
      color: #b45309;
      line-height: 1.55;
      font-size: 13px;
    }

    details {
      border-top: 1px solid rgba(17,24,39,.08);
      padding-top: 8px;
      color: #5c3b1a;
      font-size: 13px;
    }

    summary {
      cursor: pointer;
      color: #7a5a38;
      font-weight: 650;
      user-select: none;
      transition: color .16s var(--ease);
    }

    summary:hover {
      color: var(--blue);
    }

    .desc {
      margin: 8px 0 0;
      white-space: pre-wrap;
      line-height: 1.6;
      color: #5c3b1a;
      animation: floatIn .22s var(--ease-pop) both;
    }

    .click-ripple {
      position: fixed;
      width: 12px;
      height: 12px;
      margin-left: -6px;
      margin-top: -6px;
      border-radius: 50%;
      pointer-events: none;
      background: rgba(245, 158, 11, .26);
      transform: scale(1);
      animation: ripple .52s var(--ease) forwards;
      z-index: 20;
    }

    @keyframes ripple {
      to {
        opacity: 0;
        transform: scale(9);
      }
    }

    .footer-line {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px 18px;
      color: #667085;
      font-size: 12px;
    }

    .developer-credit {
      justify-self: end;
      margin-top: 18px;
      padding: 9px 12px;
      border-radius: 999px;
      border: 1px solid rgba(126, 78, 27, .16);
      background: rgba(255, 248, 232, .72);
      backdrop-filter: blur(22px) saturate(1.35);
      box-shadow: 0 12px 30px rgba(120, 72, 20, .12);
      color: #6b3a12;
      font-size: 12px;
      line-height: 1.2;
      animation: floatIn .52s var(--ease-pop) both;
    }

    /* ASL Bloom inspired warm learning-product skin. */
    :root {
      --ink: #263931;
      --muted: #6f8178;
      --cream: #fff8ec;
      --butter: #ffefc9;
      --peach: #ffd9c6;
      --mint: #dff2df;
      --green: #2f8b57;
      --green-dark: #1f6f48;
      --sun: #f4b545;
      --surface: #fffdf7;
      --surface-soft: #fff5e7;
      --hairline: rgba(47, 139, 87, .16);
      --shadow: 0 22px 52px rgba(92, 70, 45, .12);
      --radius: 30px;
      --blue: var(--green);
      --amber: var(--sun);
    }

    body {
      background:
        radial-gradient(circle at 11% 8%, rgba(255, 217, 198, .82), transparent 29%),
        radial-gradient(circle at 86% 0%, rgba(223, 242, 223, .9), transparent 30%),
        linear-gradient(135deg, #fffaf0 0%, #fff0d9 46%, #fff8ed 100%);
    }

    body::before {
      background-image: radial-gradient(circle, rgba(47, 139, 87, .10) 1.1px, transparent 1.5px);
      background-size: 28px 28px;
    }

    .shell {
      width: min(1120px, calc(100vw - 40px));
      padding-top: 22px;
    }

    .topbar {
      margin-bottom: 18px;
      padding: 10px 0 2px;
    }

    .mark {
      width: 52px;
      height: 52px;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--green-dark), var(--green));
      box-shadow: 0 14px 30px rgba(47, 139, 87, .28);
    }

    .brand:hover .mark {
      box-shadow: 0 18px 38px rgba(47, 139, 87, .34);
    }

    h1 {
      font-size: clamp(32px, 4.6vw, 60px);
      color: var(--ink);
    }

    .status-pill {
      color: var(--green-dark);
      border-color: rgba(47, 139, 87, .16);
      background: rgba(255, 253, 247, .9);
      box-shadow: 0 12px 26px rgba(92, 70, 45, .08);
      backdrop-filter: none;
    }

    .hero-card {
      position: relative;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 24px;
      align-items: center;
      margin-bottom: 22px;
      padding: clamp(22px, 4vw, 34px);
      border-radius: 38px;
      border: 1px solid rgba(47, 139, 87, .14);
      background:
        linear-gradient(135deg, rgba(255, 253, 247, .96), rgba(255, 239, 201, .78)),
        var(--surface);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .hero-card::before {
      content: "";
      position: absolute;
      inset: auto -40px -60px auto;
      width: 230px;
      aspect-ratio: 1;
      border-radius: 50%;
      background: rgba(223, 242, 223, .72);
    }

    .hero-kicker {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 32px;
      padding: 7px 12px;
      border-radius: 999px;
      background: var(--mint);
      color: var(--green-dark);
      font-size: 13px;
      font-weight: 760;
      margin-bottom: 14px;
    }

    .hero-title {
      margin: 0;
      color: var(--ink);
      font-size: clamp(28px, 5vw, 56px);
      line-height: 1.04;
      font-weight: 820;
    }

    .hero-copy {
      margin: 14px 0 0;
      max-width: 640px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.7;
    }

    .hero-visual {
      position: relative;
      width: clamp(116px, 18vw, 178px);
      aspect-ratio: 1;
      border-radius: 42px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, var(--peach), var(--mint));
      box-shadow: 0 18px 34px rgba(92, 70, 45, .13);
      color: var(--green-dark);
      font-size: clamp(48px, 8vw, 82px);
      font-weight: 820;
      transform: rotate(2deg);
    }

    .hero-visual::after {
      content: "";
      position: absolute;
      right: 18px;
      top: 18px;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: var(--sun);
      box-shadow: -52px 96px 0 rgba(255,255,255,.62);
    }

    .panel {
      border-color: rgba(47, 139, 87, .16);
      background: var(--surface);
      box-shadow: var(--shadow);
      backdrop-filter: none;
    }

    .panel::before {
      background: linear-gradient(135deg, rgba(255,255,255,.65), transparent 44%);
      opacity: .8;
    }

    .panel:hover {
      box-shadow: 0 28px 64px rgba(92, 70, 45, .14);
      border-color: rgba(47, 139, 87, .24);
    }

    .panel-head {
      padding: 20px 24px;
      border-bottom-color: rgba(47, 139, 87, .10);
      background: linear-gradient(90deg, rgba(223, 242, 223, .58), rgba(255, 217, 198, .36));
    }

    .panel-title {
      color: var(--ink);
      font-size: 20px;
      font-weight: 800;
    }

    .panel-body {
      padding: 22px;
    }

    .label-row {
      color: var(--green-dark);
    }

    .segmented {
      border-radius: 999px;
      background: #fff3df;
      border-color: rgba(47, 139, 87, .12);
    }

    .segmented button {
      border-radius: 999px;
      color: var(--muted);
    }

    .segmented button:hover {
      background: #fffdf7;
    }

    .segmented button.active {
      background: var(--green);
      box-shadow: 0 10px 22px rgba(47, 139, 87, .25);
    }

    .slots {
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }

    .slot {
      min-height: 132px;
      padding: 12px;
      border-radius: 26px;
      border-color: rgba(47, 139, 87, .13);
      background: #fffdf7;
    }

    .slot:hover,
    .slot.dragging {
      border-color: rgba(47, 139, 87, .27);
      background: #fff;
      box-shadow: 0 16px 30px rgba(92, 70, 45, .10);
    }

    .slot.dragging {
      outline-color: rgba(47, 139, 87, .20);
    }

    .thumb {
      border-radius: 22px;
      border-color: rgba(47, 139, 87, .13);
      background: var(--mint);
      color: var(--green);
    }

    .file-button,
    .icon-button {
      border-radius: 999px;
      border-color: rgba(47, 139, 87, .14);
      background: #fff3df;
      color: var(--green-dark);
    }

    .file-button:hover,
    .icon-button:hover:not(:disabled) {
      border-color: rgba(47, 139, 87, .24);
      background: #fffdf7;
      box-shadow: 0 10px 22px rgba(47, 139, 87, .10);
    }

    .primary {
      border-radius: 999px;
      background: linear-gradient(135deg, var(--green-dark), var(--green));
      box-shadow: 0 16px 32px rgba(47, 139, 87, .28);
    }

    .primary:hover:not(:disabled) {
      box-shadow: 0 20px 40px rgba(47, 139, 87, .34);
    }

    .secondary {
      border-radius: 999px;
      color: var(--green-dark);
      background: #fff3df;
      border-color: rgba(47, 139, 87, .13);
    }

    .secondary:hover:not(:disabled) {
      background: #fffdf7;
      box-shadow: 0 12px 24px rgba(92, 70, 45, .09);
    }

    .loading,
    .message,
    .empty-copy {
      color: var(--muted);
    }

    .spinner {
      border-color: rgba(47, 139, 87, .18);
      border-top-color: var(--green);
    }

    .results-empty {
      min-height: 300px;
      background: linear-gradient(135deg, rgba(255, 253, 247, .45), rgba(223, 242, 223, .28));
    }

    .summary {
      background: #fff7eb;
      border-bottom-color: rgba(47, 139, 87, .10);
    }

    .query-text {
      border-radius: 24px;
      background: #fffdf7;
      border-color: rgba(47, 139, 87, .12);
      color: var(--ink);
    }

    .chip {
      background: var(--mint);
      color: var(--green-dark);
      border-color: rgba(47, 139, 87, .12);
    }

    .chip b {
      color: var(--green-dark);
    }

    .cards {
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }

    .candidate {
      border-radius: 28px;
      border-color: rgba(47, 139, 87, .12);
      background: #fffdf7;
    }

    .candidate::before {
      background: radial-gradient(circle at var(--mx, 50%) var(--my, 0%), rgba(47,139,87,.14), transparent 34%);
    }

    .candidate:hover {
      border-color: rgba(47, 139, 87, .24);
      background: #fff;
      box-shadow: 0 20px 40px rgba(92, 70, 45, .12);
    }

    .candidate.top-match {
      border-color: rgba(47, 139, 87, .28);
      background:
        linear-gradient(135deg, rgba(223, 242, 223, .92), rgba(255, 217, 198, .62)),
        #fffdf7;
      box-shadow: 0 28px 54px rgba(47, 139, 87, .16);
    }

    .candidate.top-match::after {
      background: linear-gradient(120deg, rgba(255,255,255,.62), transparent 36%, rgba(47,139,87,.10));
    }

    .candidate.top-match .candidate-image {
      border-color: rgba(47, 139, 87, .20);
      box-shadow: 0 16px 32px rgba(47, 139, 87, .12);
    }

    .candidate.top-match .rank,
    .rank,
    summary:hover {
      color: var(--green);
    }

    .best-badge {
      background: var(--green);
      box-shadow: 0 10px 22px rgba(47, 139, 87, .22);
    }

    .candidate-image {
      border-radius: 22px;
      border-color: rgba(47, 139, 87, .12);
    }

    .confidence {
      color: var(--green-dark);
    }

    .meta,
    .footer-line {
      color: var(--muted);
    }

    .reason {
      color: var(--rose);
    }

    details,
    .desc {
      color: var(--ink);
    }

    summary {
      color: var(--muted);
    }

    .click-ripple {
      background: rgba(47, 139, 87, .24);
    }

    .developer-credit {
      border-color: rgba(47, 139, 87, .14);
      background: rgba(255, 253, 247, .90);
      backdrop-filter: none;
      box-shadow: 0 12px 28px rgba(92, 70, 45, .10);
      color: var(--green-dark);
    }

    .credit-row {
      display: grid;
      justify-items: end;
    }

    @media (max-width: 760px) {
      .hero-card {
        grid-template-columns: 1fr;
      }

      .hero-visual {
        width: 112px;
        border-radius: 30px;
      }
    }

    @media (max-width: 760px) {
      .shell {
        width: min(100vw - 24px, 640px);
        padding-top: 16px;
      }

      .topbar {
        grid-template-columns: 1fr;
      }

      .status-pill {
        width: 100%;
      }

      .panel-head,
      .panel-body,
      .summary,
      .cards {
        padding-left: 12px;
        padding-right: 12px;
      }

      .slots {
        grid-template-columns: 1fr;
      }

      .slot {
        grid-template-columns: 68px minmax(0, 1fr);
        min-height: 96px;
      }

      .thumb {
        width: 58px;
      }

      .slot-actions {
        grid-column: 1 / -1;
        justify-content: stretch;
      }

      .actions {
        grid-template-columns: 1fr;
      }

      .cards {
        grid-template-columns: 1fr;
      }

      .candidate {
        grid-template-columns: 116px minmax(0, 1fr);
      }

      .candidate.top-match {
        grid-template-columns: 132px minmax(0, 1fr);
        min-height: 220px;
      }

      .words {
        font-size: 17px;
      }
    }

    @media (max-width: 460px) {
      .segmented {
        grid-template-columns: repeat(5, 1fr);
      }

      .candidate {
        grid-template-columns: 1fr;
      }

      .candidate.top-match {
        grid-template-columns: 1fr;
      }

      .candidate-image {
        max-height: 240px;
      }

      .credit-row {
        justify-items: center;
      }

      .developer-credit {
        max-width: 100%;
        font-size: 11px;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .001ms !important;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="mark" aria-hidden="true">S</div>
        <div>
          <h1 class="brand-title"><span>见手知意</span><span>SignSeek</span></h1>
        </div>
      </div>
    </header>

    <section class="hero-card">
      <div>
        <h2 class="hero-title">上传手语图片，检索其真意</h2>
        <p class="hero-copy">生命是一场旅程，我们等了多少个轮回，才有机会去享受这一次旅程。</p>
      </div>
      <div class="hero-visual" aria-hidden="true">手</div>
    </section>

    <section class="layout">
      <section class="panel">
        <div class="panel-head">
          <h2 class="panel-title">关键帧</h2>
          <div class="loading" id="loading">
            <div class="spinner" aria-hidden="true"></div>
            <span>反查中</span>
          </div>
        </div>
        <div class="panel-body">
          <div class="frame-control">
            <div class="label-row">
              <span>动作帧数</span>
              <span id="filledCount">0 / 1</span>
            </div>
            <div class="segmented" id="frameButtons" aria-label="动作帧数"></div>
          </div>

          <div class="slots" id="slots"></div>

          <div class="actions">
            <button class="primary" id="runButton" disabled>开始反查</button>
            <button class="secondary" id="resetButton" type="button">清空</button>
          </div>
          <div class="message" id="message"></div>
        </div>
      </section>

      <section class="panel" id="resultPanel">
        <div class="results-empty" id="emptyState">
          <div class="empty-inner">
            <div class="empty-title">Top-9 匹配结果</div>
            <div class="empty-copy">等待查询</div>
          </div>
        </div>

        <div class="result-wrap" id="resultWrap">
          <div class="panel-head">
            <h2 class="panel-title">匹配结果</h2>
            <div class="status-pill" id="elapsed">--</div>
          </div>
          <div class="summary">
            <div class="query-text" id="queryText"></div>
            <div class="chips" id="descChips"></div>
          </div>
          <div class="cards" id="cards"></div>
          <div class="footer-line">
            <span>Top-9</span>
            <span id="modelInfo"></span>
          </div>
        </div>
      </section>
    </section>
    <div class="credit-row">
      <div class="developer-credit">开发者：手语分社心创组 涂增基</div>
    </div>
  </main>

  <script>
    const state = {
      frameCount: 1,
      files: new Map(),
      busy: false,
      health: null,
    };

    const frameButtons = document.getElementById("frameButtons");
    const slotsEl = document.getElementById("slots");
    const runButton = document.getElementById("runButton");
    const resetButton = document.getElementById("resetButton");
    const message = document.getElementById("message");
    const filledCount = document.getElementById("filledCount");
    const loading = document.getElementById("loading");
    const resultPanel = document.getElementById("resultPanel");
    const emptyState = document.getElementById("emptyState");
    const resultWrap = document.getElementById("resultWrap");
    const queryText = document.getElementById("queryText");
    const descChips = document.getElementById("descChips");
    const cards = document.getElementById("cards");
    const elapsed = document.getElementById("elapsed");
    const modelInfo = document.getElementById("modelInfo");

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setMessage(text, isError = false) {
      message.textContent = text || "";
      message.classList.toggle("error", Boolean(isError));
    }

    function pulseElement(el) {
      if (!el) return;
      el.animate([
        { transform: "scale(.98)" },
        { transform: "scale(1.015)" },
        { transform: "scale(1)" },
      ], { duration: 260, easing: "cubic-bezier(.16, 1, .3, 1)" });
    }

    function setBusy(next) {
      state.busy = next;
      loading.style.display = next ? "flex" : "none";
      runButton.disabled = next || !canRun();
      resetButton.disabled = next;
      document.querySelectorAll("input[type=file], #frameButtons button").forEach(el => {
        el.disabled = next;
      });
    }

    function canRun() {
      if (state.busy) return false;
      for (let i = 0; i < state.frameCount; i += 1) {
        if (!state.files.has(i)) return false;
      }
      return true;
    }

    function updateControls() {
      const count = Array.from(state.files.keys()).filter(i => i < state.frameCount).length;
      filledCount.textContent = `${count} / ${state.frameCount}`;
      runButton.disabled = !canRun();
    }

    function renderFrameButtons() {
      frameButtons.innerHTML = "";
      for (let i = 1; i <= 5; i += 1) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = i;
        button.className = i === state.frameCount ? "active" : "";
        button.addEventListener("click", () => {
          state.frameCount = i;
          for (const key of Array.from(state.files.keys())) {
            if (key >= i) state.files.delete(key);
          }
          renderAll();
        });
        frameButtons.appendChild(button);
      }
    }

    function renderSlots() {
      slotsEl.innerHTML = "";
      for (let i = 0; i < state.frameCount; i += 1) {
        const item = state.files.get(i);
        const slot = document.createElement("div");
        slot.className = "slot";
        slot.style.animationDelay = `${i * 42}ms`;
        const fileId = `frame_${i}`;
        slot.innerHTML = `
          <div class="thumb">${item ? `<img src="${item.preview}" alt="第 ${i + 1} 帧预览">` : String(i + 1)}</div>
          <div class="slot-meta">
            <div class="slot-title">第 ${i + 1} 帧</div>
            <div class="slot-name">${item ? escapeHtml(item.name) : "未选择照片"}</div>
          </div>
          <div class="slot-actions">
            <label class="file-button" for="${fileId}">${item ? "替换" : "上传"}</label>
            <button class="icon-button" type="button" aria-label="清除第 ${i + 1} 帧" ${item && !state.busy ? "" : "disabled"}>x</button>
            <input class="file-input" id="${fileId}" type="file" accept="image/*">
          </div>
        `;
        const input = slot.querySelector("input");
        const clearButton = slot.querySelector("button");
        async function acceptFile(file) {
          if (!file || state.busy) return;
          try {
            setMessage("");
            slot.classList.remove("dragging");
            const prepared = await prepareImage(file);
            state.files.set(i, prepared);
            pulseElement(slot);
            renderAll();
          } catch (err) {
            setMessage(err.message || String(err), true);
          }
        }

        input.addEventListener("change", async (event) => {
          const file = event.target.files && event.target.files[0];
          await acceptFile(file);
        });
        slot.addEventListener("dragenter", (event) => {
          event.preventDefault();
          if (!state.busy) slot.classList.add("dragging");
        });
        slot.addEventListener("dragover", (event) => {
          event.preventDefault();
          if (!state.busy) event.dataTransfer.dropEffect = "copy";
        });
        slot.addEventListener("dragleave", (event) => {
          if (!slot.contains(event.relatedTarget)) {
            slot.classList.remove("dragging");
          }
        });
        slot.addEventListener("drop", async (event) => {
          event.preventDefault();
          slot.classList.remove("dragging");
          const file = event.dataTransfer.files && event.dataTransfer.files[0];
          await acceptFile(file);
        });
        clearButton.addEventListener("click", () => {
          state.files.delete(i);
          pulseElement(slot);
          renderAll();
        });
        slotsEl.appendChild(slot);
      }
    }

    function renderAll() {
      renderFrameButtons();
      renderSlots();
      updateControls();
      setBusy(state.busy);
    }

    function readAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("图片读取失败"));
        reader.readAsDataURL(file);
      });
    }

    function loadImage(src) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("图片解码失败"));
        img.src = src;
      });
    }

    async function prepareImage(file) {
      if (!file.type.startsWith("image/")) {
        throw new Error("请选择图片文件");
      }

      const original = await readAsDataUrl(file);
      try {
        const img = await loadImage(original);
        const maxEdge = 1600;
        const scale = Math.min(1, maxEdge / Math.max(img.naturalWidth, img.naturalHeight));
        const width = Math.max(1, Math.round(img.naturalWidth * scale));
        const height = Math.max(1, Math.round(img.naturalHeight * scale));
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d", { alpha: false });
        ctx.drawImage(img, 0, 0, width, height);
        const dataUrl = canvas.toDataURL("image/jpeg", .86);
        return {
          name: file.name || "frame.jpg",
          mime: "image/jpeg",
          data: dataUrl,
          preview: dataUrl,
        };
      } catch (_) {
        return {
          name: file.name || "frame",
          mime: file.type || "application/octet-stream",
          data: original,
          preview: original,
        };
      }
    }

    async function loadHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        state.health = data;
        modelInfo.textContent = data.index_meta && data.index_meta.model ? data.index_meta.model : "";
      } catch (_) {
        modelInfo.textContent = "";
      }
    }

    async function runQuery() {
      if (!canRun()) return;
      pulseElement(runButton);
      setBusy(true);
      setMessage("正在提交关键帧");
      emptyState.style.display = "none";
      resultWrap.style.display = "block";
      queryText.textContent = "";
      descChips.innerHTML = "";
      cards.innerHTML = "";
      elapsed.textContent = "处理中";

      const frames = [];
      for (let i = 0; i < state.frameCount; i += 1) {
        const item = state.files.get(i);
        frames.push({ name: item.name, mime: item.mime, data: item.data });
      }

      try {
        setMessage("正在理解关键帧、召回词条并重排");
        const res = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frames, top_k: 9 }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "反查失败");
        }
        renderResult(data);
        resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
        setMessage("完成");
      } catch (err) {
        elapsed.textContent = "--";
        setMessage(err.message || String(err), true);
      } finally {
        setBusy(false);
      }
    }

    function labelForKey(key) {
      return {
        hands: "手数",
        handshape: "手型",
        orientation: "朝向",
        location: "位置",
        movement: "运动",
        expression: "表情",
        iconicity: "象形",
        resembles: "字形",
      }[key] || key;
    }

    function renderResult(data) {
      queryText.textContent = data.query_text || "";
      elapsed.textContent = data.elapsed ? `${data.elapsed.toFixed(1)}s` : "--";

      descChips.innerHTML = "";
      const desc = data.description || {};
      ["hands", "handshape", "orientation", "location", "movement", "resembles"].forEach((key, index) => {
        const value = desc[key];
        if (!value || String(value).toLowerCase() === "uncertain") return;
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.style.animationDelay = `${index * 42}ms`;
        chip.innerHTML = `<b>${escapeHtml(labelForKey(key))}</b>${escapeHtml(value)}`;
        descChips.appendChild(chip);
      });

      cards.innerHTML = "";
      (data.results || []).forEach((item, index) => {
        const confidence = typeof item.confidence === "number" ? `${Math.round(item.confidence * 100)}%` : "--";
        const recall = item.recall_rank === 0 ? "词法注入" : `召回#${item.recall_rank}`;
        const sim = typeof item.recall_sim === "number" ? item.recall_sim.toFixed(3) : "--";
        const card = document.createElement("article");
        card.className = index === 0 ? "candidate top-match" : "candidate";
        card.style.animationDelay = `${index * 54}ms`;
        card.innerHTML = `
          <div class="candidate-image">
            ${item.image_data ? `<img src="${item.image_data}" alt="${escapeHtml(item.words)}">` : ""}
          </div>
          <div class="candidate-main">
            <div class="rank-line">
              <div class="rank">#${index + 1}</div>
              ${index === 0 ? `<div class="best-badge">最匹配</div>` : ""}
              <div class="confidence">${confidence}</div>
            </div>
            <div class="words">${escapeHtml(item.words)}</div>
            <div class="meta">id=${escapeHtml(item.id)} · ${escapeHtml(recall)} · 相似度 ${escapeHtml(sim)}</div>
            <div class="reason">${escapeHtml(item.reason || "")}</div>
            <details>
              <summary>打法描述</summary>
              <div class="desc">${escapeHtml(item.description || "")}</div>
            </details>
          </div>
        `;
        card.addEventListener("pointermove", (event) => {
          const rect = card.getBoundingClientRect();
          card.style.setProperty("--mx", `${event.clientX - rect.left}px`);
          card.style.setProperty("--my", `${event.clientY - rect.top}px`);
        });
        cards.appendChild(card);
      });
    }

    resetButton.addEventListener("click", () => {
      pulseElement(resetButton);
      state.files.clear();
      emptyState.style.display = "grid";
      resultWrap.style.display = "none";
      setMessage("");
      renderAll();
    });

    runButton.addEventListener("click", runQuery);

    document.addEventListener("pointerdown", (event) => {
      const target = event.target.closest("button, .file-button, summary");
      if (!target || target.disabled) return;
      const dot = document.createElement("span");
      dot.className = "click-ripple";
      dot.style.left = `${event.clientX}px`;
      dot.style.top = `${event.clientY}px`;
      document.body.appendChild(dot);
      dot.addEventListener("animationend", () => dot.remove(), { once: true });
    });

    renderAll();
    loadHealth();
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    data = html.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    if "," not in data_url:
        raise ValueError("图片数据格式不正确")
    header, encoded = data_url.split(",", 1)
    mime = "application/octet-stream"
    if header.startswith("data:") and ";base64" in header:
        mime = header[5:header.index(";base64")] or mime
    raw = base64.b64decode(encoded, validate=True)
    if not raw:
        raise ValueError("图片数据为空")
    return mime, raw


def _safe_suffix(name: str, mime: str) -> str:
    suffix = Path(name or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return suffix
    guessed = mimetypes.guess_extension(mime or "")
    if guessed in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return guessed
    return ".jpg"


def _image_data_uri(path: Path, max_px: int = 520) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=84)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length") or "0")
    if content_length <= 0:
        raise ValueError("请求体为空")
    if content_length > MAX_BODY_BYTES:
        raise ValueError("上传图片过大")
    body = handler.rfile.read(content_length)
    return json.loads(body.decode("utf-8"))


def _run_query_api(payload: dict) -> dict:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("缺少关键帧")
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"关键帧最多 {MAX_FRAMES} 张")

    top_k = int(payload.get("top_k") or 9)
    top_k = max(1, min(top_k, 9))

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="signseek_web_") as tmp:
        tmpdir = Path(tmp)
        frame_paths: list[Path] = []
        for index, frame in enumerate(frames, start=1):
            if not isinstance(frame, dict):
                raise ValueError("关键帧格式不正确")
            name = str(frame.get("name") or f"frame_{index}.jpg")
            mime, raw = _decode_data_url(str(frame.get("data") or ""))
            suffix = _safe_suffix(name, mime)
            path = tmpdir / f"frame_{index:02d}{suffix}"
            path.write_bytes(raw)
            frame_paths.append(path)

        result = query.run_query(
            frame_paths,
            top_recall=50,
            top_text=max(16, top_k),
            top_final=top_k,
            visual=True,
            max_rerank_images=10,
        )

    response_results = []
    for rank, item in enumerate(result.get("results", []), start=1):
        image_data = ""
        try:
            image_data = _image_data_uri(config.resolve_image(item["image_path"]))
        except Exception:  # noqa: BLE001
            image_data = ""
        response_results.append({
            "rank": rank,
            "id": item.get("id"),
            "words": item.get("words", ""),
            "description": item.get("description", ""),
            "image_path": item.get("image_path", ""),
            "image_data": image_data,
            "confidence": item.get("confidence"),
            "reason": item.get("reason", ""),
            "recall_rank": item.get("recall_rank"),
            "recall_sim": item.get("recall_sim"),
        })

    return {
        "elapsed": time.time() - started,
        "description": result.get("description", {}),
        "query_text": result.get("query_text", ""),
        "results": response_results,
    }


class SignSeekHandler(BaseHTTPRequestHandler):
    server_version = "SignSeekLocal/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            _html_response(self, INDEX_HTML)
            return
        if path == "/api/health":
            index_meta = {}
            if config.INDEX_META_PATH.exists():
                index_meta = json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
            _json_response(self, HTTPStatus.OK, {
                "ok": True,
                "model": config.LLM_MODEL,
                "has_api_key": bool(config.LLM_API_KEY),
                "index_meta": index_meta,
            })
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html", "/api/health"}:
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/query":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = _read_json_body(self)
            data = _run_query_api(payload)
            _json_response(self, HTTPStatus.OK, data)
        except ValueError as exc:
            _json_response(self, HTTPStatus.BAD_REQUEST, {
                "error": str(exc),
            })
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": str(exc),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="SignSeek local web UI")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="open the UI in the default browser")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SignSeekHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"SignSeek Web UI: {url}")
    print("Press Ctrl-C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SignSeek Web UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
