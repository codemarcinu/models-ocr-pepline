#!/bin/bash
cd /home/marcin/obsidian
source venv/bin/activate
mkdir -p logs
python3 -u brain_guard.py > logs/brain_guard.log 2>&1