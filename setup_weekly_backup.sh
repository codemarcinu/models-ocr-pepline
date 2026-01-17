#!/bin/bash

# Configuration
PYTHON_EXEC="/home/marcin/obsidian/venv/bin/python"
SCRIPT_PATH="/home/marcin/obsidian/backup_to_drive.py"
LOG_FILE="/home/marcin/obsidian/backup_wrapper.log"

# Check if files exist
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "❌ Error: Python executable not found at $PYTHON_EXEC"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Backup script not found at $SCRIPT_PATH"
    exit 1
fi

# The cron command (Runs every Monday at 4:00 AM)
CRON_CMD="0 4 * * 1 $PYTHON_EXEC $SCRIPT_PATH >> $LOG_FILE 2>&1"

# Check if job already exists
(crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH") >/dev/null

if [ $? -eq 0 ]; then
    echo "⚠️  Backup job already exists in crontab."
else
    # Add the job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Successfully added weekly backup job to crontab."
    echo "📅 Schedule: Every Monday at 4:00 AM"
    echo "📜 Command: $CRON_CMD"
fi

echo ""
echo "Do you want to run a test backup now? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    echo "🚀 Starting test backup..."
    $PYTHON_EXEC $SCRIPT_PATH
    echo "✅ Test run complete. Check 'backup.log' for details."
else
    echo "Skipping test run."
fi
