import requests
import datetime
import logging
from pathlib import Path
from config import ProjectConfig
from collections import defaultdict

# Configure logger
logger = logging.getLogger("OpenAI_Usage_Monitor")

def get_monthly_token_usage(api_key: str) -> dict:
    """
    Fetches daily usage from OpenAI API for the entire month and aggregates it.
    This is a workaround for the locked billing API.
    """
    today = datetime.date.today()
    start_of_month = today.replace(day=1)
    
    total_requests = 0
    model_stats = defaultdict(lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0})

    # Iterate day by day from the start of the month to today
    current_day = start_of_month
    while current_day <= today:
        date_str = current_day.strftime('%Y-%m-%d')
        url = f"https://api.openai.com/v1/usage?date={date_str}"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        try:
            logger.debug(f"Fetching usage for {date_str}...", extra={"tags": "FETCH-DAILY"})
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            daily_data = response.json().get("data", [])

            for entry in daily_data:
                model = entry.get("snapshot_id", "unknown-model")
                requests_count = entry.get("n_requests", 0)
                context_tokens = entry.get("n_context_tokens_total", 0)
                generated_tokens = entry.get("n_generated_tokens_total", 0)
                
                model_stats[model]["requests"] += requests_count
                model_stats[model]["input_tokens"] += context_tokens
                model_stats[model]["output_tokens"] += generated_tokens
                total_requests += requests_count

        except requests.exceptions.RequestException as e:
            # A 404 for a future date is normal, otherwise log error
            if e.response and e.response.status_code == 404 and current_day >= today:
                 logger.info(f"No usage data available for {date_str} (yet).", extra={"tags": "FETCH-NODATA"})
            else:
                logger.error(f"Failed to fetch usage for {date_str}: {e}", extra={"tags": "OPENAI-ERROR"})
            # Continue to next day even if one day fails
        
        current_day += datetime.timedelta(days=1)

    return {
        "total_requests": total_requests,
        "model_stats": dict(model_stats) # Convert back to regular dict
    }

def format_usage_report(usage_data: dict) -> str:
    """Formats the aggregated token usage data into a Markdown report."""
    
    total_requests = usage_data.get("total_requests", 0)
    model_stats = usage_data.get("model_stats", {})

        # Build the Markdown table
    breakdown_table = "| Model | Requests | Input Tokens | Output Tokens | Total Tokens |\n"
    breakdown_table += "|---|---|---|---|---|" + "\n"
    
    total_month_tokens = 0
    for model, stats in sorted(model_stats.items()):
        total_model_tokens = stats["input_tokens"] + stats["output_tokens"]
        total_month_tokens += total_model_tokens
        breakdown_table += f"| {model} | {stats['requests']:,} | {stats['input_tokens']:,} | {stats['output_tokens']:,} | {total_model_tokens:,} |\n"

    report = f"""
---
title: \"Stan Wykorzystania API OpenAI\"
tags:
  - automation
  - report
  - finance
---

# Stan Wykorzystania API OpenAI (Tokeny)

> [!INFO] Dane aktualizowane automatycznie
> Ostatnia aktualizacja: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> Uwaga: API rozliczeniowe jest niedostępne. Poniższe dane to suma tokenów z dziennych raportów.

## 📈 Podsumowanie Miesięczne (od {datetime.date.today().replace(day=1).strftime('%Y-%m-%d')})

- **Całkowita liczba zapytań:** {total_requests:,}
- **Całkowita liczba tokenów:** {total_month_tokens:,}

---

## 📊 Podział na modele

{breakdown_table}
"""
    return report

def main():
    if not ProjectConfig.OPENAI_API_KEY:
        logger.warning("OpenAI API Key is missing.", extra={"tags": "CONFIG-WARN"})
        print("Error: OPENAI_API_KEY not set in .env file.")
        return

    logger.info("Fetching OpenAI aggregated monthly usage...", extra={"tags": "FETCH-START"})
    usage_data = get_monthly_token_usage(ProjectConfig.OPENAI_API_KEY)
    
    if not usage_data:
        logger.error("No aggregated usage data retrieved.", extra={"tags": "FETCH-FAIL"})
        print("Could not retrieve aggregated usage data.")
        return

    report_content = format_usage_report(usage_data)
    
    # Save to the root of the Obsidian vault
    filename = "Stan Wykorzystania API OpenAI.md"
    file_path = ProjectConfig.OBSIDIAN_VAULT / filename
    
    try:
        file_path.write_text(report_content, encoding="utf-8")
        print(f"Report successfully saved to: {file_path}")
        logger.info(f"Usage report saved to {file_path}", extra={"tags": "REPORT-SAVED"})
    except Exception as e:
        logger.error(f"Failed to save report to {file_path}: {e}", extra={"tags": "FILE-ERROR"})
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
