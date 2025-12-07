#!/bin/bash
set -e

# Ensure we are in the repo root
# Get the directory of the script, then go up one level
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT="documents/reports/final_project_report/final_project_report_v2.md"
OUTPUT="documents/reports/final_project_report/final_project_report_v2.pdf"

echo "Building PDF from $INPUT..."

# 1. Check for Pandoc
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed."
    echo "Please install pandoc to continue."
    exit 1
fi

# 2. Setup Mermaid Filter (for diagrams)
USE_FILTER=false
FILTER_CMD=""

if command -v npm &> /dev/null; then
    # Check if mermaid-filter is available locally
    if [ -x "./node_modules/.bin/mermaid-filter" ]; then
        FILTER_CMD="./node_modules/.bin/mermaid-filter"
        USE_FILTER=true
    else
        echo "Mermaid filter not found locally. Attempting to install..."
        # Try to install locally (don't save to package.json to avoid clutter)
        if npm install mermaid-filter --no-save; then
            FILTER_CMD="./node_modules/.bin/mermaid-filter"
            USE_FILTER=true
        else
            echo "Warning: Failed to install mermaid-filter. Diagrams will be rendered as code blocks."
        fi
    fi
else
    echo "Warning: npm not found. Cannot process Mermaid diagrams. They will appear as code blocks."
fi

# 3. Build Command
PANDOC_CMD=(pandoc "$INPUT" -o "$OUTPUT" --pdf-engine=xelatex --variable geometry:margin=1in --variable fontsize=11pt --highlight-style=pygments)

if [ "$USE_FILTER" = true ]; then
    echo "Using mermaid-filter for diagrams..."
    # We add the filter. Note: mermaid-filter needs to be executable.
    # We also explicitly tell mermaid-filter where to find puppeteer config if needed, 
    # but usually defaults work if dependencies are met.
    PANDOC_CMD+=(-F "$FILTER_CMD")
fi

# 4. Execute
echo "Running: ${PANDOC_CMD[*]}"
if "${PANDOC_CMD[@]}"; then
    echo "✅ Success! Report knitted to: $OUTPUT"
else
    echo "❌ PDF Build failed (likely due to missing or broken TeX environment)."
    
    # Fallback to HTML
    OUTPUT_HTML="${OUTPUT%.pdf}.html"
    echo "Attempting to build HTML report as fallback..."
    
    # Basic HTML command
    PANDOC_HTML_CMD=(pandoc "$INPUT" -o "$OUTPUT_HTML" --standalone --toc --metadata title="Final Project Report")
    
    if [ "$USE_FILTER" = true ]; then
        PANDOC_HTML_CMD+=(-F "$FILTER_CMD")
    fi
    
    echo "Running: ${PANDOC_HTML_CMD[*]}"
    if "${PANDOC_HTML_CMD[@]}"; then
        echo "⚠️  PDF generation failed, but HTML report was generated successfully."
        echo "📄 HTML Report: $OUTPUT_HTML"
        echo "👉 You can open this HTML file in a browser and 'Print to PDF' to get your PDF."
    else
        echo "❌ HTML Build also failed."
        exit 1
    fi
fi
