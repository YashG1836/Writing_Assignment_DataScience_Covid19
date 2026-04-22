# Writing Assignment: Data Science (COVID-19)

This repository contains a data science writing assignment that investigates COVID-19 outcomes with a focus on inequality across countries and Indian states.

## Project Overview

The assignment analyzes public COVID-19 data and presents:
- a project introduction and context
- data summary and preprocessing
- two hypotheses with analysis and findings
- concluding insights

Main source content is stored in `Reseach.md`, and section-based HTML pages render selected parts of that content for easier reading.

## Repository Structure

- `index.html` – landing page linking all sections
- `introduction.html` – introduction section
- `data-summary.html` – data summary section
- `hypothesis-1.html` – first hypothesis analysis
- `hypothesis-2.html` – second hypothesis analysis
- `Reseach.md` – full notebook-style write-up used by the section pages
- `Reseach_files/` and `output_*.png` – generated visual outputs
- `results.html`, `conclusion.html` – additional exported pages

## How to View Locally

Because section pages fetch `Reseach.md`, run a local server instead of opening files directly:

```bash
cd <repository-root>
python -m http.server 8000
```

Then open:
- `http://localhost:8000/index.html`

## Notes

- This is a static, assignment-style project (no package manager or build pipeline required).
- External CDN loading is used for Markdown rendering (`marked`).
