# Level 3 Task 2: Building Dashboards with Power BI/Tableau

## Objective

Prepare a professional dashboard package that can be used to build an interactive churn dashboard in Power BI or Tableau.

## Dataset

- Source datasets:
  - `data/churn_train.csv`
  - `data/churn_test.csv`
- Chosen dashboard domain: customer churn

## What Was Prepared

- Combined dashboard-ready customer-level dataset
- KPI summary table
- State-level churn summary
- Plan-segment churn summary
- Customer-service-call churn summary
- Dashboard notes for layout and slicers
- Supporting charts to guide dashboard design

## Why Churn Was Chosen

- It supports strong business KPIs and executive storytelling
- It has meaningful segmentation variables such as state, international plan, voice mail plan, and customer service calls
- It works very well for filters, slicers, and comparative visuals in Power BI/Tableau

## Recommended Dashboard Structure

### Page 1: Executive Overview

- KPI cards:
  - Total Customers
  - Churned Customers
  - Churn Rate
  - Average Total Charges
  - Average Customer Service Calls
- Visuals:
  - Churn rate by plan type
  - Churn rate by customer service call band
  - Dataset split summary

### Page 2: Customer Segments

- Visuals:
  - Churn by international plan
  - Churn by voice mail plan
  - Scatter or bar view of charges versus churn segments
  - Service call band comparison

### Page 3: Geographic Analysis

- Visuals:
  - State churn ranking
  - Map or filled map by state
  - Drill-down table for state-level customer and churn counts

## Recommended Filters / Slicers

- State
- International plan
- Voice mail plan
- Customer service calls band
- Dataset split (`train` / `test`)

## Power BI / Tableau Build Note

- This environment cannot create `.pbix` or Tableau workbook files directly.
- Instead, the repo contains a dashboard-ready data layer and a professional dashboard blueprint so you can build the final interactive dashboard quickly in Power BI or Tableau.

## Files

- `task2_dashboard_prep.py`
- `outputs/dashboard_customer_detail.csv`
- `outputs/kpi_summary.csv`
- `outputs/state_summary.csv`
- `outputs/plan_summary.csv`
- `outputs/service_call_summary.csv`
- `outputs/dataset_split_summary.csv`
- `outputs/dashboard_notes.txt`
- `outputs/01_top_states_churn_rate.png`
- `outputs/02_plan_segment_churn.png`
- `outputs/03_service_calls_churn.png`
