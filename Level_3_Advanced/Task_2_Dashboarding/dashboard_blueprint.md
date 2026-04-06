# Dashboard Blueprint

## Title

Customer Churn Insights Dashboard

## Business Question

Which customer segments show the highest churn risk, and what service or plan-related patterns are associated with churn?

## Audience

- Internship reviewers
- Hiring managers
- Business stakeholders

## Visual Layout

### Top Row

- KPI Card: Total Customers
- KPI Card: Churned Customers
- KPI Card: Churn Rate
- KPI Card: Average Total Charges
- KPI Card: Average Customer Service Calls

### Middle Row

- Bar Chart: Churn Rate by International Plan and Voice Mail Plan
- Bar Chart: Churn Rate by Customer Service Calls Band

### Bottom Row

- Bar Chart or Map: Top States by Churn Rate
- Detail Table: State, Customers, Churned Customers, Churn Rate, Average Total Charges

## Interactivity

- Slicer: State
- Slicer: International plan
- Slicer: Voice mail plan
- Slicer: Customer service calls band
- Slicer: Dataset split

## Suggested Power BI Measures

```text
Total Customers = COUNTROWS(dashboard_customer_detail)
Churned Customers = CALCULATE(COUNTROWS(dashboard_customer_detail), dashboard_customer_detail[Churn] = TRUE())
Churn Rate = DIVIDE([Churned Customers], [Total Customers])
Average Total Charges = AVERAGE(dashboard_customer_detail[Total Charges])
Average Customer Service Calls = AVERAGE(dashboard_customer_detail[Customer service calls])
```

## Suggested Color Logic

- Churn / risk visuals: red scale
- Stable / retained visuals: blue or teal scale
- Neutral supporting visuals: gray

## Submission Tip

When you build the dashboard in Power BI or Tableau, export at least one screenshot and place it in this task folder before the final portfolio submission.
