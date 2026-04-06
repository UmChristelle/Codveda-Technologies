from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "churn_train.csv"
TEST_PATH = BASE_DIR / "data" / "churn_test.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def run_dashboard_prep() -> pd.DataFrame:
    """Prepare dashboard-ready churn data and summary tables."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    train_df = pd.read_csv(TRAIN_PATH).assign(dataset="train")
    test_df = pd.read_csv(TEST_PATH).assign(dataset="test")
    df = pd.concat([train_df, test_df], ignore_index=True)

    df["Churn Label"] = df["Churn"].map({False: "No", True: "Yes"})
    df["International Plan Flag"] = df["International plan"].map({"No": "No International Plan", "Yes": "International Plan"})
    df["Voice Mail Flag"] = df["Voice mail plan"].map({"No": "No Voice Mail", "Yes": "Voice Mail"})
    df["Customer Service Calls Band"] = pd.cut(
        df["Customer service calls"],
        bins=[-1, 1, 3, 10],
        labels=["0-1 Calls", "2-3 Calls", "4+ Calls"],
    )
    df["Total Charges"] = (
        df["Total day charge"]
        + df["Total eve charge"]
        + df["Total night charge"]
        + df["Total intl charge"]
    ).round(2)

    df.to_csv(OUTPUTS_DIR / "dashboard_customer_detail.csv", index=False)

    total_customers = len(df)
    churned_customers = int(df["Churn"].sum())
    churn_rate = round(churned_customers / total_customers, 4)
    avg_total_charge = round(df["Total Charges"].mean(), 2)
    avg_customer_service_calls = round(df["Customer service calls"].mean(), 2)

    kpi_df = pd.DataFrame(
        {
            "metric": [
                "Total Customers",
                "Churned Customers",
                "Churn Rate",
                "Average Total Charges",
                "Average Customer Service Calls",
            ],
            "value": [
                total_customers,
                churned_customers,
                churn_rate,
                avg_total_charge,
                avg_customer_service_calls,
            ],
        }
    )
    kpi_df.to_csv(OUTPUTS_DIR / "kpi_summary.csv", index=False)

    state_summary = (
        df.groupby("State")
        .agg(
            customers=("Churn", "size"),
            churned_customers=("Churn", "sum"),
            churn_rate=("Churn", "mean"),
            avg_total_charge=("Total Charges", "mean"),
            avg_customer_service_calls=("Customer service calls", "mean"),
        )
        .sort_values(by=["churn_rate", "customers"], ascending=[False, False])
        .round({"churn_rate": 4, "avg_total_charge": 2, "avg_customer_service_calls": 2})
        .reset_index()
    )
    state_summary.to_csv(OUTPUTS_DIR / "state_summary.csv", index=False)

    plan_summary = (
        df.groupby(["International Plan Flag", "Voice Mail Flag"])
        .agg(
            customers=("Churn", "size"),
            churned_customers=("Churn", "sum"),
            churn_rate=("Churn", "mean"),
            avg_total_charge=("Total Charges", "mean"),
        )
        .round({"churn_rate": 4, "avg_total_charge": 2})
        .reset_index()
    )
    plan_summary.to_csv(OUTPUTS_DIR / "plan_summary.csv", index=False)

    service_call_summary = (
        df.groupby("Customer Service Calls Band", observed=False)
        .agg(
            customers=("Churn", "size"),
            churned_customers=("Churn", "sum"),
            churn_rate=("Churn", "mean"),
        )
        .round({"churn_rate": 4})
        .reset_index()
    )
    service_call_summary.to_csv(OUTPUTS_DIR / "service_call_summary.csv", index=False)

    dataset_summary = (
        df.groupby("dataset")
        .agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"))
        .round({"churn_rate": 4})
        .reset_index()
    )
    dataset_summary.to_csv(OUTPUTS_DIR / "dataset_split_summary.csv", index=False)

    dashboard_notes = (
        "Recommended Dashboard Pages\n"
        "1. Executive Overview\n"
        "2. Churn Drivers and Customer Segments\n"
        "3. Geographic Churn Analysis\n\n"
        "Recommended Slicers\n"
        "- State\n"
        "- International plan\n"
        "- Voice mail plan\n"
        "- Customer Service Calls Band\n"
        "- dataset (train/test)\n"
    )
    (OUTPUTS_DIR / "dashboard_notes.txt").write_text(dashboard_notes, encoding="utf-8")

    top_states = state_summary.head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_states, x="State", y="churn_rate", color="#e74c3c")
    plt.title("Top 10 States by Churn Rate", fontweight="bold")
    plt.xlabel("State")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "01_top_states_churn_rate.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plan_summary, x="International Plan Flag", y="churn_rate", hue="Voice Mail Flag", palette="Set2")
    plt.title("Churn Rate by Plan Type", fontweight="bold")
    plt.xlabel("Plan Segment")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "02_plan_segment_churn.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=service_call_summary, x="Customer Service Calls Band", y="churn_rate", color="#3498db")
    plt.title("Churn Rate by Customer Service Call Band", fontweight="bold")
    plt.xlabel("Customer Service Calls Band")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "03_service_calls_churn.png", bbox_inches="tight")
    plt.close()

    print("Dashboard-ready customer rows:", len(df))
    print("Overall churn rate:", churn_rate)
    print("Average total charges:", avg_total_charge)
    print("\nTop churn states:")
    print(state_summary.head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    run_dashboard_prep()
