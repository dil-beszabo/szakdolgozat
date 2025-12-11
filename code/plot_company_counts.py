import os
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = "/Users/beszabo/bene/szakdolgozat"
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
PANEL_DIR = os.path.join(DATA_DIR, "panels")

MEMES_CSV = os.path.join(PANEL_DIR, "memes_weekly_activity.csv")
NYT_CSV = os.path.join(PANEL_DIR, "nyt_weekly_sentiment.csv")

os.makedirs(FIG_DIR, exist_ok=True)


def _plot_bar(df_counts: pd.DataFrame, value_col: str, title: str, out_path: str):
    df_plot = df_counts.sort_values(value_col, ascending=False)
    n = len(df_plot)
    plt.figure(figsize=(max(10, n * 0.35), 6))
    plt.bar(df_plot["company"], df_plot[value_col])
    plt.title(title, fontsize=16)
    plt.ylabel(value_col, fontsize=16)
    plt.xticks(rotation=85, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # Load and aggregate memes counts
    memes = pd.read_csv(MEMES_CSV)
    memes_counts = memes.groupby("company", as_index=False)["num_memes"].sum()
    memes_out = os.path.join(FIG_DIR, "descriptive", "company_num_memes.png")
    _plot_bar(memes_counts, "num_memes", "Total memes per company", memes_out)
    print(f"Saved {memes_out}")

    # Load and aggregate NYT article counts
    nyt = pd.read_csv(NYT_CSV)
    articles_counts = nyt.groupby("company", as_index=False)["NYT_mention"].sum()
    articles_out = os.path.join(FIG_DIR, "descriptive", "company_nyt_mentions.png")
    _plot_bar(articles_counts, "NYT_mention", "Total NYT mentions per company", articles_out)
    print(f"Saved {articles_out}")

    # Print quick top-5 summaries
    print("\nTop companies by memes:")
    print(memes_counts.sort_values("num_memes", ascending=False).head(10).to_string(index=False))
    print("\nTop companies by NYT mentions:")
    print(articles_counts.sort_values("NYT_mention", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
