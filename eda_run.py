import os
import pandas as pd

def run_notebook_eda_all_viz(
    df,
    show_plots: bool = True,
    save_plots: bool = False,
    plot_dir: str = "plots",
):
    """
    Runs ALL EDA visualizations that were present in the uploaded .ipynb:
      1) Hist: Price_in_Lakhs
      2) Hist: Size_in_SqFt
      3) Pointplot: Property_Type vs Price_per_SqFt (mean ± sd)
      4) Scatter: Size_in_SqFt vs Price_in_Lakhs
      5) Boxplot outliers: Price_per_SqFt and Size_in_SqFt
      6) Bar with error bars: State vs Price_per_SqFt (mean ± std) [top 10]
      7) ECDF: Price_in_Lakhs by City (top 20 cities by mean price)
      8) Bar: Locality vs Age_of_Property (median) [top 20]
      9) Stacked bar: BHK distribution across top 10 cities (by count)
     10) Line: Year_Built vs Price_in_Lakhs for top 5 expensive localities (by avg PPSF)
     11) Correlation heatmap of numeric features
     12) Pointplot: Nearby_Schools vs Price_per_SqFt (mean ± sd)
     13) Stripplot: Nearby_Hospitals vs Price_per_SqFt
     14) Violin: Facing vs Price_per_SqFt
     15) Boxen: Facing vs Price_per_SqFt

    Parameters
    ----------
    df : pd.DataFrame
    show_plots : bool
    save_plots : bool
    plot_dir : str

    Notes
    -----
    - This function uses safe column checks; plots are skipped if columns are missing.
    - Matches the notebook intent but fixes typos like `plt.tight_la t()` -> `plt.tight_layout()`.
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid", font_scale=1.05)

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    def _finish(fig_name: str):
        """Save + show + close."""
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plot_dir, fig_name), dpi=300, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close()

    # -------------------------------
    # 1) Distribution of property prices
    # -------------------------------
    if "Price_in_Lakhs" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Price_in_Lakhs"].dropna(), kde=True)
        plt.title("Distribution of Property Prices (in Lakhs)")
        plt.xlabel("Price in Lakhs")
        plt.ylabel("Count")
        _finish("01_dist_price_in_lakhs.png")

    # -------------------------------
    # 2) Distribution of property sizes
    # -------------------------------
    if "Size_in_SqFt" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Size_in_SqFt"].dropna(), kde=True)
        plt.title("Distribution of Property Sizes (SqFt)")
        plt.xlabel("Size in SqFt")
        plt.ylabel("Count")
        _finish("02_dist_size_in_sqft.png")

    # -------------------------------
    # 3) Price per SqFt by property type (pointplot mean ± sd)
    # -------------------------------
    if {"Price_per_SqFt", "Property_Type"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.pointplot(
            data=df,
            x="Property_Type",
            y="Price_per_SqFt",
            estimator="mean",
            errorbar="sd" if "errorbar" in sns.pointplot.__code__.co_varnames else None,  # compatibility
            ci="sd" if "errorbar" not in sns.pointplot.__code__.co_varnames else None,
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Price per SqFt by Property Type")
        _finish("03_ppsf_by_property_type.png")

    # -------------------------------
    # 4) Size vs Price (scatter)
    # -------------------------------
    if {"Size_in_SqFt", "Price_in_Lakhs"}.issubset(df.columns):
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            x=df["Size_in_SqFt"],
            y=df["Price_in_Lakhs"],
            alpha=0.6
        )
        plt.title("Size vs Price")
        plt.xlabel("Size in SqFt")
        plt.ylabel("Price in Lakhs")
        _finish("04_size_vs_price_scatter.png")

    # -------------------------------
    # 5) Outliers – boxplots for PPSF and Size
    # -------------------------------
    if "Price_per_SqFt" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(y=df["Price_per_SqFt"].dropna())
        plt.title("Q5: Outliers – Price per SqFt")
        _finish("05_outliers_ppsf_box.png")

    if "Size_in_SqFt" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(y=df["Size_in_SqFt"].dropna())
        plt.title("Q5: Outliers – Property Size (SqFt)")
        _finish("05_outliers_size_box.png")

    # -------------------------------
    # 6) Avg Price per SqFt by State (mean ± std) – bar with error bars (top 10)
    # -------------------------------
    if {"State", "Price_per_SqFt"}.issubset(df.columns):
        state_stats = (
            df.groupby("State")["Price_per_SqFt"]
              .agg(["mean", "std", "count"])
              .sort_values("mean", ascending=False)
        )
        top_n_states = 10
        state_stats_top = state_stats.head(top_n_states)

        plt.figure(figsize=(10, 6))
        x_positions = np.arange(len(state_stats_top))
        means = state_stats_top["mean"].values
        stds = state_stats_top["std"].fillna(0).values

        plt.bar(
            x_positions,
            means,
            yerr=stds,
            capsize=4,
            alpha=0.85
        )
        plt.xticks(x_positions, state_stats_top.index, rotation=45, ha="right")
        plt.ylabel("Avg Price per SqFt")
        plt.title("Q6: Average Price per SqFt by State (Mean ± Std)")
        _finish("06_state_ppsf_mean_std.png")

    # -------------------------------
    # 7) ECDF of Property Prices by City (Top 20 cities by mean price)
    # -------------------------------
    if {"City", "Price_in_Lakhs"}.issubset(df.columns):
        top_cities = (
            df.groupby("City")["Price_in_Lakhs"]
              .mean()
              .sort_values(ascending=False)
              .head(20)
        )

        plt.figure(figsize=(12, 6))
        for city in top_cities.index:
            sns.ecdfplot(
                data=df[df["City"] == city],
                x="Price_in_Lakhs",
                label=city,
                alpha=0.7
            )
        plt.title("Q7: ECDF of Property Prices by City (Top 20 Cities)")
        plt.xlabel("Price in Lakhs")
        plt.ylabel("ECDF")
        plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc="upper left")
        _finish("07_ecdf_price_by_city_top20.png")

    # -------------------------------
    # 8) Median Age of Properties by Locality (Top 20)
    # -------------------------------
    if {"Locality", "Age_of_Property"}.issubset(df.columns):
        locality_age = (
            df.groupby("Locality")["Age_of_Property"]
              .median()
              .sort_values(ascending=False)
        )
        top_locality_age = locality_age.head(20)

        plt.figure(figsize=(12, 6))
        top_locality_age.plot(kind="bar")
        plt.title("Q8: Median Age of Properties by Locality (Top 20)")
        plt.ylabel("Median Age of Property (Years)")
        plt.xticks(rotation=45, ha="right")
        _finish("08_median_age_by_locality_top20.png")

    # -------------------------------
    # 9) BHK distribution across top 10 cities – stacked bar
    # -------------------------------
    if {"City", "BHK"}.issubset(df.columns):
        top_cities_by_count = df["City"].value_counts().head(10).index
        df_bhk = df[df["City"].isin(top_cities_by_count)]

        bhk_city = (
            df_bhk.groupby(["City", "BHK"])
                  .size()
                  .reset_index(name="Count")
        )
        bhk_pivot = bhk_city.pivot(index="City", columns="BHK", values="Count").fillna(0)

        plt.figure(figsize=(12, 6))
        bhk_pivot.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Q9: BHK Distribution Across Top 10 Cities")
        plt.ylabel("Number of Properties")
        plt.xticks(rotation=45, ha="right")
        _finish("09_bhk_stacked_bar_top10_cities.png")

    # -------------------------------
    # 10) Price trends by Year_Built for top 5 expensive localities (by avg PPSF)
    # -------------------------------
    if {"Locality", "Price_per_SqFt", "Year_Built", "Price_in_Lakhs"}.issubset(df.columns):
        top_localities = (
            df.groupby("Locality")["Price_per_SqFt"]
              .mean()
              .sort_values(ascending=False)
              .head(5)
              .index
        )
        df_top_loc = df[df["Locality"].isin(top_localities)].copy()

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_top_loc,
            x="Year_Built",
            y="Price_in_Lakhs",
            hue="Locality",
            estimator="mean",
            errorbar=None
        )
        plt.title("Q10: Price Trends by Year for Top 5 Expensive Localities")
        plt.ylabel("Avg Price (Lakhs)")
        plt.xlabel("Year Built")
        _finish("10_price_trends_top5_localities.png")

    # -------------------------------
    # 11) Correlation heatmap (numeric)
    # -------------------------------
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
        plt.title("Q11: Correlation Heatmap (Numeric Features)")
        _finish("11_corr_heatmap_numeric.png")

    # -------------------------------
    # 12) Nearby Schools vs PPSF (mean ± sd) – pointplot
    # -------------------------------
    if {"Nearby_Schools", "Price_per_SqFt"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.pointplot(
            data=df,
            x="Nearby_Schools",
            y="Price_per_SqFt",
            estimator="mean",
            errorbar="sd",
            markers="o",
            linestyles="-"
        )
        plt.title("Q12: Nearby Schools vs Price per SqFt (Mean ± SD)")
        plt.xlabel("Number of Nearby Schools")
        plt.ylabel("Price per SqFt")
        _finish("12_schools_vs_ppsf_pointplot.png")

    # -------------------------------
    # 13) Nearby Hospitals vs PPSF – stripplot
    # -------------------------------
    if {"Nearby_Hospitals", "Price_per_SqFt"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.stripplot(
            data=df,
            x="Nearby_Hospitals",
            y="Price_per_SqFt",
            jitter=True,
            alpha=0.5
        )
        plt.title("Q13: Nearby Hospitals vs Price per SqFt")
        plt.xlabel("Number of Nearby Hospitals")
        plt.ylabel("Price per SqFt")
        _finish("13_hospitals_vs_ppsf_stripplot.png")

    # -------------------------------
    # 14) Facing vs PPSF – violin
    # (Notebook had a mismatched if-check; we run it if Facing+PPSF exist)
    # -------------------------------
    if {"Facing", "Price_per_SqFt"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x="Facing", y="Price_per_SqFt", inner="quartile")
        plt.title("Price per SqFt by Facing (Violin Plot)")
        plt.xticks(rotation=45, ha="right")
        _finish("14_facing_vs_ppsf_violin.png")

    # -------------------------------
    # 15) Facing vs PPSF – boxen
    # -------------------------------
    if {"Facing", "Price_per_SqFt"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.boxenplot(data=df, x="Facing", y="Price_per_SqFt")
        plt.title("Q15: Price per SqFt by Property Facing Direction")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Price per SqFt")
        _finish("15_facing_vs_ppsf_boxen.png")

    print("✅ Notebook EDA visualizations completed (plots skipped automatically if columns missing).")

if __name__ == "__main__":
    DATA_PATH = "data/india_housing_prices.csv"
    PLOT_DIR = "plots"

    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    run_notebook_eda_all_viz(
        df=df,
        show_plots=False,   # True = interactive
        save_plots=True,    # THIS creates files
        plot_dir=PLOT_DIR
    )

    print("✅ EDA completed. Plots saved to:", PLOT_DIR)
