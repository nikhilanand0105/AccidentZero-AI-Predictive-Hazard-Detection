import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df, title="EDA"):
    print(f"\n[INFO] Performing {title}...")
    df.hist(figsize=(10, 8))
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title(f"{title} - Correlation Matrix")
    plt.show()