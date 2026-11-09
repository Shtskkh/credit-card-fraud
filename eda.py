import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo
    import polars as pl
    import polars.selectors as cs
    import kagglehub

    path = kagglehub.competition_download(
        "playground-series-s6e11", force_download=True, output_dir="data/"
    )

    train_df = pl.read_csv(f"{path}/train.csv")


@app.cell
def _():
    train_df.head()
    return


@app.cell
def _():
    print(train_df.shape)
    return


@app.cell
def _():
    _schema = zip(train_df.columns, train_df.dtypes)

    for _col in _schema:
        print(_col)
    return


@app.cell
def _():
    train_df.get_column("is_fraud").value_counts()
    return


@app.cell
def _():
    train_df.null_count()
    return


@app.cell
def _():
    train_df.filter(train_df.is_duplicated())
    return


@app.cell
def _():
    for _str_col in train_df.select(cs.string()):
        print(_str_col.name, _str_col.n_unique())
    return


if __name__ == "__main__":
    app.run()
