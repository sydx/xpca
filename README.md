# xpca
Implementations of extended PCA methods, such as IPCA and EWMPCA.

## Installation

<pre>
pip install xpca
</pre>

## Paper

See *Iterated and exponentially weighted moving principal component analysis* on arXiv.

## Usage

### Iterated PCA (IPCA)

<pre>
# df is a pandas DataFrame
first_year = 2008; last_year = 2020
Z_periods_ipca = []
model = xpca.IPCA()
for period in [str(x) for x in range(first_year, last_year+1)]:
    df_period = df.loc[period]
    model.fit(df_period)
    Z_period = model.transform(df_period)
    Z_periods_ipca.append(Z_period)
Z_periods_ipca = np.vstack(Z_periods_ipca)
</pre>

### Exponentially weighted moving PCA (EWMPCA)

<pre>
xs = df.values
ewmpca = xpca.EWMPCA(alpha=.9305)
zs_ewmpca = ewmpca.add_all(xs)
</pre>
