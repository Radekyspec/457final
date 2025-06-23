import pandas as pd

df_co2 = pd.read_csv('owid-co2-data.csv')
df_gdp_raw = pd.read_csv('world_bank_gdp.csv')

year_cols = [col for col in df_gdp_raw.columns if col.isdigit()]

df_gdp_long = df_gdp_raw.melt(
    id_vars=['Country Name', 'Country Code'],
    value_vars=year_cols,
    var_name='year',
    value_name='gdp'
)
df_gdp_long = df_gdp_long.rename(
    columns={
        'Country Name': 'country',
        'Country Code': 'country_code'
    }
)
df_gdp_long['year'] = df_gdp_long['year'].astype(int)
df_gdp_long = df_gdp_long.rename(columns={'gdp': 'gdp_wb'})

merged = pd.merge(
    df_co2,
    df_gdp_long[['country', 'year', 'gdp_wb']],
    on=['country', 'year'],
    how='left'
)
merged['gdp'] = merged['gdp_wb'].combine_first(merged['gdp'])
merged = merged.drop(columns=['gdp_wb'])
merged.to_csv('owid-co2-with-gdp-updated.csv', index=False)

