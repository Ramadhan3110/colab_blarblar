import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

# Load dataset
df = pd.read_csv("bread basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

df["month"] = df['date_time'].dt.month
df["day"] = df['date_time'].dt.weekday

df["month"].replace(
    [i for i in range(1, 13)],
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    inplace=True
)

df["day"].replace(
    [i for i in range(0, 7)],
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    inplace=True
)

st.title("Market Basket Analysis Menggunakan Algoritma Apriori")

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data["period_day"].str.contains(period_day)) &
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

def user_input_features():
    item = st.selectbox("Item", [
        'Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies', 'Muffin',
        'Coffee', 'Pastry', 'Medialuna', 'Tea', 'Tartine', 'Basket',
        'Mineral water', 'Farm House', 'Fudge', 'Juice', "Ella's Kitchen Pouches",
        'Victorian Sponge', 'Frittata', 'Hearty & Seasonal', 'Soup',
        'Pick and Mix Bowls', 'Smoothies', 'Cake', 'Mighty Protein',
        'Chicken sand', 'Coke', 'My-5 Fruit Shoot', 'Focaccia', 'Sandwich',
        'Alfajores', 'Eggs', 'Brownie', 'Dulce de Leche', 'Honey', 'The BART',
        'Granola', 'Fairy Doors', 'Empanadas', 'Keeping It Local', 'Art Tray',
        'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles',
        'Chimichurri Oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings',
        'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta', 'Polenta',
        'The Nomad', 'Hack the stack', 'Bakewell', 'Lemon and coconut', 'Toast',
        'Scone', 'Crepes', 'Vegan mincepie', 'Bare Popcorn', 'Muesli', 'Crisps',
        'Pintxos', 'Gingerbread syrup', 'Panatone', 'Brioche and salami',
        'Afternoon with the baker', 'Salad', 'Chicken Stew', 'Spanish Brunch',
        'Raspberry shortbread sandwich', 'Extra Salami or Feta', 'Duck egg',
        'Baguette', "Valentine's card", 'Tshirt', 'Vegan Feast', 'Postcard',
        'Nomad bag', 'Chocolates', 'Coffee granules', 'Drinking chocolate spoons',
        'Christmas common', 'Argentina Night', 'Half slice Monster',
        'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars',
        'Tacos/Fajita'
    ])
    
    period_day = st.selectbox('Period Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox('Weekday / Weekend', ['Weekend', 'Weekday'])
    month = st.select_slider("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.select_slider('Day', ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value='Sat')
    
    return period_day, weekday_weekend, month, day, item

period_day, weekday_weekend, month, day, item = user_input_features()
data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode(x):
    return 1 if x >= 1 else 0

rules = None
if type(data) != str:
    item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)
    
    frequent_items = apriori(item_count_pivot, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    return x[0] if len(x) == 1 else ", ".join(x)

def return_item_df(item_antecedents):
    if rules is None or rules.empty:
        return "Tidak ada rule ditemukan."
    
    temp = rules[["antecedents", "consequents"]].copy()
    temp["antecedents"] = temp["antecedents"].apply(parse_list)
    temp["consequents"] = temp["consequents"].apply(parse_list)
    
    if item_antecedents not in temp["antecedents"].values:
        return "Tidak ada rekomendasi."
    
    row = temp.loc[temp["antecedents"] == item_antecedents].iloc[0]
    return row["consequents"]

if type(data) != str:
    st.markdown("### Hasil Rekomendasi :")
    rekom = return_item_df(item)
    st.success(f"Jika konsumen membeli **{item}**, maka membeli **{rekom}**")
else:
    st.warning("Data tidak ditemukan sesuai filter yang dipilih.")
