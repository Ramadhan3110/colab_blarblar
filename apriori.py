import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

# Load dataset
df = pd.read_csv("bread basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

df["month"] = df['date_time'].dt.month
df["day"] = df['date_time'].dt.weekday

df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], inplace = True)
df["day"].replace([i for i in range(6 + 1)], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], inplace = True)

st.title("Market Basket Analysis Menggunakan Algoritma Apriori")

def get_data(period_day = '', weekday_weekend = '', month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["period_day"].str.contains(period_day)) &
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

def user_input_features():
    item = st.selectbox("Item", items_list)
    period_day = st.selectbox("Period Day", ["morning", "afternoon", "evening", "night"])
    weekday_weekend = st.selectbox("Weekday/Weekend", ["weekday", "weekend"])
    month = st.select_slider("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    day = st.select_slider("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],value="Saturday")

    return period_day, weekday_weekend, month, day, item

period_day, weekday_weekend, month, day, item = user_input_features()

data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

if type(data) != type("No Result!"):
    try:
        item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
        item_count_pivot = item_count.pivot_table(index="Transaction", columns="Item", values="Count", aggfunc="sum").fillna(0)
        item_count_pivot = item_count_pivot.applymap(encode)

        support = 0.01
        frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

        if len(frequent_items) > 0:
            metric = "lift"
            min_threshold = 1
            
            rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
            rules.sort_values('confidence', ascending=False, inplace=True)

            def return_item_df(item_antecedents):
                data_rules = rules[["antecedents", "consequents"]].copy()

                data_rules["antecedents"] = data_rules["antecedents"].apply(parse_list)
                data_rules["consequents"] = data_rules["consequents"].apply(parse_list)

                filtered = data_rules.loc[data_rules["antecedents"] == item_antecedents]
                if filtered.shape[0] > 0:
                    return list(filtered.iloc[0,:])
                else:
                    return ["No recommendation", "No recommendation"]

            st.markdown("### Hasil Rekomendasi : ")
            recommendation = return_item_df(item)
            if recommendation[1] != "No recommendation":
                st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{recommendation[1]}** secara bersamaan")
            else:
                st.warning(f"Tidak ada rekomendasi untuk item **{item}** dengan filter yang dipilih.")
        else:
            st.warning(f"Tidak ada frequent itemset yang ditemukan dengan support {support}. Coba ubah filter atau kurangi nilai support.")
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")
        st.info("Pastikan data yang difilter memiliki transaksi yang cukup untuk analisis.")
else:
    st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih. Silakan coba kombinasi filter lain.")