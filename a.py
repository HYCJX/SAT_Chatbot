import pandas as pd

# df = pd.read_csv('TEXTOIR/data/clinc/test.tsv', sep="\t")
# df = df[df["label"].isin(["maybe", "goodbye", "greeting", "are_you_a_bot",
#                          "what_is_your_name", "how_old_are_you", "yes", "no", "what_are_your_hobbies"])]
# print(df.size)
# df.to_csv("c.csv")

df2 = pd.read_csv("test2.csv")
df2 = df2[["Utterance", "Cause"]]
dfa = pd.read_csv("a.csv")
dfb = pd.read_csv("b.csv")
dfc = pd.read_csv("c.csv")
labels = set(df2["Cause"])
print(labels)
for l in labels:
    df_l = df2[df2["Cause"] == l]
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df_l, test_size=0.2)
    print(l)
    print(train.size)

    valid, t = train_test_split(test, test_size=0.5)
    dfa = pd.concat([dfa, train.rename(
        columns={'Cause': 'label', "Utterance": "text"})], ignore_index=True)
    dfb = pd.concat([dfb, valid.rename(
        columns={'Cause': 'label', "Utterance": "text"})], ignore_index=True)
    dfc = pd.concat([dfc, t.rename(
        columns={'Cause': 'label', "Utterance": "text"})], ignore_index=True)
dfa.to_csv("train.csv")
dfb.to_csv("valid.csv")
dfc.to_csv("test.csv")

df = pd.read_csv("train.csv")
print(set(df["label"]))
