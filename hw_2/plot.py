import pandas as pd
import matplotlib.pyplot as plt


def plot(df):
    df = df.set_index('length')
    df[['togpu', 'kernel', 'todevice']].plot(kind='bar', stacked=True)
    plt.xlabel('Length')
    plt.xticks(rotation=30, ha="right")
    plt.ylabel('Time')
    plt.title('VecAdd')
    plt.savefig('ex1_times.png')
    # plt.figure(figsize=(30, 10), dpi=100)
    # plt.show()

def plot2(df, title, fn):
    df.set_index('label')
    df[['togpu', 'kernel', 'todevice']].plot(kind='bar', stacked=True)
    plt.xlabel('Matrices')
    plt.xticks(rotation=30, ha="right")
    plt.ylabel('Time')
    plt.title(title)
    plt.savefig(fn)
    # plt.show()

fname = 'ex1.csv'
df = pd.read_csv(fname)
plot(df)

fname = 'ex2.csv'
df = pd.read_csv(fname)
print(df)
label = []
for i, row in df.iterrows():
    label.append(f"{int(row['s1'])},{int(row['s2'])},{int(row['s3'])},{int(row['s4'])}")
df['label'] = label
plot2(df, title='MatMul (double)', fn='ex2.png')

fname = 'ex3.csv'
df = pd.read_csv(fname)
print(df)
label = []
for i, row in df.iterrows():
    label.append(f"{int(row['s1'])},{int(row['s2'])},{int(row['s3'])},{int(row['s4'])}")
df['label'] = label
plot2(df, title='MatMul (float)', fn='ex22.png')
print(df)

df.columns = [
    'Arow', 'Acol',
    'Brow', 'Bcol',
    'togpu', 'kernel', 'todevice', 'label'
]
print(df)