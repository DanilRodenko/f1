import matplotlib.pyplot as plt

# Bar plot for constructor nationality
def plot_constructor_nationality(df):
    nationality_counts = df.groupby('nationality').size().reset_index(name='Count')

    nationality_counts.plot(kind='bar', x='nationality', y='Count', color='skyblue', legend=False)

    plt.xticks(rotation=45)
    plt.xlabel('Nationality')
    plt.ylabel('Count')
    plt.title('Frequency Distribution of Constructors by Nationality')
    plt.tight_layout()
    plt.show()
    
# Generic bar plot
def plot_bar(df, x_col, y_col, title, xlabel, ylabel, color='skyblue', rotate=45, figsize=(14,6)):
    plt.figure(figsize=figsize)
    df.plot(kind='bar', x=x_col, y=y_col, color=color, legend=False)
    plt.xticks(rotation=rotate, ha='right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()