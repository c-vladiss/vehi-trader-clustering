import matplotlib.pyplot as plt


def plot_clusters_2d(df, x, y):
    plt.figure(figsize=(15, 8))
    plt.scatter(df[x], df[y], c=df['Cluster'], cmap='viridis')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Clusters of Cars')
    plt.colorbar(label='Cluster')
    plt.show()

def plot_clusters_3d(df, x, y, z):
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection='3d')
    scatter = ax.scatter3D(df[x], df[y], df[z], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title('Clusters of Cars')
    fig.colorbar(scatter, ax=ax, label='Cluster')
    plt.show()

