import tensorflow_datasets as tfds

def downloadDataset(path=".", download=True):
    ds = tfds.load('celeb_a', data_dir=path, download=download)
    return ds

if __name__ == "__main__":
    downloadDataset()