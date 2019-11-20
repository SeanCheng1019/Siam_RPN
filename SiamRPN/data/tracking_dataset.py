class Dataset(object):
    def __init__(self, name, dataset_path):
        self.name = name
        self.dataset_path = dataset_path
        self.videos = None

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        keys = sorted((list(self.videos.keys)))
        for key in keys:
            yield self.videos[key]



