from models import Basic_GRU
from Process.feature import generate_data


if __name__ == '__main__':
    path = '/home/range/Data/MusicFeature/GTZAN'
    x, y = generate_data(path)
