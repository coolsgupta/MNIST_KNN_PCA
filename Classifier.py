import struct
import numpy as np
import sys
from sklearn.decomposition import PCA


class Constants:
    path_images = 'train-images.idx3-ubyte'
    path_labels = 'train-labels.idx1-ubyte'


class PreprocessData:
    @classmethod
    def get_images(cls, path):
        with open(path, 'rb') as f:
            # loading headers
            magic_number, num_images = struct.unpack('>II', f.read(8))
            img_size = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, img_size[0], img_size[1])[:800]

    @classmethod
    def get_labels(cls, path):
        with open(path, 'rb') as f:
            # loading headers
            magic_number, num_labels = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)[:800]

    @classmethod
    def reshape_flatten_images(cls, images):
        return [x.flatten() for x in images]

    @classmethod
    def train_test_split(cls, images, labels, test_num):
        return images[test_num:], labels[test_num:], images[:test_num], labels[:test_num]


class DataTransformation:
    def __init__(self, fit_data, dimensions):
        self.model = PCA(n_components=dimensions, svd_solver='full').fit(fit_data)

    def apply_pca_transformation(self, train_images, test_images):
        return self.model.transform(train_images), self.model.transform(test_images)


class KNNClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def calc_distance(self, point):
        dist_dict = {}
        for i, sample in enumerate(self.train_images):
            dist_dict[i] = sum([(point[x] - sample[x])**2 for x in range(len(point))])
        return dist_dict

    def calc_vote(self, point):
        dist_map = {k: v for k, v in sorted(self.calc_distance(point).items(), key=lambda item: item[1])}
        return sorted([(index, 1/dist) for index, dist in dist_map.items()], reverse=True, key=lambda x: x[1])

    def make_predictions(self, test_data):
        predictions = []
        for point in test_data:
            vote_map = self.calc_vote(point)[:K]
            vote_map = [(train_labels[x[0]], x[1]) for x in vote_map]
            knn_labels = [x[0] for x in vote_map]
            ans = max(set(knn_labels), key=knn_labels.count)
            predictions.append(ans)

        return predictions

    def write_results(self, predictions, test_labels):
        results = [' '.join([str(pred), str(label)]) for pred, label in zip(predictions, test_labels)]
        with open('results.txt', 'w') as result_file:
            result_file.write('\n'.join(results))


if __name__ == '__main__':
    # read arguments
    K, D, N = map(int, sys.argv[1:4])
    PATH_TO_DATA_DIR = sys.argv[4]

    # task_1 : preprocess data
    images = PreprocessData.reshape_flatten_images(PreprocessData.get_images('/'.join([PATH_TO_DATA_DIR, Constants.path_images])))
    labels = PreprocessData.get_labels('/'.join([PATH_TO_DATA_DIR, Constants.path_labels]))
    train_images, train_labels, test_images, test_labels = PreprocessData.train_test_split(images, labels, N)
    avg_first_image = np.average(images[0])

    # task_2: PCA
    transformed_train_images, transformed_test_images = DataTransformation(train_images, D).apply_pca_transformation(train_images, test_images)

    # task_3: KNN
    classifier = KNNClassifier(transformed_train_images, train_labels)
    predictions = classifier.make_predictions(transformed_test_images)
    classifier.write_results(predictions, test_labels)

    correct = 0
    for x, y in zip(predictions, test_labels):
        if x==y:
            correct += 1
    print("     ".join(map(str, [correct, correct/N])))


    print('Done')

