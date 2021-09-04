from src.gwnet import create_tfrecords

if __name__ == '__main__':
    fpath = '../data/tmp/train/'
    labels_fn = '../data/test/training_labels.csv'
    out_path = '../data/tfrecords/'
    out_task = {}
    out_task[out_path] = 100

    create_tfrecords(fpath, out_task, labels_fn,
                     shuffle=True, batch_size=1024, remove_older=True,
                     image_size=(71, 71), verbose=True)