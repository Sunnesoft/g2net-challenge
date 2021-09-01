from src.gwnet import GwEfficientNetB0, TfDevice

if __name__ == '__main__':
    data_path = '../data/tfrecords/'

    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics = ['AUC']

    solver = GwEfficientNetB0('eff_net_b0', TfDevice.CPU, (71, 71, 1), 255.0)
    solver.load_train_dataset(data_path, batch_size=64)
    solver.show_random_train_batch(subs={'1': 'GW_TRUE', '0': 'GW_FALSE'})

    solver.compile()
    solver.print_model()

    history = solver.fit(epochs=20)