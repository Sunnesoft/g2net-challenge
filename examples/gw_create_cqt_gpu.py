from src.gwnet import CQTProcessor, TfDevice


if __name__ == '__main__':
    input_path = '../data/test/'
    output_path = '../data/tmp/train/'

    proc = CQTProcessor(mode=TfDevice.GPU)
    print(proc.scan_directory(input_path))
    proc.run(
        batch_size=1024, out_path=output_path, sample_rate=2048,
        qrange=(4, 64), mismatch=0.05, time_range=(0, 2.0, 1e-2),
        freq_range=(50, 250, 1), img_size=(512, 512), shuffle_tasks=True, verbose=True
    )