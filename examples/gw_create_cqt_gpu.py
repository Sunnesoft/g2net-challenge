from src.gwnet import CQTProcessor, TfDevice
import tensorflow as tf

if __name__ == '__main__':
    input_path = '../data/test/'
    output_path = '../data/tmp/train/'

    tf.debugging.set_log_device_placement(False)

    proc = CQTProcessor(mode=TfDevice.GPU, multidevice_strategy=False)
    print(proc.scan_directory(input_path, output_path, reject_if_exists=False, imitate_loaded=100))
    proc.run(
        batch_size=1024, out_path=output_path, sample_rate=2048,
        qrange=(4, 64), mismatch=0.05, time_range=(0, 2.0, 1e-2),
        freq_range=(50, 250, 1), img_size=(512, 512), shuffle_tasks=True, verbose=True
    )