from tensorflow import keras
from tensorflow.python.lib.io import file_io
import os


def copy_file(filepath, out_dir):
    if os.path.isfile(filepath):
        with file_io.FileIO(filepath, mode="rb") as inp:
            with file_io.FileIO(os.path.join(out_dir, filepath), mode="wb+") as out:
                out.write(inp.read())


class ModelCheckpointInGcs(keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        gcs_dir: str,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
        **kwargs,
    ):
        super().__init__(
            filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options,
            **kwargs,
        )
        self._gcs_dir = gcs_dir

    def _save_model(self, epoch, batch, logs):
        super()._save_model(epoch, batch, logs)
        filepath = self._get_file_path(epoch, logs)
        copy_file(filepath, self._gcs_dir)
